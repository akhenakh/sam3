// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
)

// linear applies an nn.Linear layer whose weight is stored in PyTorch layout
// [out, in] under `scope` as `weight`/`bias`. The weight is transposed on the
// fly so the result equals x @ weight^T + bias.
func linear(scope *model.Scope, x *Node) *Node {
	g := x.Graph()
	w := scope.GetVariable("weight").NodeValue(g) // [out, in]
	bias := variableIfExists(scope, g, "bias")
	return nn.Dense(x, TransposeAllAxes(w, 1, 0), bias, compute.DenseLayoutInputOutputs)
}

// variableIfExists returns the node value of a variable, or nil if absent.
func variableIfExists(scope *model.Scope, g *Graph, name string) *Node {
	v := scope.GetVariable(name)
	if v == nil {
		return nil
	}
	return v.NodeValue(g)
}

// conv2d applies an nn.Conv2d layer (channels-first input) whose weight is
// stored in PyTorch layout [out, in, kh, kw].
func conv2d(scope *model.Scope, x *Node, outChannels, kernelSize, stride, padding int) *Node {
	g := x.Graph()
	w := scope.GetVariable("weight").NodeValue(g) // [out, in, kh, kw]
	out := Convolve(x, w).
		ChannelsAxis(images.ChannelsFirst).
		Strides(stride).
		PaddingPerDim([][2]int{{padding, padding}, {padding, padding}}).
		Done()
	if bias := variableIfExists(scope, g, "bias"); bias != nil {
		out = Add(out, Reshape(bias, 1, outChannels, 1, 1))
	}
	return out
}

// convTranspose2d applies an nn.ConvTranspose2d layer whose weight is stored in
// PyTorch layout [in, out, kh, kw]. Only stride==kernelSize==2 is supported
// (the ViTDet neck deconvolutions).
func convTranspose2d(scope *model.Scope, x *Node, outChannels, kernelSize, stride int) *Node {
	g := x.Graph()
	w := scope.GetVariable("weight").NodeValue(g) // [in, out, kh, kw]
	w = TransposeAllAxes(w, 1, 0, 2, 3)           // [out, in, kh, kw]

	dims := x.Shape().Dimensions
	b, h, wd := dims[0], dims[2], dims[3]

	out := Einsum("bchw,ockl->bohkwl", x, w)
	out = Reshape(out, b, outChannels, h*kernelSize, wd*kernelSize)
	if bias := variableIfExists(scope, g, "bias"); bias != nil {
		out = Add(out, Reshape(bias, 1, outChannels, 1, 1))
	}
	return out
}

// layerNorm applies LayerNorm over the given axes. The gain/offset are read
// from `scope` as `weight`/`bias` (PyTorch LayerNorm naming).
func layerNorm(scope *model.Scope, x *Node, axes []int, eps float64) *Node {
	g := x.Graph()
	gamma := variableIfExists(scope, g, "weight")
	beta := variableIfExists(scope, g, "bias")
	return nn.LayerNorm(x, axes, eps, gamma, beta, nil)
}

// layerNormLast applies LayerNorm over the last axis of x.
func layerNormLast(scope *model.Scope, x *Node, eps float64) *Node {
	return layerNorm(scope, x, []int{x.Rank() - 1}, eps)
}

// groupNorm applies GroupNorm(numGroups, C) to a channels-first [batch, C, H, W]
// tensor. The per-channel scale/offset are read as `weight`/`bias`.
func groupNorm(scope *model.Scope, x *Node, numGroups int, eps float64) *Node {
	g := x.Graph()
	dims := x.Shape().Dimensions
	b, c, h, w := dims[0], dims[1], dims[2], dims[3]
	channelsPerGroup := c / numGroups

	reshaped := Reshape(x, b, numGroups, channelsPerGroup, h, w)
	mean := ReduceAndKeep(reshaped, ReduceMean, 2, 3, 4)
	centered := Sub(reshaped, mean)
	variance := ReduceAndKeep(Square(centered), ReduceMean, 2, 3, 4)
	normalized := Div(centered, Sqrt(Add(variance, ConstAs(x, eps))))
	normalized = Reshape(normalized, b, c, h, w)

	gamma := scope.GetVariable("weight").NodeValue(g) // [c]
	beta := scope.GetVariable("bias").NodeValue(g)    // [c]
	normalized = Mul(normalized, Reshape(gamma, 1, c, 1, 1))
	return Add(normalized, Reshape(beta, 1, c, 1, 1))
}

// embedding gathers rows from a [vocab, dim] embedding table stored as `weight`.
func embedding(scope *model.Scope, idx *Node, vocab, dim int) *Node {
	g := idx.Graph()
	table := scope.GetVariable("weight").NodeValue(g) // [vocab, dim]
	return Gather(table, ExpandAxes(idx, -1))
}

// keyPaddingMaskToAttention converts a PyTorch key-padding mask (bool, true =
// padded/masked) into a GoMLX attention mask (bool, true = attend) broadcastable
// to [batch, heads, qSeq, kvSeq].
func keyPaddingMaskToAttention(kpm *Node, heads int) *Node {
	attend := LogicalNot(kpm) // [batch, kvSeq]
	// [batch, 1, 1, kvSeq]
	return Reshape(attend, attend.Shape().Dimensions[0], 1, 1, attend.Shape().Dimensions[1])
}

// causalMask builds a [1, 1, L, L] boolean mask where a query may attend to
// keys with index <= its own (lower-triangular).
func causalMask(g *Graph, length int) *Node {
	qIdx := IotaFull(g, shapes.Make(dtypes.Int32, length, 1))
	kIdx := IotaFull(g, shapes.Make(dtypes.Int32, 1, length))
	mask := LessOrEqual(kIdx, qIdx) // [length, length]
	return Reshape(mask, 1, 1, length, length)
}

// multiHeadAttention implements PyTorch's nn.MultiheadAttention using the
// checkpoint's variable naming (`in_proj_weight`, `in_proj_bias`, `out_proj`).
//
// q, k, v are [seq, batch, dim] when batchFirst is false, or [batch, seq, dim]
// when true. mask is optional and broadcastable to [batch, heads, qSeq, kvSeq];
// it may be boolean (true = attend) or floating point (additive).
func multiHeadAttention(scope *model.Scope, q, k, v *Node, numHeads int, batchFirst bool, mask *Node) *Node {
	g := q.Graph()
	dim := q.Shape().Dim(-1)
	headDim := dim / numHeads
	scale := 1.0 / math.Sqrt(float64(headDim))

	// Normalize to batch-first [batch, seq, dim].
	if !batchFirst {
		q = TransposeAllAxes(q, 1, 0, 2)
		k = TransposeAllAxes(k, 1, 0, 2)
		v = TransposeAllAxes(v, 1, 0, 2)
	}

	inProjW := scope.GetVariable("in_proj_weight").NodeValue(g) // [3*dim, dim]
	inProjB := scope.GetVariable("in_proj_bias").NodeValue(g)   // [3*dim]
	wT := TransposeAllAxes(inProjW, 1, 0)                       // [dim, 3*dim]

	qW := Slice(wT, AxisRange(), AxisRange(0, dim))
	kW := Slice(wT, AxisRange(), AxisRange(dim, 2*dim))
	vW := Slice(wT, AxisRange(), AxisRange(2*dim, 3*dim))
	qB := Slice(inProjB, AxisRange(0, dim))
	kB := Slice(inProjB, AxisRange(dim, 2*dim))
	vB := Slice(inProjB, AxisRange(2*dim, 3*dim))

	q = nn.Dense(q, qW, qB, compute.DenseLayoutInputOutputs)
	k = nn.Dense(k, kW, kB, compute.DenseLayoutInputOutputs)
	v = nn.Dense(v, vW, vB, compute.DenseLayoutInputOutputs)

	// Reshape to [batch, seq, heads, headDim] then [batch, heads, seq, headDim].
	batch := q.Shape().Dimensions[0]
	qSeq := q.Shape().Dimensions[1]
	kSeq := k.Shape().Dimensions[1]
	q = Reshape(q, batch, qSeq, numHeads, headDim)
	k = Reshape(k, batch, kSeq, numHeads, headDim)
	v = Reshape(v, batch, kSeq, numHeads, headDim)
	q = TransposeAllAxes(q, 0, 2, 1, 3)
	k = TransposeAllAxes(k, 0, 2, 1, 3)
	v = TransposeAllAxes(v, 0, 2, 1, 3)

	out, _ := attention.Core(q, k, v, attention.LayoutBHSD, attention.CoreOptions{
		Scale:         scale,
		AttentionMask: mask,
	})

	// [batch, heads, qSeq, headDim] -> [batch, qSeq, dim]
	out = TransposeAllAxes(out, 0, 2, 1, 3)
	out = Reshape(out, batch, qSeq, dim)

	outProjScope := scope.In("out_proj")
	outProjW := outProjScope.GetVariable("weight").NodeValue(g) // [dim, dim]
	outProjB := outProjScope.GetVariable("bias").NodeValue(g)   // [dim]
	out = nn.Dense(out, TransposeAllAxes(outProjW, 1, 0), outProjB, compute.DenseLayoutInputOutputs)

	if !batchFirst {
		out = TransposeAllAxes(out, 1, 0, 2)
	}
	return out
}

// positionEmbeddingSine computes the PositionEmbeddingSine of a channels-first
// feature map [batch, C, H, W], returning [batch, C, H, W] positional encoding.
func positionEmbeddingSine(g *Graph, x *Node, numPosFeats int, temperature float64) *Node {
	dims := x.Shape().Dimensions
	b, h, w := dims[0], dims[2], dims[3]
	scale := 2 * math.Pi

	yEmbed := IotaFull(g, shapes.Make(dtypes.Float32, h, 1))
	xEmbed := IotaFull(g, shapes.Make(dtypes.Float32, 1, w))
	// 1-indexed positions.
	yEmbed = AddScalar(yEmbed, 1.0)
	xEmbed = AddScalar(xEmbed, 1.0)

	// Normalize.
	eps := 1e-6
	yEmbed = Div(MulScalar(yEmbed, scale), AddScalar(Slice(yEmbed, AxisRange(-1), AxisRange()), eps))
	xEmbed = Div(MulScalar(xEmbed, scale), AddScalar(Slice(xEmbed, AxisRange(), AxisRange(-1)), eps))

	// Frequency bands.
	half := numPosFeats / 2
	dimT := IotaFull(g, shapes.Make(dtypes.Float32, half))
	dimT = Pow(ConstAs(dimT, temperature), DivScalar(MulScalar(Floor(DivScalar(dimT, 2.0)), 2.0), float64(half)))

	posX := Div(ExpandAxes(xEmbed, -1), Reshape(dimT, 1, 1, half)) // [1, w, half]
	posY := Div(ExpandAxes(yEmbed, -1), Reshape(dimT, 1, 1, half)) // [h, 1, half]

	posX = interleaveSinCos(posX)
	posY = interleaveSinCos(posY)

	// [h, w, half] -> [h, w, numPosFeats]
	posX = BroadcastToDims(posX, h, w, half)
	posY = BroadcastToDims(posY, h, w, half)

	pos := Concatenate([]*Node{posY, posX}, -1) // [h, w, numPosFeats]
	pos = ExpandAxes(pos, 0)                    // [1, h, w, C]
	pos = TransposeAllAxes(pos, 0, 3, 1, 2)     // [1, C, h, w]
	return BroadcastToDims(pos, b, numPosFeats, h, w)
}

// interleaveSinCos interleaves sin/cos along the last axis to match PyTorch's
// `stack((x[...,0::2].sin(), x[...,1::2].cos()), dim=-1).flatten(-2)` pattern:
// the result at index 2j is sin(x[2j]) and at 2j+1 is cos(x[2j+1]).
func interleaveSinCos(x *Node) *Node {
	even := Slice(x, AxisRange().Spacer(), AxisRange().Stride(2))
	odd := Slice(x, AxisRange().Spacer(), AxisRange(1).Stride(2))
	stacked := Concatenate([]*Node{ExpandAxes(Sin(even), -1), ExpandAxes(Cos(odd), -1)}, -1)
	dims := x.Shape().Dimensions
	outDims := append(append([]int{}, dims[:len(dims)-1]...), dims[len(dims)-1])
	return Reshape(stacked, outDims...)
}
