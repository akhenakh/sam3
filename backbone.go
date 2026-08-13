// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
)

// axialCis computes the real/imaginary parts of the 2D axial RoPE frequencies,
// each of shape [size*size, headDim/2]. ropePTSize is the pretrained RoPE grid
// size (24); scalePos = ropePTSize/size performs interpolation.
func axialCis(g *Graph, headDim, size, ropePTSize int) (cos, sin *Node) {
	scalePos := float64(ropePTSize) / float64(size)
	quarter := headDim / 4

	idx := IotaFull(g, shapes.Make(dtypes.Float32, quarter))
	freqs := Reciprocal(Pow(ConstAs(idx, 10000.0), DivScalar(idx, float64(quarter))))

	t := IotaFull(g, shapes.Make(dtypes.Float32, size*size))
	tx := MulScalar(ModScalar(t, float64(size)), scalePos)
	ty := MulScalar(Floor(DivScalar(t, float64(size))), scalePos)

	freqsX := Mul(ExpandAxes(tx, -1), Reshape(freqs, 1, quarter))
	freqsY := Mul(ExpandAxes(ty, -1), Reshape(freqs, 1, quarter))

	cos = Concatenate([]*Node{Cos(freqsX), Cos(freqsY)}, -1)
	sin = Concatenate([]*Node{Sin(freqsX), Sin(freqsY)}, -1)
	return cos, sin
}

// applyRoPE applies the rotary-position encoding to q of shape
// [batch, heads, L, headDim] using precomputed cos/sin of shape [L, headDim/2].
func applyRoPE(q, cos, sin *Node) *Node {
	dims := q.Shape().Dimensions
	b, h, l, headDim := dims[0], dims[1], dims[2], dims[3]
	half := headDim / 2

	q = Reshape(q, b, h, l, half, 2)
	real := Squeeze(Slice(q, AxisRange(), AxisRange(), AxisRange(), AxisRange(), AxisElem(0)), 4)
	imag := Squeeze(Slice(q, AxisRange(), AxisRange(), AxisRange(), AxisRange(), AxisElem(1)), 4)

	cosB := Reshape(cos, 1, 1, l, half)
	sinB := Reshape(sin, 1, 1, l, half)

	outReal := Sub(Mul(real, cosB), Mul(imag, sinB))
	outImag := Add(Mul(real, sinB), Mul(imag, cosB))

	interleaved := Concatenate([]*Node{ExpandAxes(outReal, -1), ExpandAxes(outImag, -1)}, -1)
	return Reshape(interleaved, b, h, l, headDim)
}

// vitAttention runs a single ViT attention block on a [batch, H, W, dim] input.
// cos/sin are the RoPE frequencies for the current window/global grid.
func vitAttention(scope *model.Scope, x *Node, numHeads int, cos, sin *Node) *Node {
	dims := x.Shape().Dimensions
	b, h, w, dim := dims[0], dims[1], dims[2], dims[3]
	headDim := dim / numHeads
	scale := 1.0 / math.Sqrt(float64(headDim))
	l := h * w

	x2d := Reshape(x, b, l, dim)
	qkv := linear(scope.In("qkv"), x2d) // [b, l, 3*dim]
	qkv = Reshape(qkv, b, l, 3, numHeads, headDim)

	q := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(0), AxisRange(), AxisRange()), 2)
	k := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(1), AxisRange(), AxisRange()), 2)
	v := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(2), AxisRange(), AxisRange()), 2)

	q = TransposeAllAxes(q, 0, 2, 1, 3)
	k = TransposeAllAxes(k, 0, 2, 1, 3)
	v = TransposeAllAxes(v, 0, 2, 1, 3)

	q = applyRoPE(q, cos, sin)
	k = applyRoPE(k, cos, sin)

	out, _ := attention.Core(q, k, v, attention.LayoutBHSD, attention.CoreOptions{Scale: scale})

	out = TransposeAllAxes(out, 0, 2, 1, 3)
	out = Reshape(out, b, l, dim)
	out = linear(scope.In("proj"), out)
	return Reshape(out, b, h, w, dim)
}

// vitBlock runs a single ViT transformer block (window or global attention).
func vitBlock(scope *model.Scope, x *Node, numHeads, windowSize, ropePTSize int) *Node {
	g := x.Graph()
	dim := x.Shape().Dim(-1)
	headDim := dim / numHeads

	h := layerNormLast(scope.In("norm1"), x, 1e-5)

	var attnOut *Node
	if windowSize > 0 {
		ih, iw := h.Shape().Dim(1), h.Shape().Dim(2)
		win, padded := windowPartition(h, windowSize)
		cos, sin := axialCis(g, headDim, windowSize, ropePTSize)
		winAttn := vitAttention(scope.In("attn"), win, numHeads, cos, sin)
		attnOut = windowUnpartition(winAttn, windowSize, padded[0], padded[1], ih, iw)
	} else {
		ih := h.Shape().Dim(1)
		cos, sin := axialCis(g, headDim, ih, ropePTSize)
		attnOut = vitAttention(scope.In("attn"), h, numHeads, cos, sin)
	}

	x = Add(x, attnOut)

	h = layerNormLast(scope.In("norm2"), x, 1e-5)
	mlpScope := scope.In("mlp")
	mlp := linear(mlpScope.In("fc1"), h)
	mlp = activation.Gelu(mlp)
	mlp = linear(mlpScope.In("fc2"), mlp)
	return Add(x, mlp)
}

// windowPartition partitions [b, h, w, c] into non-overlapping windows of
// [b*numWindows, windowSize, windowSize, c], returning the padded spatial dims.
func windowPartition(x *Node, windowSize int) (windows *Node, padded []int) {
	dims := x.Shape().Dimensions
	b, h, w, c := dims[0], dims[1], dims[2], dims[3]

	padH := (windowSize - h%windowSize) % windowSize
	padW := (windowSize - w%windowSize) % windowSize
	if padH > 0 || padW > 0 {
		x = Pad(x, ScalarZero(x.Graph(), x.DType()),
			compute.PadAxis{}, compute.PadAxis{End: padH}, compute.PadAxis{End: padW}, compute.PadAxis{})
	}
	hp, wp := h+padH, w+padW

	x = Reshape(x, b, hp/windowSize, windowSize, wp/windowSize, windowSize, c)
	x = TransposeAllAxes(x, 0, 1, 3, 2, 4, 5)
	windows = Reshape(x, -1, windowSize, windowSize, c)
	return windows, []int{hp, wp}
}

// windowUnpartition merges windows back into [b, h, w, c].
func windowUnpartition(windows *Node, windowSize, hp, wp, h, w int) *Node {
	dims := windows.Shape().Dimensions
	c := dims[3]
	numWindows := (hp / windowSize) * (wp / windowSize)
	b := dims[0] / numWindows

	x := Reshape(windows, b, hp/windowSize, wp/windowSize, windowSize, windowSize, c)
	x = TransposeAllAxes(x, 0, 1, 3, 2, 4, 5)
	x = Reshape(x, b, hp, wp, c)

	if hp > h || wp > w {
		x = Slice(x, AxisRange(), AxisRange(0, h), AxisRange(0, w), AxisRange())
	}
	return x
}

// getAbsPos tiles the absolute positional embedding from its pretraining grid
// (24x24) up to the input grid (h x w).
func getAbsPos(scope *model.Scope, g *Graph, h, w, dim, pretrainSize int) *Node {
	posEmbed := scope.GetVariable("pos_embed").NodeValue(g) // [1, 1+pretrainSize^2, dim]
	// Drop the class token.
	posEmbed = Slice(posEmbed, AxisRange(), AxisRange(1, 1+pretrainSize*pretrainSize), AxisRange())
	posEmbed = Reshape(posEmbed, 1, pretrainSize, pretrainSize, dim)

	tileH := (h + pretrainSize - 1) / pretrainSize
	tileW := (w + pretrainSize - 1) / pretrainSize

	r := Reshape(posEmbed, 1, 1, pretrainSize, 1, pretrainSize, dim)
	b := BroadcastToDims(r, 1, tileH, pretrainSize, tileW, pretrainSize, dim)
	tiled := Reshape(b, 1, tileH*pretrainSize, tileW*pretrainSize, dim)
	return Slice(tiled, AxisRange(), AxisRange(0, h), AxisRange(0, w), AxisRange())
}

// neckLevel applies one ViTDet SimpleFPN level to the backbone output.
func neckLevel(scope *model.Scope, x *Node, scale float64, dModel int) *Node {
	switch scale {
	case 4.0:
		x = convTranspose2d(scope.In("dconv_2x2_0"), x, x.Shape().Dim(1)/2, 2, 2)
		x = activation.Gelu(x)
		x = convTranspose2d(scope.In("dconv_2x2_1"), x, x.Shape().Dim(1)/2, 2, 2)
	case 2.0:
		x = convTranspose2d(scope.In("dconv_2x2"), x, x.Shape().Dim(1)/2, 2, 2)
	case 1.0:
		// No up/down sampling.
	default:
		panic("unsupported neck scale factor")
	}
	x = conv2d(scope.In("conv_1x1"), x, dModel, 1, 1, 0)
	x = conv2d(scope.In("conv_3x3"), x, dModel, 3, 1, 1)
	return x
}

// visionBackboneForward runs the ViT backbone and ViTDet neck, returning the
// FPN features (high-to-low resolution) and their positional encodings.
func visionBackboneForward(scope *model.Scope, x *Node, cfg *Config) (features, posEnc []*Node) {
	g := x.Graph()
	trunk := scope.In("trunk")

	// Patch embedding: [batch, 3, H, W] -> [batch, embed, H/patch, W/patch].
	x = conv2d(trunk.In("patch_embed").In("proj"), x, cfg.Backbone.EmbedDim, cfg.Backbone.PatchSize, cfg.Backbone.PatchSize, 0)
	x = TransposeAllAxes(x, 0, 2, 3, 1) // [batch, H, W, embed]

	h := x.Shape().Dim(1)
	w := x.Shape().Dim(2)
	pretrainSize := cfg.Backbone.PretrainImgSize / cfg.Backbone.PatchSize
	x = Add(x, getAbsPos(trunk, g, h, w, cfg.Backbone.EmbedDim, pretrainSize))

	x = layerNormLast(trunk.In("ln_pre"), x, 1e-5)

	blocks := trunk.In("blocks")
	for i := range cfg.Backbone.Depth {
		global := false
		for _, gb := range cfg.Backbone.GlobalBlocks {
			if i == gb {
				global = true
				break
			}
		}
		ws := 0
		if !global {
			ws = cfg.Backbone.WindowSize
		}
		x = vitBlock(blocks.In("%d", i), x, cfg.Backbone.NumHeads, ws, cfg.Backbone.WindowSize)
	}

	// Backbone output: [batch, embed, H, W].
	x = TransposeAllAxes(x, 0, 3, 1, 2)

	// Neck: apply the scale factors (4, 2, 1).
	features = make([]*Node, 0, len(cfg.Neck.ScaleFactors))
	posEnc = make([]*Node, 0, len(cfg.Neck.ScaleFactors))
	convsScope := scope.In("convs")
	for i, scale := range cfg.Neck.ScaleFactors {
		lvl := neckLevel(convsScope.In("%d", i), x, scale, cfg.Neck.DModel)
		features = append(features, lvl)
		posEnc = append(posEnc, positionEmbeddingSine(g, lvl, cfg.Neck.DModel, 10000.0))
	}
	return features, posEnc
}
