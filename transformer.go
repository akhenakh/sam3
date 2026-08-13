// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
)

// boxCxcywhToXyxy converts center-size boxes to corner format along the last
// axis (4 values).
func boxCxcywhToXyxy(x *Node) *Node {
	cx := SliceAxis(x, -1, AxisRange(0, 1))
	cy := SliceAxis(x, -1, AxisRange(1, 2))
	w := SliceAxis(x, -1, AxisRange(2, 3))
	h := SliceAxis(x, -1, AxisRange(3, 4))
	x0 := Sub(cx, MulScalar(w, 0.5))
	y0 := Sub(cy, MulScalar(h, 0.5))
	x1 := Add(cx, MulScalar(w, 0.5))
	y1 := Add(cy, MulScalar(h, 0.5))
	return Concatenate([]*Node{x0, y0, x1, y1}, -1)
}

// inverseSigmoid computes the numerically-stable inverse of the sigmoid.
func inverseSigmoid(x *Node, eps float64) *Node {
	g := x.Graph()
	zero := Scalar(g, x.DType(), 0.0)
	one := Scalar(g, x.DType(), 1.0)
	x = Min(Max(x, zero), one)
	x1 := Max(x, Scalar(g, x.DType(), eps))
	x2 := Max(Sub(one, x), Scalar(g, x.DType(), eps))
	return Log(Div(x1, x2))
}

// mlp applies a PyTorch-style MLP (linear layers with ReLU between them).
// Variables are read from scope/layers/<i>. Optional residual connection and
// a final LayerNorm (read from scope/out_norm) are supported.
func mlp(scope *model.Scope, x *Node, numLayers int, residual, outNorm bool) *Node {
	orig := x
	layersScope := scope.In("layers")
	for i := range numLayers {
		x = linear(layersScope.In("%d", i), x)
		if i < numLayers-1 {
			x = activation.Relu(x)
		}
	}
	if residual {
		x = Add(x, orig)
	}
	if outNorm {
		x = layerNormLast(scope.At("out_norm"), x, 1e-5)
	}
	return x
}

// encoderLayer runs one TransformerEncoderLayer (pre-norm, batch-first).
func encoderLayer(scope *model.Scope, tgt, queryPos, memory, keyPaddingMask *Node, numHeads int) *Node {
	tgt2 := layerNormLast(scope.In("norm1"), tgt, 1e-5)
	qk := Add(tgt2, queryPos)
	self := multiHeadAttention(scope.In("self_attn"), qk, qk, tgt2, numHeads, true, nil)
	tgt = Add(tgt, self)

	tgt2 = layerNormLast(scope.In("norm2"), tgt, 1e-5)
	mask := keyPaddingMaskToAttention(keyPaddingMask, numHeads)
	cross := multiHeadAttention(scope.In("cross_attn_image"), tgt2, memory, memory, numHeads, true, mask)
	tgt = Add(tgt, cross)

	tgt2 = layerNormLast(scope.In("norm3"), tgt, 1e-5)
	ff := linear(scope.In("linear1"), tgt2)
	ff = activation.Relu(ff)
	ff = linear(scope.In("linear2"), ff)
	return Add(tgt, ff)
}

// transformerEncoderFusion runs the TransformerEncoderFusion stack. src/pos are
// [batch, c, H, W] (single feature level); prompt is [seq, batch, c] and
// promptMask [batch, seq] (bool, true = padding). It returns the encoded memory
// and positional embedding in sequence-first layout [H*W, batch, c].
func transformerEncoderFusion(scope *model.Scope, src, pos, prompt, promptMask *Node, cfg *Config) (memory, posEmbed *Node) {
	dims := src.Shape().Dimensions
	b, c, h, w := dims[0], dims[1], dims[2], dims[3]
	hw := h * w

	srcFlat := TransposeAllAxes(Reshape(src, b, c, hw), 0, 2, 1)
	posFlat := TransposeAllAxes(Reshape(pos, b, c, hw), 0, 2, 1)
	promptB := TransposeAllAxes(prompt, 1, 0, 2)

	tgt := srcFlat
	layersScope := scope.In("encoder").In("layers")
	for i := range cfg.Transformer.NumEncoderLayers {
		tgt = encoderLayer(layersScope.In("%d", i), tgt, posFlat, promptB, promptMask, cfg.Transformer.NumHeads)
	}

	memory = TransposeAllAxes(tgt, 1, 0, 2)
	posEmbed = TransposeAllAxes(posFlat, 1, 0, 2)
	return memory, posEmbed
}

// boxRPB computes the logarithmic relative-position bias used as the additive
// cross-attention mask in the decoder. referenceBoxes is [nq, batch, 4]
// (cxcywh); featH/featW are the memory grid. Returns [batch, heads, nq, hw].
func boxRPB(scope *model.Scope, referenceBoxes *Node, featH, featW, numHeads int) *Node {
	g := referenceBoxes.Graph()
	nq := referenceBoxes.Shape().Dim(0)
	bs := referenceBoxes.Shape().Dim(1)

	boxesXYXY := TransposeAllAxes(boxCxcywhToXyxy(referenceBoxes), 1, 0, 2) // [batch, nq, 4]

	coordsH := DivScalar(IotaFull(g, shapes.Make(dtypes.Float32, featH)), float64(featH))
	coordsW := DivScalar(IotaFull(g, shapes.Make(dtypes.Float32, featW)), float64(featW))

	// deltasY: [batch, nq, H, 2] = coordsH - boxes[..., {y0, y1}]
	yCorners := Concatenate([]*Node{
		Slice(boxesXYXY, AxisRange(), AxisRange(), AxisRange(1, 2)),
		Slice(boxesXYXY, AxisRange(), AxisRange(), AxisRange(3, 4)),
	}, -1)
	deltasY := Sub(Reshape(coordsH, 1, 1, featH, 1), ExpandAxes(yCorners, 2)) // [batch, nq, H, 2]

	xCorners := Concatenate([]*Node{
		Slice(boxesXYXY, AxisRange(), AxisRange(), AxisRange(0, 1)),
		Slice(boxesXYXY, AxisRange(), AxisRange(), AxisRange(2, 3)),
	}, -1)
	deltasX := Sub(Reshape(coordsW, 1, 1, featW, 1), ExpandAxes(xCorners, 2)) // [batch, nq, W, 2]

	deltasX = boxRPBLogTransform(deltasX)
	deltasY = boxRPBLogTransform(deltasY)

	deltasX = mlp(scope.At("boxRPB_embed_x"), deltasX, 2, false, false) // [batch, nq, W, heads]
	deltasY = mlp(scope.At("boxRPB_embed_y"), deltasY, 2, false, false) // [batch, nq, H, heads]

	// B = deltasY[..., None, :] + deltasX[:, :, None, :, :] -> [batch, nq, H, W, heads]
	b := Add(ExpandAxes(deltasY, 3), ExpandAxes(deltasX, 2))
	// flatten H,W -> [batch, nq, H*W, heads], then permute -> [batch, heads, nq, H*W].
	b = Reshape(b, bs, nq, featH*featW, numHeads)
	b = TransposeAllAxes(b, 0, 3, 1, 2)
	return b
}

// boxRPBLogTransform applies sign(d*8) * log2(|d*8|+1) / log2(8).
func boxRPBLogTransform(d *Node) *Node {
	d8 := MulScalar(d, 8.0)
	log2 := Div(Log(Add(Abs(d8), OnesLike(d8))), ConstAs(d8, math.Log(2.0)))
	return DivScalar(Mul(Sign(d8), log2), math.Log2(8.0))
}

// decoderLayer runs one TransformerDecoderLayer (seq-first, pre-norm) with the
// presence token prepended.
func decoderLayer(scope *model.Scope, tgt, queryPos, presenceToken, memory, memoryPos, memoryText, textMask, crossAttnMask *Node, numHeads int) (outTgt, outPresence *Node) {
	zero := ZerosLike(presenceToken)

	// Prepend the presence token (with zero query position).
	tgtWithPres := Concatenate([]*Node{presenceToken, tgt}, 0)
	queryPosWithPres := Concatenate([]*Node{zero, queryPos}, 0)

	// Self attention.
	qk := Add(tgtWithPres, queryPosWithPres)
	self := multiHeadAttention(scope.In("self_attn"), qk, qk, tgtWithPres, numHeads, false, nil)
	tgt = layerNormLast(scope.In("norm2"), Add(tgtWithPres, self), 1e-5)

	// Text cross attention.
	q := Add(tgt, queryPosWithPres)
	tMask := keyPaddingMaskToAttention(textMask, numHeads)
	caText := multiHeadAttention(scope.In("ca_text"), q, memoryText, memoryText, numHeads, false, tMask)
	tgt = layerNormLast(scope.In("catext_norm"), Add(tgt, caText), 1e-5)

	// Cross attention to image (with boxRPB additive mask extended for the
	// presence token).
	q = Add(tgt, queryPosWithPres)
	k := Add(memory, memoryPos)
	crossMask := crossAttnMaskWithPresence(crossAttnMask)
	cross := multiHeadAttention(scope.In("cross_attn"), q, k, memory, numHeads, false, crossMask)
	tgt = layerNormLast(scope.In("norm1"), Add(tgt, cross), 1e-5)

	// FFN (post-norm: linear on the raw input, then residual, then norm3).
	ff := linear(scope.In("linear1"), tgt)
	ff = activation.Relu(ff)
	ff = linear(scope.In("linear2"), ff)
	tgt = layerNormLast(scope.In("norm3"), Add(tgt, ff), 1e-5)

	// Split presence token back out.
	outPresence = Slice(tgt, AxisRange(0, 1), AxisRange(), AxisRange())
	outTgt = Slice(tgt, AxisRange(1, tgt.Shape().Dim(0)), AxisRange(), AxisRange())
	return outTgt, outPresence
}

// crossAttnMaskWithPresence prepends a zero-bias row for the presence token to
// the boxRPB cross-attention mask [batch, heads, nq, hw].
func crossAttnMaskWithPresence(mask *Node) *Node {
	dims := mask.Shape().Dimensions
	zeroRow := Zeros(mask.Graph(), shapes.Make(mask.DType(), dims[0], dims[1], 1, dims[3]))
	return Concatenate([]*Node{zeroRow, mask}, 2)
}

// genSineEmbed generates the sine embedding for a [.., n] coordinate tensor.
func genSineEmbed(g *Graph, pos *Node, numFeats int) *Node {
	half := numFeats / 2
	scale := 2 * math.Pi
	dimT := IotaFull(g, shapes.Make(dtypes.Float32, half))
	dimT = Pow(ConstAs(dimT, 10000.0), DivScalar(MulScalar(Floor(DivScalar(dimT, 2.0)), 2.0), float64(half)))
	dimT = Reshape(dimT, 1, 1, half)

	n := pos.Shape().Dim(-1)
	embeds := make([]*Node, 0, n)
	for i := range n {
		coord := Slice(pos, AxisRange().Spacer(), AxisRange(i, i+1))
		coord = Div(MulScalar(coord, scale), dimT)
		embeds = append(embeds, interleaveSinCos(coord))
	}
	return Concatenate(embeds, -1)
}

// dotProductScoring computes the detection scores. hs is [numLayers, batch, nq,
// dim], prompt is [seq, batch, dim], promptMask [batch, seq]. Returns
// [numLayers, batch, nq, 1].
func dotProductScoring(scope *model.Scope, hs, prompt, promptMask *Node, cfg *Config) *Node {
	// prompt_mlp (residual MLP with LayerNorm output).
	projPrompt := mlp(scope.In("prompt_mlp"), prompt, 2, true, true)

	// Mean-pool the prompt over valid tokens.
	isValid := ConvertDType(LogicalNot(promptMask), projPrompt.DType()) // [batch, seq] 1 = valid
	projPrompt = TransposeAllAxes(projPrompt, 1, 0, 2)                  // [batch, seq, dim]
	projPrompt = Mul(projPrompt, ExpandAxes(isValid, -1))
	numValid := ReduceSum(isValid, 1) // [batch]
	numValid = Max(numValid, OnesLike(numValid))
	pooled := Div(ReduceSum(projPrompt, 1), ExpandAxes(numValid, -1)) // [batch, dim]

	pooledProj := linear(scope.In("prompt_proj"), pooled) // [batch, dProj]
	hsProj := linear(scope.In("hs_proj"), hs)             // [numLayers, batch, nq, dProj]

	scores := Einsum("lbqc,bc->lbq", hsProj, pooledProj) // [numLayers, batch, nq]
	scores = MulScalar(scores, 1.0/math.Sqrt(float64(cfg.Transformer.DotProjDim)))
	scores = Clamp(ConstAs(scores, -cfg.Transformer.DotClampMax), scores, ConstAs(scores, cfg.Transformer.DotClampMax))
	return ExpandAxes(scores, -1)
}

// decoderOutput is the result of the DETR decoder.
type decoderOutput struct {
	// Queries [batch, nq, dim] (final layer).
	Queries *Node
	// HS [numLayers, batch, nq, dim].
	HS *Node
	// PredLogits [batch, nq, 1].
	PredLogits *Node
	// PredBoxes [batch, nq, 4] (cxcywh).
	PredBoxes *Node
	// PredBoxesXYXY [batch, nq, 4].
	PredBoxesXYXY *Node
	// PresenceLogitDec [batch, 1].
	PresenceLogitDec *Node
	// ReferenceBoxes [numLayers, batch, nq, 4].
	ReferenceBoxes *Node
}

// transformerDecoder runs the DETR decoder (6 layers, 200 queries).
//
// scope is the "transformer" scope; rootScope is the top-level model scope used
// to reach the dot-product scoring head (which lives outside the transformer).
func transformerDecoder(scope, rootScope *model.Scope, memory, pos, prompt, promptMask *Node, cfg *Config) *decoderOutput {
	g := memory.Graph()
	nq := cfg.Transformer.NumQueries
	bs := memory.Shape().Dim(1)
	dim := cfg.Transformer.DModel
	featSize := cfg.Transformer.Resolution / cfg.Transformer.Stride

	dec := scope.In("decoder")

	queryEmbed := dec.In("query_embed").GetVariable("weight").NodeValue(g)        // [nq, dim]
	referencePts := dec.In("reference_points").GetVariable("weight").NodeValue(g) // [nq, 4]
	presenceTok := dec.In("presence_token").GetVariable("weight").NodeValue(g)    // [1, dim]

	tgt := BroadcastToDims(ExpandAxes(queryEmbed, 1), nq, bs, dim)
	referenceBoxes := Sigmoid(BroadcastToDims(ExpandAxes(referencePts, 1), nq, bs, 4))
	presence := BroadcastToDims(ExpandAxes(presenceTok, 1), 1, bs, dim)

	layersScope := dec.In("layers")
	intermediate := make([]*Node, 0, cfg.Transformer.NumDecoderLayers)
	intermediateRef := make([]*Node, 0, cfg.Transformer.NumDecoderLayers)
	presenceLogits := make([]*Node, 0, cfg.Transformer.NumDecoderLayers)

	for i := range cfg.Transformer.NumDecoderLayers {
		querySine := genSineEmbed(g, referenceBoxes, dim) // [nq, bs, dim*2]
		queryPos := mlp(dec.At("ref_point_head"), querySine, 2, false, false)

		var crossMask *Node
		if cfg.Transformer.BoxRPB != "none" {
			crossMask = boxRPB(dec, referenceBoxes, featSize, featSize, cfg.Transformer.NumHeads)
		}

		tgt, presence = decoderLayer(layersScope.In("%d", i), tgt, queryPos, presence, memory, pos, prompt, promptMask, crossMask, cfg.Transformer.NumHeads)

		// Box refinement.
		normed := layerNormLast(dec.At("norm"), tgt, 1e-5)
		delta := mlp(dec.At("bbox_embed"), normed, 3, false, false)
		refBefore := inverseSigmoid(referenceBoxes, 1e-3)
		newRef := Sigmoid(Add(delta, refBefore))
		referenceBoxes = newRef

		intermediate = append(intermediate, normed)
		intermediateRef = append(intermediateRef, newRef)

		presLogit := layerNormLast(dec.At("presence_token_out_norm"), presence, 1e-5)
		presLogit = mlp(dec.At("presence_token_head"), presLogit, 3, false, false) // [1, bs, 1]
		presLogit = Squeeze(presLogit, -1)                                         // [1, bs]
		presLogit = Clamp(ConstAs(presLogit, -cfg.Transformer.PresenceClampMax), presLogit, ConstAs(presLogit, cfg.Transformer.PresenceClampMax))
		presenceLogits = append(presenceLogits, presLogit)
	}

	// Stack intermediates: [numLayers, nq, bs, dim] -> [numLayers, bs, nq, dim].
	hs := Stack(intermediate, 0)
	hs = TransposeAllAxes(hs, 0, 2, 1, 3)
	refBoxes := Stack(intermediateRef, 0)
	refBoxes = TransposeAllAxes(refBoxes, 0, 2, 1, 3)
	presLogits := Stack(presenceLogits, 0)             // [numLayers, 1, bs]
	presLogits = TransposeAllAxes(presLogits, 0, 2, 1) // [numLayers, bs, 1]

	outputsClass := dotProductScoring(rootScope.In("dot_prod_scoring"), hs, prompt, promptMask, cfg)
	anchorOffsets := mlp(dec.At("bbox_embed"), hs, 3, false, false)
	outputsCoord := Sigmoid(Add(inverseSigmoid(refBoxes, 1e-3), anchorOffsets))
	outputsXYXY := boxCxcywhToXyxy(outputsCoord)

	nl := cfg.Transformer.NumDecoderLayers
	return &decoderOutput{
		Queries:          Squeeze(Slice(hs, AxisRange(nl-1, nl), AxisRange(), AxisRange(), AxisRange()), 0),
		HS:               hs,
		PredLogits:       Squeeze(Slice(outputsClass, AxisRange(nl-1, nl), AxisRange(), AxisRange(), AxisRange()), 0),
		PredBoxes:        Squeeze(Slice(outputsCoord, AxisRange(nl-1, nl), AxisRange(), AxisRange(), AxisRange()), 0),
		PredBoxesXYXY:    Squeeze(Slice(outputsXYXY, AxisRange(nl-1, nl), AxisRange(), AxisRange(), AxisRange()), 0),
		PresenceLogitDec: Squeeze(Slice(presLogits, AxisRange(nl-1, nl), AxisRange(), AxisRange()), 0),
		ReferenceBoxes:   refBoxes,
	}
}
