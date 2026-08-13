// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
)

// textEncoderForward runs the CLIP-style text encoder (VETextEncoder) in graph.
//
// tokenIDs has shape [batch, contextLength] (int32). It returns the encoded
// text memory in sequence-first layout [contextLength, batch, dModel] and the
// padding mask [batch, contextLength] (bool, true = padding).
func textEncoderForward(scope *model.Scope, tokenIDs *Node, cfg *Config) (textMemory, textMask *Node) {
	g := tokenIDs.Graph()
	batch := tokenIDs.Shape().Dimensions[0]
	ctx := tokenIDs.Shape().Dimensions[1]

	enc := scope.In("encoder")

	// Token embedding [vocab, width] -> [batch, ctx, width].
	x := embedding(enc.In("token_embedding"), tokenIDs, cfg.Text.VocabSize, cfg.Text.Width)

	// Positional embedding [contextLength, width].
	posEmb := enc.GetVariable("positional_embedding").NodeValue(g)
	posEmb = Slice(posEmb, AxisRange(0, ctx), AxisRange())
	x = Add(x, BroadcastToDims(Reshape(posEmb, 1, ctx, cfg.Text.Width), batch, ctx, cfg.Text.Width))

	cMask := causalMask(g, ctx)

	tr := enc.In("transformer").In("resblocks")
	for i := range cfg.Text.Layers {
		b := tr.In("%d", i)
		x = residualAttentionBlock(b, x, cMask, cfg.Text.Width, cfg.Text.Heads)
	}

	x = layerNormLast(enc.In("ln_final"), x, 1e-5)

	// Resizer: width -> dModel.
	resized := linear(scope.In("resizer"), x) // [batch, ctx, dModel]

	textMemory = TransposeAllAxes(resized, 1, 0, 2) // [ctx, batch, dModel]
	textMask = Equal(tokenIDs, Scalar(g, dtypes.Int32, 0))
	return textMemory, textMask
}

// residualAttentionBlock runs one CLIP-style transformer block with a causal
// attention mask.
func residualAttentionBlock(scope *model.Scope, x *Node, attnMask *Node, width, heads int) *Node {
	h := layerNormLast(scope.In("ln_1"), x, 1e-5)
	h = multiHeadAttention(scope.In("attn"), h, h, h, heads, true, attnMask)
	x = Add(x, h)

	h = layerNormLast(scope.In("ln_2"), x, 1e-5)
	mlpScope := scope.In("mlp")
	h = linear(mlpScope.In("c_fc"), h)
	h = activation.Gelu(h)
	h = linear(mlpScope.In("c_proj"), h)
	x = Add(x, h)
	return x
}
