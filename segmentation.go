// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
)

// pixelDecoder implements the Mask2Former-style PixelDecoder over the FPN
// features. backboneFeats are channels-first [batch, 256, H, W] (high-to-low
// resolution); the last level has already been replaced by the encoded image
// features.
func pixelDecoder(scope *model.Scope, backboneFeats []*Node) *Node {
	prev := backboneFeats[len(backboneFeats)-1]
	fpnFeats := backboneFeats[:len(backboneFeats)-1]

	convLayersScope := scope.In("conv_layers")
	normsScope := scope.In("norms")
	for i := range fpnFeats {
		curr := fpnFeats[len(fpnFeats)-1-i]
		h := curr.Shape().Dim(2)
		w := curr.Shape().Dim(3)
		up := Interpolate(prev, -1, -1, h, w).Nearest().Done()
		prev = Add(curr, up)
		prev = conv2d(convLayersScope.In("%d", i), prev, 256, 3, 1, 1)
		prev = groupNorm(normsScope.In("%d", i), prev, 8, 1e-5)
		prev = activation.Relu(prev)
	}
	return prev
}

// segmentationHead runs the UniversalSegmentationHead, returning the predicted
// masks [batch, numQueries, outH, outW] (logits).
//
// backboneFeats are the FPN features (high-to-low), objQueries is the final
// decoder layer output [batch, numQueries, dModel], encoderHiddenStates is the
// encoded image memory [H*W, batch, dModel], and prompt/promptMask are the
// text+geometry prompt (seq-first).
func segmentationHead(scope *model.Scope, backboneFeats []*Node, objQueries, encoderHiddenStates, prompt, promptMask *Node, cfg *Config) *Node {
	// Cross-attend the encoder output to the prompt.
	tgt2 := layerNormLast(scope.In("cross_attn_norm"), encoderHiddenStates, 1e-5)
	mask := keyPaddingMaskToAttention(promptMask, cfg.Transformer.NumHeads)
	ca := multiHeadAttention(scope.In("cross_attend_prompt"), tgt2, prompt, prompt, cfg.Transformer.NumHeads, false, mask)
	encoderHiddenStates = Add(encoderHiddenStates, ca)

	// Replace the lowest-resolution FPN level with the encoder output.
	bs := encoderHiddenStates.Shape().Dim(1)
	lastDims := backboneFeats[len(backboneFeats)-1].Shape().Dimensions
	c, lh, lw := lastDims[1], lastDims[2], lastDims[3]
	encPerm := TransposeAllAxes(encoderHiddenStates, 1, 2, 0) // [batch, c, hw]
	encVisual := Reshape(encPerm, bs, c, lh, lw)

	feats := make([]*Node, len(backboneFeats))
	copy(feats, backboneFeats)
	feats[len(feats)-1] = encVisual

	pixelEmbed := pixelDecoder(scope.In("pixel_decoder"), feats)

	instanceEmbeds := conv2d(scope.In("instance_seg_head"), pixelEmbed, cfg.Segmentation.MaskDim, 1, 1, 0) // [batch, 256, outH, outW]

	queryEmb := mlp(scope.In("mask_predictor").In("mask_embed"), objQueries, 3, false, false) // [batch, nq, 256]

	predMasks := Einsum("bqc,bchw->bqhw", queryEmb, instanceEmbeds)
	return predMasks
}
