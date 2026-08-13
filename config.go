// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the architecture configuration of the SAM 3.1 image model.
//
// The official facebook/sam3.1 checkpoint does not ship a full model config
// (the architecture is encoded in the reference Python implementation). The
// values below mirror the reference `model_builder.py` defaults for the image
// grounding (PCS) model.
type Config struct {
	// ImageSize is the resolution the image is resized to before inference.
	ImageSize int

	// ViT backbone parameters.
	Backbone ViTConfig

	// Neck parameters (ViTDet SimpleFPN).
	Neck NeckConfig

	// Text encoder parameters.
	Text TextConfig

	// Transformer encoder/decoder parameters.
	Transformer TransformerConfig

	// Geometry encoder parameters.
	Geometry GeometryConfig

	// Segmentation head parameters.
	Segmentation SegmentationConfig
}

// ViTConfig describes the ViT-Det vision backbone.
type ViTConfig struct {
	PatchSize    int
	EmbedDim     int
	Depth        int
	NumHeads     int
	MLPRatio     float64
	WindowSize   int
	GlobalBlocks []int // block indices using full (global) attention.
	// PretrainImgSize is the patch-grid resolution used to pretrain the absolute
	// positional embedding (336 / 14 == 24 positions).
	PretrainImgSize int
	// UseAbsPos enables absolute positional embeddings.
	UseAbsPos bool
	// TileAbsPos tiles the absolute position embedding instead of interpolating.
	TileAbsPos bool
	// UseRoPE enables 2D axial RoPE.
	UseRoPE bool
	// UseInterpRoPE interpolates RoPE positions to the input size.
	UseInterpRoPE bool
	// RoPETheta controls the RoPE frequency base.
	RoPETheta float64
}

// NeckConfig describes the ViTDet SimpleFPN neck. The image model uses three
// FPN levels produced by the scale factors below (after dropping the lowest
// resolution level).
type NeckConfig struct {
	DModel       int
	ScaleFactors []float64
}

// TextConfig describes the CLIP-style text encoder.
type TextConfig struct {
	Width         int // transformer width (1024).
	Heads         int
	Layers        int
	ContextLength int
	VocabSize     int
	DModel        int // output feature dim after the resizer projection.
}

// TransformerConfig describes the DETR-style encoder/decoder.
type TransformerConfig struct {
	DModel           int
	DimFeedforward   int
	NumHeads         int
	NumEncoderLayers int
	NumDecoderLayers int
	NumQueries       int
	Resolution       int // image resolution, used to derive the boxRPB grid.
	Stride           int // patch stride, used to derive the boxRPB grid.
	NumFeatureLevels int
	UseTextCrossAttn bool
	PresenceToken    bool
	PresenceClampMax float64
	BoxRPB           string // "none" or "log".
	DotProjDim       int
	DotClampMax      float64
}

// GeometryConfig describes the geometric prompt encoder.
type GeometryConfig struct {
	DModel              int
	NumLayers           int
	ROISize             int
	NumLabels           int
	EncodeBoxesAsPoints bool
	AddCls              bool
	AddPostEncodeProj   bool
}

// SegmentationConfig describes the universal segmentation head.
type SegmentationConfig struct {
	DModel           int
	UpsamplingStages int
	MaskDim          int
	NumGroups        int // GroupNorm groups in the pixel decoder.
}

// DefaultConfig returns the SAM 3.1 image-model configuration matching the
// reference `build_sam3_image_model`.
func DefaultConfig() *Config {
	return &Config{
		ImageSize: 1008,
		Backbone: ViTConfig{
			PatchSize:       14,
			EmbedDim:        1024,
			Depth:           32,
			NumHeads:        16,
			MLPRatio:        4.625,
			WindowSize:      24,
			GlobalBlocks:    []int{7, 15, 23, 31},
			PretrainImgSize: 336,
			UseAbsPos:       true,
			TileAbsPos:      true,
			UseRoPE:         true,
			UseInterpRoPE:   true,
			RoPETheta:       10000.0,
		},
		Neck: NeckConfig{
			DModel:       256,
			ScaleFactors: []float64{4.0, 2.0, 1.0},
		},
		Text: TextConfig{
			Width:         1024,
			Heads:         16,
			Layers:        24,
			ContextLength: 32,
			VocabSize:     49408,
			DModel:        256,
		},
		Transformer: TransformerConfig{
			DModel:           256,
			DimFeedforward:   2048,
			NumHeads:         8,
			NumEncoderLayers: 6,
			NumDecoderLayers: 6,
			NumQueries:       200,
			Resolution:       1008,
			Stride:           14,
			NumFeatureLevels: 1,
			UseTextCrossAttn: true,
			PresenceToken:    true,
			PresenceClampMax: 10.0,
			BoxRPB:           "log",
			DotProjDim:       256,
			DotClampMax:      12.0,
		},
		Geometry: GeometryConfig{
			DModel:              256,
			NumLayers:           3,
			ROISize:             7,
			NumLabels:           2,
			EncodeBoxesAsPoints: false,
			AddCls:              true,
			AddPostEncodeProj:   true,
		},
		Segmentation: SegmentationConfig{
			DModel:           256,
			UpsamplingStages: 3,
			MaskDim:          256,
			NumGroups:        8,
		},
	}
}

// LoadConfig parses an optional config.json. It falls back to DefaultConfig
// values for any fields not present.
func LoadConfig(path string) (*Config, error) {
	c := DefaultConfig()
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	if err := json.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}
	return c, nil
}
