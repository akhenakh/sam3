// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package sam3 implements Facebook's Segment Anything Model 3.1 (SAM 3.1)
// image grounding / segmentation model in GoMLX.
//
// This package implements the image part of the model (the ViT-Det vision
// backbone, ViTDet neck, text encoder, DETR-style transformer encoder/decoder,
// geometry encoder and universal segmentation head), not the video tracking
// part (Object Multiplex).
//
// It is based on the reference PyTorch implementation:
//
//	https://github.com/facebookresearch/sam3
//
// and loads the bf16 safetensors checkpoint from a HuggingFace repository.
package sam3

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	modelimage "github.com/gomlx/go-huggingface/models/image"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/model"
)

// ForwardOutput holds the outputs of the SAM 3.1 image grounding model.
type ForwardOutput struct {
	// PredLogits are the raw detection scores [batch, numQueries, 1].
	PredLogits *Node
	// PredBoxes are the predicted boxes in cxcywh format [batch, numQueries, 4].
	PredBoxes *Node
	// PredMasks are the predicted mask logits [batch, numQueries, outH, outW].
	PredMasks *Node
	// PresenceLogit is the presence-token logit [batch, 1].
	PresenceLogit *Node
}

// Forward builds the full SAM 3.1 image-grounding graph.
//
// Inputs:
//   - pixelValues: [batch, 3, imageSize, imageSize], normalized with mean/std 0.5.
//   - tokenIDs: [batch, contextLength] int32 text tokens.
//   - points/pointLabels/pointMask: point prompts ([nPoints, batch, 2],
//     [nPoints, batch] int32, [batch, nPoints] bool padding mask).
//   - boxes/boxLabels/boxMask: box prompts ([nBoxes, batch, 4] cxcywh
//     normalized, [nBoxes, batch] int32, [batch, nBoxes] bool padding mask).
func Forward(scope *model.Scope, pixelValues, tokenIDs, points, pointLabels, pointMask, boxes, boxLabels, boxMask *Node, cfg *Config) *ForwardOutput {
	batch := pixelValues.Shape().Dim(0)

	// 1. Vision backbone + neck.
	backboneScope := scope.In("backbone")
	features, posEnc := visionBackboneForward(backboneScope.In("vision_backbone"), pixelValues, cfg)

	// 2. Text encoder.
	textMemory, textMask := textEncoderForward(backboneScope.In("language_backbone"), tokenIDs, cfg)

	// 3. Geometry encoder (uses the lowest-resolution FPN level).
	lastFeat := features[len(features)-1]
	lastPos := posEnc[len(posEnc)-1]
	dims := lastFeat.Shape().Dimensions
	c, lh, lw := dims[1], dims[2], dims[3]
	hw := lh * lw
	lastFeatSeq := TransposeAllAxes(Reshape(lastFeat, batch, c, hw), 2, 0, 1) // [hw, batch, c]
	lastPosSeq := TransposeAllAxes(Reshape(lastPos, batch, c, hw), 2, 0, 1)   // [hw, batch, c]

	geoFeats, geoMask := geometryEncoder(scope.In("geometry_encoder"), points, pointLabels, pointMask, boxes, boxLabels, boxMask, lastFeatSeq, lastFeat, lastPosSeq, cfg)

	// 4. Concatenate text + geometry prompts.
	prompt := Concatenate([]*Node{textMemory, geoFeats}, 0)
	promptMask := Concatenate([]*Node{textMask, geoMask}, 1)

	// 5. Transformer encoder (single feature level).
	transformerScope := scope.In("transformer")
	memory, posEmbed := transformerEncoderFusion(transformerScope, lastFeat, lastPos, prompt, promptMask, cfg)

	// 6. Transformer decoder.
	decOut := transformerDecoder(transformerScope, scope, memory, posEmbed, prompt, promptMask, cfg)

	// 7. Segmentation head.
	predMasks := segmentationHead(scope.In("segmentation_head"), features, decOut.Queries, memory, prompt, promptMask, cfg)

	return &ForwardOutput{
		PredLogits:    decOut.PredLogits,
		PredBoxes:     decOut.PredBoxes,
		PredMasks:     predMasks,
		PresenceLogit: decOut.PresenceLogitDec,
	}
}

// PointLabel indicates whether a point is foreground or background.
type PointLabel int

const (
	LabelBackground PointLabel = 0
	LabelForeground PointLabel = 1
)

// PromptPoint is a point prompt with its label.
type PromptPoint struct {
	X, Y  int
	Label PointLabel
}

// PromptBox is a bounding-box prompt (pixel coordinates).
type PromptBox struct {
	MinX, MinY int
	MaxX, MaxY int
}

// SegmentationOptions describes the prompts for a segmentation request.
type SegmentationOptions struct {
	// Text is the text prompt (e.g. "buildings"). Optional.
	Text string
	// Points are point prompts. Optional.
	Points []PromptPoint
	// Boxes are box prompts. Optional.
	Boxes []PromptBox
	// Threshold is the detection probability threshold (default 0.5).
	Threshold float64
	// MaxDetections caps the number of returned detections.
	MaxDetections int
}

// Detection is a single predicted segmentation.
type Detection struct {
	// Mask is the binary segmentation mask resized to the original image size.
	Mask image.Image
	// Box is the predicted bounding box (pixel coordinates).
	Box PromptBox
	// Score is the detection probability in [0, 1].
	Score float32
}

// Segmenter runs SAM 3.1 image grounding.
type Segmenter struct {
	backend   compute.Backend
	model     *Model
	store     *model.Store
	config    *Config
	exec      *model.Exec
	maxPoints int
	maxBoxes  int
	tokenizer *tokenizer
}

// NewSegmenter initializes a SAM 3.1 segmenter.
//
// bpePath is the path to the gzipped BPE merge file (the sam3 asset
// `bpe_simple_vocab_16e6.txt.gz`).
func NewSegmenter(backend compute.Backend, modelObj *Model, bpePath string) (*Segmenter, error) {
	if backend == nil {
		var err error
		backend, err = compute.New()
		if err != nil {
			return nil, err
		}
	}

	tok, err := newTokenizer(bpePath)
	if err != nil {
		return nil, err
	}

	store := model.NewStore()
	if err := modelObj.LoadStore(backend, store); err != nil {
		return nil, fmt.Errorf("failed to load model weights: %w", err)
	}

	s := &Segmenter{
		backend:   backend,
		model:     modelObj,
		store:     store,
		config:    modelObj.Config,
		maxPoints: 16,
		maxBoxes:  16,
		tokenizer: tok,
	}

	exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*Node) []*Node {
		rawImage, tokenIDs := inputs[0], inputs[1]
		points, pointLabels, pointMask := inputs[2], inputs[3], inputs[4]
		boxes, boxLabels, boxMask := inputs[5], inputs[6], inputs[7]
		mean := []float64{0.5, 0.5, 0.5}
		std := []float64{0.5, 0.5, 0.5}
		preprocessed := modelimage.PreprocessGraph(rawImage, s.config.ImageSize, s.config.ImageSize, mean, std)
		out := Forward(scope, preprocessed, tokenIDs, points, pointLabels, pointMask, boxes, boxLabels, boxMask, s.config)
		return []*Node{out.PredLogits, out.PredBoxes, out.PredMasks, out.PresenceLogit}
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create executor: %w", err)
	}
	s.exec = exec

	return s, nil
}

// Segment runs SAM 3.1 on the given image and prompts.
func (p *Segmenter) Segment(img image.Image, options *SegmentationOptions) ([]Detection, error) {
	if options == nil {
		options = &SegmentationOptions{}
	}
	threshold := options.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	maxDet := options.MaxDetections
	if maxDet <= 0 {
		maxDet = 200
	}

	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()

	// Tokenize text.
	var tokenIDs [][]int32
	text := options.Text
	if text == "" {
		text = "visual"
	}
	ids := p.tokenizer.tokenize(text)
	ids32 := make([]int32, len(ids))
	for i, v := range ids {
		ids32[i] = int32(v)
	}
	tokenIDs = append(tokenIDs, ids32)

	// Build point/box tensors.
	points, pointLabels, pointMask := p.buildPoints(options.Points, origW, origH)
	boxes, boxLabels, boxMask := p.buildBoxes(options.Boxes, origW, origH)

	imgTensor := images.ToTensor(dtypes.Float32).Single(img)
	tokenTensor := tensors.FromValue([][]int32{ids32})
	pointsTensor := tensors.FromValue(points)
	pointLabelsTensor := tensors.FromValue(pointLabels)
	pointMaskTensor := tensors.FromValue(pointMask)
	boxesTensor := tensors.FromValue(boxes)
	boxLabelsTensor := tensors.FromValue(boxLabels)
	boxMaskTensor := tensors.FromValue(boxMask)
	defer imgTensor.MustFinalizeAll()
	defer tokenTensor.MustFinalizeAll()
	defer pointsTensor.MustFinalizeAll()
	defer pointLabelsTensor.MustFinalizeAll()
	defer pointMaskTensor.MustFinalizeAll()
	defer boxesTensor.MustFinalizeAll()
	defer boxLabelsTensor.MustFinalizeAll()
	defer boxMaskTensor.MustFinalizeAll()

	outputs, err := p.exec.Call(imgTensor, tokenTensor, pointsTensor, pointLabelsTensor, pointMaskTensor, boxesTensor, boxLabelsTensor, boxMaskTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.MustFinalizeAll()
		}
	}()

	logitsVal := outputs[0].Value().([][][]float32)  // [1, nq, 1]
	boxesVal := outputs[1].Value().([][][]float32)   // [1, nq, 4]
	masksVal := outputs[2].Value().([][][][]float32) // [1, nq, outH, outW]
	presenceVal := outputs[3].Value().([][]float32)  // [1, 1]

	nq := len(logitsVal[0])
	presenceProb := sigmoid(presenceVal[0][0])

	var detections []Detection
	for q := 0; q < nq; q++ {
		prob := sigmoid(logitsVal[0][q][0]) * presenceProb
		if prob < float32(threshold) {
			continue
		}
		b := boxesVal[0][q] // cxcywh normalized
		box := denormalizeBox(b, origW, origH)
		mask := resizeMaskBilinear(masksVal[0][q], origW, origH, 0.5)
		detections = append(detections, Detection{Mask: mask, Box: box, Score: prob})
	}

	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score > detections[j].Score
	})
	if len(detections) > maxDet {
		detections = detections[:maxDet]
	}
	return detections, nil
}

func (p *Segmenter) buildPoints(points []PromptPoint, origW, origH int) ([][][]float32, [][]int32, [][]bool) {
	n := p.maxPoints
	pts := make([][][]float32, n)
	lbls := make([][]int32, n)
	mask := make([]bool, n)
	for i := range n {
		if i < len(points) {
			pt := points[i]
			pts[i] = [][]float32{{float32(pt.X) / float32(origW), float32(pt.Y) / float32(origH)}}
			lbls[i] = []int32{int32(pt.Label)}
			mask[i] = false
		} else {
			pts[i] = [][]float32{{0, 0}}
			lbls[i] = []int32{1}
			mask[i] = true
		}
	}
	return pts, lbls, [][]bool{mask}
}

func (p *Segmenter) buildBoxes(boxes []PromptBox, origW, origH int) ([][][]float32, [][]int32, [][]bool) {
	n := p.maxBoxes
	bxs := make([][][]float32, n)
	lbls := make([][]int32, n)
	mask := make([]bool, n)
	for i := range n {
		if i < len(boxes) {
			b := boxes[i]
			cx := (float32(b.MinX) + float32(b.MaxX)) / 2 / float32(origW)
			cy := (float32(b.MinY) + float32(b.MaxY)) / 2 / float32(origH)
			w := (float32(b.MaxX) - float32(b.MinX)) / float32(origW)
			h := (float32(b.MaxY) - float32(b.MinY)) / float32(origH)
			bxs[i] = [][]float32{{cx, cy, w, h}}
			lbls[i] = []int32{1}
			mask[i] = false
		} else {
			bxs[i] = [][]float32{{0, 0, 0, 0}}
			lbls[i] = []int32{1}
			mask[i] = true
		}
	}
	return bxs, lbls, [][]bool{mask}
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func denormalizeBox(b []float32, origW, origH int) PromptBox {
	cx, cy, w, h := b[0], b[1], b[2], b[3]
	x0 := (cx - w/2) * float32(origW)
	y0 := (cy - h/2) * float32(origH)
	x1 := (cx + w/2) * float32(origW)
	y1 := (cy + h/2) * float32(origH)
	return PromptBox{MinX: clampInt(int(x0), 0, origW), MinY: clampInt(int(y0), 0, origH), MaxX: clampInt(int(x1), 0, origW), MaxY: clampInt(int(y1), 0, origH)}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// resizeMaskBilinear resizes a mask-logit grid to the target size using
// bilinear interpolation, applies sigmoid, and thresholds into a binary image.
func resizeMaskBilinear(logits [][]float32, targetW, targetH int, threshold float32) image.Image {
	srcH := len(logits)
	srcW := len(logits[0])
	img := image.NewGray(image.Rect(0, 0, targetW, targetH))

	for y := range targetH {
		srcY := (float64(y)+0.5)*float64(srcH)/float64(targetH) - 0.5
		y0 := int(math.Floor(srcY))
		y1 := y0 + 1
		wy := srcY - float64(y0)
		for x := range targetW {
			srcX := (float64(x)+0.5)*float64(srcW)/float64(targetW) - 0.5
			x0 := int(math.Floor(srcX))
			x1 := x0 + 1
			wx := srcX - float64(x0)

			cy0 := clampInt(y0, 0, srcH-1)
			cy1 := clampInt(y1, 0, srcH-1)
			cx0 := clampInt(x0, 0, srcW-1)
			cx1 := clampInt(x1, 0, srcW-1)

			v := float64(logits[cy0][cx0])*(1-wy)*(1-wx) +
				float64(logits[cy0][cx1])*(1-wy)*wx +
				float64(logits[cy1][cx0])*wy*(1-wx) +
				float64(logits[cy1][cx1])*wy*wx

			p := sigmoid(float32(v))
			val := byte(0)
			if p > threshold {
				val = 255
			}
			img.SetGray(x, y, color.Gray{Y: val})
		}
	}
	return img
}
