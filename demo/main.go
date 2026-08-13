// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Command sam3-demo segments a single image with SAM 3.1 (Segment Anything
// Model 3.1) and writes one annotated image containing every detection.
//
// SAM 3.1 is Meta's foundation model for image grounding / segmentation. Unlike
// a classifier that assigns one label to the whole image, this model takes a
// *prompt* and returns a set of *instance* segmentations — per-object binary
// masks plus a bounding box and a confidence score — each of which "answers"
// the prompt somewhere in the image.
//
// The demo supports the three prompt modalities the image model understands:
//
//   - text   (-text "buildings"): a natural-language query encoded by the
//     model's CLIP-style text encoder.
//   - points (-points "x,y,label"): click-like points; label 1 = foreground
//     (this object), label 0 = background (not this object).
//   - boxes  (-boxes "xmin,ymin,xmax,ymax"): a tight bounding box around an
//     object.
//
// All three are optional, but at least one must be given. For aerial imagery,
// text prompts such as "buildings", "roads", or "vehicles" are usually the most
// convenient.
//
// Example:
//
//	go run . -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -text "buildings"
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/jpeg" // register the JPEG decoder
	"image/png"
	_ "image/png" // register the PNG decoder
	"log"
	"os"
	"strings"

	"github.com/akhenakh/sam3"
	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/hub"
	_ "github.com/gomlx/gomlx/backends/default" // register the default GoMLX compute backend
)

// Command-line flags. Each flag maps to one part of the segmentation pipeline:
// an input image, a prompt, and an output location.
var (
	inputFlag  = flag.String("input", "", "Path to the input image file (JPEG or PNG)")
	outputFlag = flag.String("output", "output.png", "Path to save the annotated image (all detections in one file)")
	modelFlag  = flag.String("model", "dummy9996/SAM3.1-safetensors-bf16-x-fp8", "HuggingFace repo ID containing the bf16 safetensors checkpoint")
	bpeFlag    = flag.String("bpe", "", "Path to the gzipped BPE vocab file (bpe_simple_vocab_16e6.txt.gz)")
	textFlag   = flag.String("text", "", "Text prompt (e.g. \"buildings\")")
	pointsFlag = flag.String("points", "", "Point prompts in 'x1,y1,label1;x2,y2,label2' format (label: 1=foreground, 0=background)")
	boxesFlag  = flag.String("boxes", "", "Box prompts in 'xmin,ymin,xmax,ymax;...' format")
	threshold  = flag.Float64("threshold", 0.5, "Minimum detection probability to keep (0..1)")
	formatFlag = flag.String("format", "", "Force output image format ('png' or 'jpg'); defaults to the output file extension")
)

func main() {
	flag.Parse()

	// --- Validate the command line ------------------------------------------
	// An input image and at least one prompt are mandatory; the BPE vocabulary
	// is required because the text encoder needs the same tokenizer the model
	// was trained with.
	if *inputFlag == "" {
		log.Fatalf("Error: -input image flag is required.")
	}
	if *textFlag == "" && *pointsFlag == "" && *boxesFlag == "" {
		log.Fatalf("Error: at least one of -text, -points, or -boxes must be provided.")
	}
	if *bpeFlag == "" {
		log.Fatalf("Error: -bpe is required (path to bpe_simple_vocab_16e6.txt.gz).")
	}

	// Parse the (optional) geometric prompts from their textual form into the
	// structured types the Segmenter expects.
	points, err := parsePoints(*pointsFlag)
	if err != nil {
		log.Fatalf("Failed to parse points: %v", err)
	}
	boxes, err := parseBoxes(*boxesFlag)
	if err != nil {
		log.Fatalf("Failed to parse boxes: %v", err)
	}

	// --- Load the input image ----------------------------------------------
	// image.Decode sniffs the format (JPEG/PNG) from the file content, so the
	// extension does not matter.
	img, err := loadImage(*inputFlag)
	if err != nil {
		log.Fatalf("Failed to load input image: %v", err)
	}

	// --- Download the model metadata and weights ---------------------------
	// The checkpoint is a ~1.7 GB bf16 safetensors file; hub.New + DownloadInfo
	// only fetch the repo metadata first, and the weights are streamed lazily
	// by sam3.LoadModel/NewSegmenter (cached on disk afterwards).
	repo := hub.New(*modelFlag)
	if err := repo.DownloadInfo(false); err != nil {
		log.Fatalf("Failed to download model info: %v", err)
	}

	modelObj, err := sam3.LoadModel(repo)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// --- Create the GoMLX compute backend ----------------------------------
	// compute.New picks the default backend (GPU if available, otherwise CPU).
	// All tensor operations run on this backend.
	fmt.Println("Initializing GoMLX backend...")
	backend, err := compute.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}
	defer backend.Finalize()

	// --- Build the Segmenter ------------------------------------------------
	// NewSegmenter loads the weights into a model.Store, builds the full SAM 3.1
	// computation graph, and compiles it. This is the slow one-time step; the
	// compiled graph is then reused for every subsequent Segment call.
	fmt.Println("Creating Segmenter...")
	segmenter, err := sam3.NewSegmenter(backend, modelObj, *bpeFlag)
	if err != nil {
		log.Fatalf("Failed to create segmenter: %v", err)
	}

	// --- Run inference ------------------------------------------------------
	// Segment returns the detections sorted by descending confidence. Each
	// detection carries a binary mask (resized to the original image size), a
	// bounding box, and a score in [0, 1].
	options := &sam3.SegmentationOptions{
		Text:      *textFlag,
		Points:    points,
		Boxes:     boxes,
		Threshold: *threshold,
	}

	fmt.Println("Running inference...")
	detections, err := segmenter.Segment(img, options)
	if err != nil {
		log.Fatalf("Segmentation failed: %v", err)
	}

	fmt.Printf("Found %d detection(s) above threshold %g.\n", len(detections), *threshold)
	if len(detections) == 0 {
		log.Fatalf("No detections above threshold %g; try lowering -threshold.", *threshold)
	}

	// Print a human-readable table of the detections (score + box).
	for i, det := range detections {
		fmt.Printf("  detection %d: score %.4f  box (%d,%d)-(%d,%d)\n",
			i, det.Score, det.Box.MinX, det.Box.MinY, det.Box.MaxX, det.Box.MaxY)
	}

	// --- Render and save a single annotated image --------------------------
	// Every detection is overlaid onto the same image, each with a distinct
	// color so overlapping/adjacent instances stay distinguishable.
	annotated := overlayDetections(img, detections)
	if err := saveImage(annotated, *outputFlag, *formatFlag); err != nil {
		log.Fatalf("Failed to save output image: %v", err)
	}
	fmt.Printf("Saved %d detection(s) to %s\n", len(detections), *outputFlag)
	fmt.Println("Done!")
}

// palette is a small set of visually distinct colors used to draw detections.
// If there are more detections than colors, the colors cycle.
var palette = []color.RGBA{
	{R: 255, G: 0, B: 0, A: 255},   // red
	{R: 0, G: 180, B: 0, A: 255},   // green
	{R: 0, G: 100, B: 255, A: 255}, // blue
	{R: 255, G: 180, B: 0, A: 255}, // amber
	{R: 255, G: 0, B: 255, A: 255}, // magenta
	{R: 0, G: 180, B: 180, A: 255}, // cyan
	{R: 220, G: 100, B: 0, A: 255}, // orange
	{R: 150, G: 0, B: 220, A: 255}, // purple
}

// maskFillAlpha is the opacity used when painting a detection mask over the
// original image (0 = invisible, 255 = fully opaque). A mid value keeps the
// underlying image visible through the highlight.
const maskFillAlpha = 96

// overlayDetections draws every detection onto a single copy of the original
// image. Each detection gets its own color: the mask is painted as a
// semi-transparent fill and the bounding box is drawn as an opaque outline.
func overlayDetections(orig image.Image, detections []sam3.Detection) image.Image {
	bounds := orig.Bounds()
	out := image.NewRGBA(bounds)
	// Start from a plain copy of the original image.
	draw.Draw(out, bounds, orig, bounds.Min, draw.Src)

	for i, det := range detections {
		c := palette[i%len(palette)]
		fillMask(out, det.Mask, c, maskFillAlpha)
		drawBoxOutline(out, det.Box, c)
	}
	return out
}

// fillMask alpha-blends `c` over `out` wherever the binary mask is set. The
// mask is a *image.Gray with value 0 (outside) or 255 (inside); `alpha`
// controls how strongly the color shows through.
func fillMask(out *image.RGBA, mask image.Image, c color.RGBA, alpha uint8) {
	bounds := out.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// mask.At indexes from (0,0), so offset by the image bounds origin.
			if mask.At(x-bounds.Min.X, y-bounds.Min.Y).(color.Gray).Y == 0 {
				continue
			}
			// Standard alpha compositing: out = src*(1-a) + color*a.
			src := out.RGBAAt(x, y)
			a := uint16(alpha)
			out.SetRGBA(x, y, color.RGBA{
				R: uint8((uint16(src.R)*(255-a) + uint16(c.R)*a) / 255),
				G: uint8((uint16(src.G)*(255-a) + uint16(c.G)*a) / 255),
				B: uint8((uint16(src.B)*(255-a) + uint16(c.B)*a) / 255),
				A: 255,
			})
		}
	}
}

// drawBoxOutline draws a 2-pixel-thick rectangle outline of `c` onto `out` at
// the (pixel-coordinate) box position, clamped to the image bounds.
func drawBoxOutline(out *image.RGBA, box sam3.PromptBox, c color.RGBA) {
	bounds := out.Bounds()
	x0 := clampInt(box.MinX, bounds.Min.X, bounds.Max.X-1)
	y0 := clampInt(box.MinY, bounds.Min.Y, bounds.Max.Y-1)
	x1 := clampInt(box.MaxX, bounds.Min.X, bounds.Max.X-1)
	y1 := clampInt(box.MaxY, bounds.Min.Y, bounds.Max.Y-1)

	// Draw each edge as two adjacent lines so the outline is visible against
	// busy aerial imagery.
	hline := func(y int) {
		for x := x0; x <= x1; x++ {
			out.SetRGBA(x, y, c)
		}
	}
	vline := func(x int) {
		for y := y0; y <= y1; y++ {
			out.SetRGBA(x, y, c)
		}
	}
	hline(y0)
	hline(y0 + 1)
	hline(y1)
	hline(y1 - 1)
	vline(x0)
	vline(x0 + 1)
	vline(x1)
	vline(x1 - 1)
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

// loadImage opens and decodes an image file, sniffing the format from content.
func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// saveImage writes img to path as PNG or JPEG. The format is taken from the
// explicit -format flag, falling back to the file extension, and finally PNG.
func saveImage(img image.Image, path, format string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	format = strings.ToLower(format)
	if format == "" {
		switch {
		case strings.HasSuffix(strings.ToLower(path), ".jpg"), strings.HasSuffix(strings.ToLower(path), ".jpeg"):
			format = "jpg"
		default:
			format = "png"
		}
	}
	if format == "jpg" || format == "jpeg" {
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 90})
	}
	return png.Encode(f, img)
}

// parsePoints turns "x,y,label;..." into []sam3.PromptPoint. Labels use the
// SAM convention: 1 = foreground (clicked on the object), 0 = background.
func parsePoints(s string) ([]sam3.PromptPoint, error) {
	if s == "" {
		return nil, nil
	}
	var points []sam3.PromptPoint
	for _, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var x, y, labelVal int
		if _, err := fmt.Sscanf(part, "%d,%d,%d", &x, &y, &labelVal); err != nil {
			return nil, fmt.Errorf("invalid point format %q", part)
		}
		points = append(points, sam3.PromptPoint{X: x, Y: y, Label: sam3.PointLabel(labelVal)})
	}
	return points, nil
}

// parseBoxes turns "xmin,ymin,xmax,ymax;..." into []sam3.PromptBox.
func parseBoxes(s string) ([]sam3.PromptBox, error) {
	if s == "" {
		return nil, nil
	}
	var boxes []sam3.PromptBox
	for _, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var minX, minY, maxX, maxY int
		if _, err := fmt.Sscanf(part, "%d,%d,%d,%d", &minX, &minY, &maxX, &maxY); err != nil {
			return nil, fmt.Errorf("invalid box format %q", part)
		}
		boxes = append(boxes, sam3.PromptBox{MinX: minX, MinY: minY, MaxX: maxX, MaxY: maxY})
	}
	return boxes, nil
}
