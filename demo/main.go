// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Demo command-line program to segment an aerial/arbitrary image with SAM 3.1
// using text, point, and/or box prompts.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"log"
	"os"
	"strings"

	"github.com/akhenakh/sam3"
	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/hub"
	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	inputFlag  = flag.String("input", "", "Path to the input image file (JPEG or PNG)")
	outputFlag = flag.String("output", "output.png", "Path to save the output segmented image")
	modelFlag  = flag.String("model", "dummy9996/SAM3.1-safetensors-bf16-x-fp8", "HuggingFace repo ID containing the bf16 safetensors checkpoint")
	bpeFlag    = flag.String("bpe", "", "Path to the gzipped BPE vocab file (bpe_simple_vocab_16e6.txt.gz)")
	textFlag   = flag.String("text", "", "Text prompt (e.g. \"buildings\")")
	pointsFlag = flag.String("points", "", "Points in format 'x1,y1,label1;...' (label: 1=fg, 0=bg)")
	boxesFlag  = flag.String("boxes", "", "Bounding boxes in format 'x_min,y_min,x_max,y_max;...'")
	threshold  = flag.Float64("threshold", 0.5, "Detection probability threshold")
	formatFlag = flag.String("format", "", "Force output image format ('png' or 'jpg')")
	colorFlag  = flag.String("color", "red", "Mask overlay color")
)

func main() {
	flag.Parse()

	if *inputFlag == "" {
		log.Fatalf("Error: -input image flag is required.")
	}
	if *textFlag == "" && *pointsFlag == "" && *boxesFlag == "" {
		log.Fatalf("Error: at least one of -text, -points, or -boxes must be provided.")
	}
	if *bpeFlag == "" {
		log.Fatalf("Error: -bpe is required (path to bpe_simple_vocab_16e6.txt.gz).")
	}

	points, err := parsePoints(*pointsFlag)
	if err != nil {
		log.Fatalf("Failed to parse points: %v", err)
	}
	boxes, err := parseBoxes(*boxesFlag)
	if err != nil {
		log.Fatalf("Failed to parse boxes: %v", err)
	}

	img, err := loadImage(*inputFlag)
	if err != nil {
		log.Fatalf("Failed to load input image: %v", err)
	}

	repo := hub.New(*modelFlag)
	if err := repo.DownloadInfo(false); err != nil {
		log.Fatalf("Failed to download model info: %v", err)
	}

	modelObj, err := sam3.LoadModel(repo)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Println("Initializing GoMLX backend...")
	backend, err := compute.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}
	defer backend.Finalize()

	fmt.Println("Creating Segmenter...")
	segmenter, err := sam3.NewSegmenter(backend, modelObj, *bpeFlag)
	if err != nil {
		log.Fatalf("Failed to create segmenter: %v", err)
	}

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

	fmt.Printf("Found %d detection(s).\n", len(detections))
	if len(detections) == 0 {
		log.Fatalf("No detections above threshold %g", *threshold)
	}

	for i, det := range detections {
		outImg := overlayMask(img, det.Mask, parseColor(*colorFlag))
		path := *outputFlag
		if len(detections) > 1 {
			path = formatIndexedPath(*outputFlag, i)
		}
		fmt.Printf("Saving detection %d (score %.4f, box %d,%d,%d,%d) to %s...\n",
			i, det.Score, det.Box.MinX, det.Box.MinY, det.Box.MaxX, det.Box.MaxY, path)
		if err := saveImage(outImg, path, *formatFlag); err != nil {
			log.Fatalf("Failed to save output image: %v", err)
		}
	}
	fmt.Println("Done!")
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func saveImage(img image.Image, path, format string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	format = strings.ToLower(format)
	if format == "" {
		if strings.HasSuffix(strings.ToLower(path), ".jpg") || strings.HasSuffix(strings.ToLower(path), ".jpeg") {
			format = "jpg"
		} else {
			format = "png"
		}
	}
	if format == "jpg" || format == "jpeg" {
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 90})
	}
	return png.Encode(f, img)
}

func formatIndexedPath(originalPath string, index int) string {
	extIdx := strings.LastIndex(originalPath, ".")
	if extIdx == -1 {
		return fmt.Sprintf("%s_%d", originalPath, index)
	}
	return fmt.Sprintf("%s_%d%s", originalPath[:extIdx], index, originalPath[extIdx:])
}

func parseColor(s string) color.RGBA {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "red":
		return color.RGBA{R: 255, G: 0, B: 0, A: 255}
	case "green":
		return color.RGBA{R: 0, G: 255, B: 0, A: 255}
	case "blue":
		return color.RGBA{R: 0, G: 0, B: 255, A: 255}
	case "yellow":
		return color.RGBA{R: 255, G: 255, B: 0, A: 255}
	case "magenta":
		return color.RGBA{R: 255, G: 0, B: 255, A: 255}
	case "cyan":
		return color.RGBA{R: 0, G: 255, B: 255, A: 255}
	case "gray", "grey":
		return color.RGBA{R: 128, G: 128, B: 128, A: 255}
	}
	if strings.HasPrefix(s, "#") {
		var r, g, b uint8
		if n, _ := fmt.Sscanf(s, "#%02x%02x%02x", &r, &g, &b); n == 3 {
			return color.RGBA{R: r, G: g, B: b, A: 255}
		}
	}
	var r, g, b int
	if n, _ := fmt.Sscanf(s, "%d,%d,%d", &r, &g, &b); n == 3 {
		return color.RGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255}
	}
	return color.RGBA{R: 255, G: 0, B: 0, A: 255}
}

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

func overlayMask(orig image.Image, mask image.Image, maskColor color.RGBA) image.Image {
	bounds := orig.Bounds()
	out := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := orig.At(x, y).RGBA()
			r8, g8, b8, a8 := uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)
			if mask.At(x-bounds.Min.X, y-bounds.Min.Y).(color.Gray).Y > 0 {
				out.Set(x, y, color.RGBA{
					R: uint8((uint16(r8) + uint16(maskColor.R)) / 2),
					G: uint8((uint16(g8) + uint16(maskColor.G)) / 2),
					B: uint8((uint16(b8) + uint16(maskColor.B)) / 2),
					A: a8,
				})
			} else {
				out.Set(x, y, color.RGBA{R: r8, G: g8, B: b8, A: a8})
			}
		}
	}
	return out
}
