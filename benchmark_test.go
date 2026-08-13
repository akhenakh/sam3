// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"image"
	_ "image/png"
	"os"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/support/testutil"
)

// benchModelRepo is the HuggingFace repo holding the bf16 safetensors checkpoint.
const benchModelRepo = "dummy9996/SAM3.1-safetensors-bf16-x-fp8"

// benchImagePath is the input image used by the benchmarks.
const benchImagePath = "classify/boats.png"

// newBenchSegmenter loads the model and compiles the segmenter exactly once per
// benchmark run. It is not part of the timed section, so benchmark numbers
// reflect inference only.
func newBenchSegmenter(tb testing.TB) *Segmenter {
	tb.Helper()
	backend := testutil.BuildTestBackend()

	repo := hub.New(benchModelRepo)
	if err := repo.DownloadInfo(false); err != nil {
		tb.Fatalf("failed to download model info: %v", err)
	}
	modelObj, err := LoadModel(repo)
	if err != nil {
		tb.Fatalf("failed to load model: %v", err)
	}
	seg, err := NewSegmenter(backend, modelObj, "test/bpe_simple_vocab_16e6.txt.gz")
	if err != nil {
		tb.Fatalf("failed to create segmenter: %v", err)
	}
	return seg
}

func loadBenchImage(tb testing.TB) image.Image {
	tb.Helper()
	f, err := os.Open(benchImagePath)
	if err != nil {
		tb.Fatalf("failed to open image: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		tb.Fatalf("failed to decode image: %v", err)
	}
	return img
}

// BenchmarkSegment measures end-to-end segmentation (image -> tensors -> graph
// execution -> mask post-processing), matching the demo CLI path. The one-time
// model load + graph compilation happens in the warm-up call, outside the timed
// section, so the reported numbers reflect steady-state inference.
func BenchmarkSegment(b *testing.B) {
	seg := newBenchSegmenter(b)
	img := loadBenchImage(b)
	opts := &SegmentationOptions{Text: "boats", Threshold: 0.5}

	// Warm up: trigger the one-time graph compilation outside the timed section.
	if _, err := seg.Segment(img, opts); err != nil {
		b.Fatalf("warmup segment failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detections, err := seg.Segment(img, opts)
		if err != nil {
			b.Fatalf("segment failed: %v", err)
		}
		if len(detections) == 0 {
			b.Fatalf("no detections returned")
		}
	}
}
