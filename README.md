# SAM 3.1: Segment Anything Model 3.1 in GoMLX

This package implements the **image grounding / segmentation** part of Facebook's
**SAM 3.1** foundation model, designed to run in Go using
[GoMLX](https://github.com/gomlx/gomlx) for hardware-accelerated tensor
computations.

It implements the image part of the model (the ViT-Det vision backbone, ViTDet
neck, CLIP-style text encoder, DETR-style transformer encoder/decoder, geometry
encoder, and universal segmentation head). The video tracking part (Object
Multiplex) is **not** implemented.


## Methodoly

Using DeepSeek V4 Pro, tasked to implement the GoMLX SAM3 version based on the Facebook implementations.
Cost: 0.60$

## References
* **Model**: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
* **Reference PyTorch Implementation**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
* **Paper**: [SAM 3](https://arxiv.org/abs/2511.16719)

## Checkpoint

The official `sam3.1_multiplex.pt` checkpoint is a gated PyTorch pickle. This
package loads a **bf16 safetensors** conversion instead, e.g.
[dummy9996/SAM3.1-safetensors-bf16-x-fp8](https://huggingface.co/dummy9996/SAM3.1-safetensors-bf16-x-fp8)
(`sam3.1_multiplex_bf16.safetensors`).

The BPE vocabulary (`bpe_simple_vocab_16e6.txt.gz`) ships with the reference
`sam3` package under `sam3/assets/` and must be provided to `NewSegmenter`.

## API

### High-level segmenter
* **[NewSegmenter](file:///models/sam3/sam3.go)**: loads the config/weights,
  compiles the computation graph, and returns a `*Segmenter`.
* **[Segment](file:///models/sam3/sam3.go)**: runs inference and returns
  `[]Detection` (binary masks, bounding boxes, and scores) sorted by score.

### Low-level graph API
* **[Forward](file:///models/sam3/sam3.go)**: builds the full SAM 3.1 image
  grounding graph from `*Node` inputs.

## Usage

```go
backend, _ := compute.New()
defer backend.Finalize()

repo := hub.New("dummy9996/SAM3.1-safetensors-bf16-x-fp8")
modelObj, _ := sam3.LoadModel(repo)

segmenter, _ := sam3.NewSegmenter(backend, modelObj, "bpe_simple_vocab_16e6.txt.gz")

detections, _ := segmenter.Segment(img, &sam3.SegmentationOptions{
    Text:      "buildings",
    Threshold: 0.5,
})
best := detections[0] // best.Mask is a binary image.Image, best.Box is the bbox
```

## Demo CLI

The demo segments a single image and writes **one** annotated image containing
every detection, each drawn in a distinct color with its bounding box.

![Example output](img/out.png)

```bash
go build -o sam3-demo ./demo
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -text "buildings"
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -points "512,512,1"
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -boxes "100,100,800,800"
```

### Flags
* `-input`: input image path (PNG/JPEG).
* `-output`: output image path (single file with all detections overlaid).
* `-model`: HuggingFace repository ID (default `dummy9996/SAM3.1-safetensors-bf16-x-fp8`).
* `-bpe`: path to `bpe_simple_vocab_16e6.txt.gz`.
* `-text`: text prompt.
* `-points`: `x,y,label;...` (label `1` foreground, `0` background).
* `-boxes`: `xmin,ymin,xmax,ymax;...`.
* `-threshold`: detection probability threshold (default `0.5`).
* `-format`: output image format (`png`/`jpg`, default inferred from `-output`).

## ROCm (AMD GPU)

GoMLX's XLA auto-installer only ships CPU, CUDA, and TPU PJRT plugins, so AMD
GPUs require installing the ROCm plugin manually.

1. Install the ROCm PJRT plugin. Pick the wheel matching your ROCm version (the
   example uses `rocm7.2.4`):

   ```bash
   curl -o /tmp/jax_rocm7_pjrt.whl \
     "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux_2_28_x86_64.whl"
   unzip -o /tmp/jax_rocm7_pjrt.whl "jax_plugins/xla_rocm7/xla_rocm_plugin.so" -d /tmp/rocm_pjrt
   mkdir -p ~/.local/lib/go-xla
   mv /tmp/rocm_pjrt/jax_plugins/xla_rocm7/xla_rocm_plugin.so \
     ~/.local/lib/go-xla/pjrt_c_api_rocm_plugin.so
   ```

   The plugin's `RUNPATH` points at `/opt/rocm/lib`, so it binds to the system
   ROCm libraries with no extra `LD_LIBRARY_PATH`.

2. Run with the `rocm` backend:

   ```bash
   export GOMLX_BACKEND=xla:rocm
   export XLA_FLAGS="--xla_gpu_graph_min_graph_size=999999"
   ```

   `compute.New()` only auto-selects the `cuda`/`cpu` plugins, so
   `GOMLX_BACKEND` is required. The `XLA_FLAGS` setting works around a HIP-graph
   assertion (`hip::GraphNode::SetStream`) seen with some ROCm builds; without
   it the backend may abort during execution.

3. Compilation vs. inference trade-off. The XLA ROCm plugin takes ~45s to
   compile the SAM 3.1 graph (vs. ~9s for CPU), so a single one-off `sam3-demo`
   run finishes faster on CPU. Once compiled, ROCm inference is ~22× faster
   (~0.7s vs. ~16s per image on `classify/boats.png`), which makes ROCm the
   better choice for batch or service workloads:

   ```bash
   GOMLX_BACKEND=xla:rocm XLA_FLAGS="--xla_gpu_graph_min_graph_size=999999" \
     go test -run '^$' -bench BenchmarkSegment -benchtime=5x -benchmem
   ```

### PyTorch comparison (RX 7900 XTX, ROCm 7.2, SAM 3.1 @ 1008×1008, text prompt)

On the same task (segment `classify/boats.png` with the text prompt "boats"),
GoMLX's compiled XLA kernels are faster than PyTorch eager, but the host-side
mask post-processing and the one-time compile narrow the end-to-end gap:

| | GoMLX (`xla:rocm`) | PyTorch (ROCm) |
|---|---|---|
| one-time setup (load + compile/warmup) | ~49 s (47 s XLA compile) | ~7 s |
| steady-state model forward | ~390 ms | ~481 ms |
| steady-state end-to-end `Segment` | ~730 ms | ~481 ms (forward only) |

GoMLX's forward is ~19% faster, but `Segment` spends ~340 ms on the CPU
bilinearly upsampling the 122 detected masks to the original resolution — a
host-side cost PyTorch doesn't pay in this comparison. Moving that upsampling
into the graph would put GoMLX ahead end-to-end. PyTorch also starts ~7× faster
because it has no XLA compilation step.

## Notes

* All weights are cast to float32 at load time for simplicity; bf16 inference
  is left as a future optimization.
* The tracker / Object Multiplex video stack is intentionally out of scope.

## Parity testing

A parity harness compares the GoMLX implementation against reference values
produced by the PyTorch model:

* `test/generate_test_data.py` — builds the reference `build_sam3_image_model`,
  loads the bf16 safetensors checkpoint, runs a text-only prompt on
  `test/test_image.png`, and dumps intermediate/final tensor statistics to
  `test/sam3_test_data.json`.
* `test/sam3_test_data.json` — the generated reference values.
* `parity_test.go` — `TestSAM3Parity` loads the GoMLX model, runs the same
  forward pass, and asserts agreement.

```bash
# Regenerate the reference values (requires torch + ROCm torchvision):
python3 test/generate_test_data.py \
    --safetensors /path/to/sam3.1_multiplex_bf16.safetensors \
    --bpe /path/to/bpe_simple_vocab_16e6.txt.gz

# Run the parity test:
go test -run TestSAM3Parity -v
```

The reference model is run in **fp32** (no autocast, fused bf16 MLP replaced)
and its complex RoPE buffers are preserved, so it matches this float32
implementation directly. Structural quantities (backbone FPN, position
encodings, text features, encoder memory, decoder queries) agree to ~1e-5; the
detection-score heads (dot-product scoring, presence head, mask logits, box
regression) amplify the accumulated rounding and are checked with a slightly
relaxed tolerance.
