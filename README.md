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

```bash
go build -o sam3-demo ./models/sam3/demo
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -text "buildings"
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -points "512,512,1"
./sam3-demo -input aerial.png -output out.png -bpe bpe_simple_vocab_16e6.txt.gz -boxes "100,100,800,800"
```

### Flags
* `-input`: input image path (PNG/JPEG).
* `-output`: output image path.
* `-model`: HuggingFace repository ID (default `dummy9996/SAM3.1-safetensors-bf16-x-fp8`).
* `-bpe`: path to `bpe_simple_vocab_16e6.txt.gz`.
* `-text`: text prompt.
* `-points`: `x,y,label;...` (label `1` foreground, `0` background).
* `-boxes`: `xmin,ymin,xmax,ymax;...`.
* `-threshold`: detection probability threshold (default `0.5`).
* `-color` / `-format`: overlay color and output format.

## Notes

* All weights are cast to float32 at load time for simplicity; bf16 inference
  is left as a future optimization.
* The tracker / Object Multiplex video stack is intentionally out of scope.
