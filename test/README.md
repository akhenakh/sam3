# SAM 3.1 parity harness

This directory contains the tooling that compares the GoMLX implementation of
SAM 3.1 against the reference PyTorch model, plus the handoff notes from that
investigation. It is intentionally self-contained so a future agent/engineer can
pick up where the previous investigation left off.

## Files

| File | Purpose |
|---|---|
| `generate_test_data.py` | Builds the reference `build_sam3_image_model`, loads the bf16 safetensors checkpoint (preserving the complex RoPE buffers), runs a **text-only** prompt (`"buildings"`) on `test_image.png` in fp32, and dumps tensor statistics to `sam3_test_data.json`. |
| `sam3_test_data.json` | The reference values (token ids, backbone/text/encoder/queries means, `pred_logits`, `pred_boxes` mean, `pred_masks` mean, `presence_logit`). |
| `test_image.png` | Deterministic 1008×1008 synthetic image (gradient + a few colored rectangles). Sized at exactly the model resolution so resize is a no-op and preprocessing is bit-reproducible. |
| `bpe_simple_vocab_16e6.txt.gz` | CLIP BPE vocabulary (copied from the reference `sam3` package). |

The Go side is `parity_test.go` in the package root: `TestSAM3Parity` loads the
GoMLX model, tokenizes the same prompt, runs the same forward pass, and asserts
agreement against `sam3_test_data.json`.

## Environment

This host is ROCm. `torchvision` must be the ROCm build matching `torch`:

```bash
pip install torchvision==0.28.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
```

(The PyPI `torchvision 0.28.0` is a CUDA build and fails at import with
`operator torchvision::nms does not exist`.)

To regenerate the reference values:

```bash
PYTHONPATH=/path/to/reference/sam3 python3 test/generate_test_data.py \
    --safetensors /path/to/sam3.1_multiplex_bf16.safetensors \
    --bpe /path/to/bpe_simple_vocab_16e6.txt.gz
```

To run the parity test (skips in `-short` mode; needs the cached weights):

```bash
go test -run TestSAM3Parity -v
```

## What was verified (and how)

The investigation bisected the model from the outputs backward. The reference
is run in **fp32**: weights are cast to fp32, `sam3.model.vitdet.addmm_act` is
monkeypatched to run `F.linear` + `F.gelu` in fp32 (instead of the fused bf16
op), and there is no autocast. This matches the GoMLX implementation, which is
fp32 throughout, and isolates real bugs from bf16-vs-fp32 quantization noise.

Confirmed **exact** (fp32, matches to ~1e-6):

- `PositionEmbeddingSine` (neck position encodings).
- Decoder `query_pos` (`gen_sineembed_for_position` + `ref_point_head`).
- `boxRPB` (log relative-position bias + `boxRPB_embed_x/y`).
- Geometry-encoder `cls_embed` + `final_proj` + `norm`.
- Geometry-encoder **self-attention**.
- ViT **block-0 input** (patch embed + tiled abs-pos + `ln_pre`).
- RoPE frequencies (`axialCis` cos/sin) match the checkpoint's `freqs_cis`
  buffers exactly, for both window (24×24) and global (72×72) blocks.

## Bugs found and fixed

1. **Decoder query positional-encoding coordinate order.** The reference
   `gen_sineembed_for_position` emits coordinates as `[y, x, w, h]`, but the Go
   implementation emitted `[x, y, w, h]`. Fixed in `transformer.go`
   (`genSineEmbed` now uses order `[1, 0, 2, 3]` for cxcywh input).

2. **Reference-data RoPE corruption (the root cause of the residual gap).** The
   safetensors checkpoint stores the ViT rotary-encoding buffers as
   **complex64** `freqs_cis` tensors (real = cos, imag = sin). The reference
   weight loader cast every value with `.float()`, which silently discards the
   imaginary part of a complex tensor — so the reference model ran with a
   cos-only rotation (no sin), while GoMLX computes cos/sin correctly from
   scratch. This single corruption produced the ~2e-3-per-element ViT attention
   difference and the ~1.25 shift in `pred_logits` mean.

   Fixed in `generate_test_data.py`'s `load_weights`: complex tensors are kept
   as-is, only real tensors are cast to fp32. After the fix, block-0 attention
   intermediates (qkv, post-RoPE q/k, softmax coefficients, proj output) agree
   with GoMLX to ~1e-7, and every output statistic agrees to ~1e-5 (see below).

## Resolution status

With the RoPE corruption fixed and the reference run in fp32, `TestSAM3Parity`
passes at fp32-level tolerances:

- backbone FPN / position encodings / text / encoder / queries: ~1e-5 … ~1e-7.
- `pred_logits` max element diff: ~4e-4 (mean diff is far smaller).
- `pred_masks` mean: ~3e-5; `presence_logit`: ~2e-5.
- `pred_boxes` mean: ~5e-3 (the largest residual; it is fp32 accumulation
  through the 6-layer box refinement sigmoid/inverse-sigmoid, not a bug — the
  `inverse_sigmoid` implementation was verified to match the reference).

The GoMLX implementation is now numerically correct against the fp32 reference;
no residual discrepancy remains open.
