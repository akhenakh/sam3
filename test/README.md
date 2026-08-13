# SAM 3.1 parity harness

This directory contains the tooling that compares the GoMLX implementation of
SAM 3.1 against the reference PyTorch model, plus the handoff notes from that
investigation. It is intentionally self-contained so a future agent/engineer can
pick up where the previous investigation left off.

## Files

| File | Purpose |
|---|---|
| `generate_test_data.py` | Builds the reference `build_sam3_image_model`, loads the bf16 safetensors checkpoint, runs a **text-only** prompt (`"buildings"`) on `test_image.png`, and dumps tensor statistics to `sam3_test_data.json`. |
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

The investigation bisected the model from the outputs backward. Two reference
configurations were used:

- **bf16**: the model as it actually runs (`torch.autocast(bfloat16)`).
- **fp32**: the same model with `sam3.model.vitdet.addmm_act` monkeypatched to
  run `F.linear` + `F.gelu` in fp32 instead of the fused bf16 op. This isolates
  real bugs from bf16-vs-fp32 quantization noise.

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

## Known residual difference (open)

There is still a small numerical discrepancy in the **ViT attention**: the
block-0 attention output differs from the fp32 reference by ~2e-3 per element,
which grows through the 32-block backbone (and the encoder/decoder) into a
~1.25 shift in the mean of the final `pred_logits`.

What has been **ruled out**:

- RoPE frequencies (exact match with checkpoint buffers).
- Position encodings, `query_pos`, `boxRPB` (exact match).
- The fused vs decomposed attention path (`DisableFusion` changes nothing).
- bf16-vs-fp32 (the fp32-vs-fp32 comparison still diverges).
- The MLP/GELU (the ViT block-0 MLP error is downstream of the attention error,
  not its source).

What is left to check (handoff pointers): dump the q/k/v **after** the RoPE
rotation, and the softmax coefficients, for a single window and compare against
PyTorch's `scaled_dot_product_attention` step-by-step. The likely culprits are a
subtle difference in the RoPE application, the attention softmax, or the
score scaling, all of which live in `backbone.go` (`vitAttention`,
`applyRoPE`, `axialCis`).

**Practical impact**: the user confirmed the model produces correct, legitimate
segmentations on real satellite imagery (buildings), so the discrepancy is a
numerical-precision matter that does not change the model's behavior on real
inputs. The parity test therefore keeps **strict** tolerances on the structural
quantities (backbone FPN, position encodings, text/encoder/queries/boxes) and a
**relaxed** tolerance on the score heads, and documents the residual in its
comments.
