#!/usr/bin/env python3
"""Generate reference values for the SAM 3.1 GoMLX parity test.

Runs the reference PyTorch SAM 3.1 image model (text-only prompt) on the test
image and dumps intermediate/final tensor statistics to ``sam3_test_data.json``.

This host uses ROCm, so torchvision must be the ROCm build matching torch:

    pip install torchvision==0.28.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2

Usage:
    python3 generate_test_data.py --safetensors /path/to/sam3.1_multiplex_bf16.safetensors \
        --bpe /path/to/bpe_simple_vocab_16e6.txt.gz [--image test_image.png] \
        [--text "buildings"] [--out sam3_test_data.json]
"""

import argparse
import json
import sys

import numpy as np
import torch

# Import the reference model (assumes the sam3 Python package is on sys.path).
from sam3.model.data_misc import FindStage  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


def preprocess_image(path, resolution=1008, device="cuda"):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    arr = (arr - 0.5) / 0.5  # normalize with mean/std 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, img.size


def load_weights(model, safetensors_path):
    from safetensors.torch import load_file

    ckpt = load_file(safetensors_path, device="cpu")
    # Cast mixed bf16/f32 to f32 and strip the "detector." prefix (the reference
    # image model exposes weights without it).
    state_dict = {k: v.float() for k, v in ckpt.items() if k.startswith("detector.")}
    state_dict = {k[len("detector."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"unexpected keys ({len(unexpected)}): {unexpected[:5]}...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="test_image.png")
    p.add_argument("--text", default="buildings")
    p.add_argument("--safetensors", default=None, required=True)
    p.add_argument("--bpe", default=None, required=True)
    p.add_argument("--out", default="sam3_test_data.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print("Building reference model...")
    model = build_sam3_image_model(
        bpe_path=args.bpe,
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
    )
    print("Loading weights...")
    load_weights(model, args.safetensors)
    model = model.to(device).eval()

    print("Preprocessing image...")
    image, _ = preprocess_image(args.image, resolution=1008, device=device)

    state = {}
    # The reference model uses bf16 fused ops (addmm_act); run under autocast,
    # matching Sam3BasePredictor.add_prompt.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state["backbone_out"] = model.backbone.forward_image(image)
        text_outputs = model.backbone.forward_text([args.text], device=device)
        state["backbone_out"].update(text_outputs)

        find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = model._get_dummy_prompt()

        print("Running forward...")
        outputs = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    backbone_out = state["backbone_out"]

    def mean(x):
        return float(x.float().mean().item())

    # Dump the actual token ids produced by the reference tokenizer.
    tokenized = model.backbone.language_backbone.tokenizer(
        [args.text], context_length=32
    )
    data = {
        "text": args.text,
        "token_ids": [int(v) for v in tokenized[0].tolist()],
        "backbone_fpn_means": [mean(x) for x in backbone_out["backbone_fpn"]],
        "vision_pos_enc_means": [mean(x) for x in backbone_out["vision_pos_enc"]],
        "language_features_mean": mean(backbone_out["language_features"]),
        "encoder_hidden_states_mean": mean(outputs["encoder_hidden_states"]),
        "queries_mean": mean(outputs["queries"]),
        "pred_logits": [float(v) for v in outputs["pred_logits"][0, :, 0].cpu().tolist()],
        "pred_boxes_mean": mean(outputs["pred_boxes"]),
        "pred_masks_mean": mean(outputs["pred_masks"]),
        "presence_logit": [float(v) for v in outputs["presence_logit_dec"][0].cpu().tolist()],
    }

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"  pred_logits[:5] = {data['pred_logits'][:5]}")
    print(f"  presence_logit  = {data['presence_logit']}")
    print(f"  pred_masks_mean = {data['pred_masks_mean']}")


if __name__ == "__main__":
    main()
