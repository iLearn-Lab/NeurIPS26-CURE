"""Load a frozen backbone from a model name or a user-supplied checkpoint."""
import os

import clip

from tools import convert_models_to_fp32


def get_model(args, device):
    if args.pre_model == "clip":
        model, _ = clip.load(args.backbone_weights or args.backbone, device=device)
        processor = None
    elif args.pre_model in ("siglip", "siglip2"):
        from transformers import AutoModel, AutoProcessor

        model_name = getattr(args, "model_name", None) or os.environ.get("CURE_MODEL_PATH")
        if not model_name:
            raise ValueError("Set --model-name or CURE_MODEL_PATH to a SigLIP/SigLIP2 model ID or local directory")
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    else:
        raise ValueError(f"Unknown pre_model: {args.pre_model}")
    convert_models_to_fp32(model)
    model.eval().requires_grad_(False)
    return model.to(device), processor
