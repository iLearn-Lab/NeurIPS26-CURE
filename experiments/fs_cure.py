"""Canonical CURE release: raw probability margin and TextMLP LR 3e-4.

This is the single public training entry point for the final paper results. It
consolidates the verified ``raw_margin_text_lr_3e4`` experiment: the margin is
the unnormalized top-1 minus top-2 probability gap, and the TextMLP learning
rate is 3e-4. Archived experiment suites remain the provenance record for the
reported checkpoints and metrics.
"""

import argparse
from torch.nn import functional as F
from torchvision.transforms import (
    Compose,
    Resize,
    Lambda,
    ToTensor,
    CenterCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms import InterpolationMode, RandomErasing
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import numpy as np
import time
import os
import sys
import collections
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clip
from cfg import DOWNSTREAM_PATH
from tools import (
    clip_attr_classifier,
    convert_models_to_fp32,
    getLogitsCLIP,
    getLogitsSigLIP,
    set_seed,
)
from datasets.ssl_utils import build_ssl_dataloaders, FixMatchDataLoader
from methods.vp import PaddingVR
from methods.backbones import get_model



CURE_RELEASE_VERSION = "cure-raw-margin-textlr3e4-v1"


def top2_margin(probs: torch.Tensor):
    # probs: (B, C)
    top2 = torch.topk(probs, k=2, dim=1).values  # (B,2)
    margin = top2[:, 0] - top2[:, 1]
    return margin


class TextMLP(torch.nn.Module):
    def __init__(self, dim=512) -> None:
        super().__init__()
        self.dim = dim
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(dim, dim, bias=False),
        )
        torch.nn.init.eye_(self.MLP[0].weight)

    def forward(self, x):
        x = self.MLP(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class AttrVRModule(torch.nn.Module):
    def __init__(
        self,
        model,
        visual_reprogram,
        text_adapter,
        txt_emb_desattr,
        txt_emb_distattr,
        args,
    ):
        super().__init__()
        self.pre_model = model
        self.visual_reprogram = visual_reprogram
        self.text_adapter = text_adapter
        self.args = args
        # Register as buffers so they're replicated to each device under DataParallel
        self.register_buffer("txt_emb_desattr", txt_emb_desattr)
        self.register_buffer("txt_emb_distattr", txt_emb_distattr)

    def train(self, mode=True):
        super().train(mode)
        self.pre_model.eval()
        return self

    def forward(self, x):
        # mode=0: attr logits; mode=1: class logits + similarity
        x = self.visual_reprogram(x)

        if self.text_adapter is not None:
            adapted_txt_emb_desattr = self.text_adapter(self.txt_emb_desattr)
            adapted_txt_emb_distattr = self.text_adapter(self.txt_emb_distattr)
        else:
            adapted_txt_emb_desattr = self.txt_emb_desattr / self.txt_emb_desattr.norm(
                dim=-1, keepdim=True
            )
            adapted_txt_emb_distattr = (
                self.txt_emb_distattr / self.txt_emb_distattr.norm(dim=-1, keepdim=True)
            )

        if self.args.pre_model == "clip":
            x_emb = self.pre_model.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = self.pre_model.logit_scale.exp()
            desattr_logits = getLogitsCLIP(
                self.args.num_attr, exp, x_emb, adapted_txt_emb_desattr, k=self.args.k
            )
            distattr_logits = getLogitsCLIP(
                self.args.num_attr, exp, x_emb, adapted_txt_emb_distattr, k=self.args.k
            )
        elif self.args.pre_model in ("siglip", "siglip2"):
            x_emb = self.pre_model.get_image_features(pixel_values=x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp, bias = self.pre_model.logit_scale.exp(), self.pre_model.logit_bias
            desattr_logits = getLogitsSigLIP(
                self.args.num_attr,
                exp,
                bias,
                x_emb,
                adapted_txt_emb_desattr,
                k=self.args.k,
            )
            distattr_logits = getLogitsSigLIP(
                self.args.num_attr,
                exp,
                bias,
                x_emb,
                adapted_txt_emb_distattr,
                k=self.args.k,
            )
        return (
            desattr_logits * self.args.alpha + distattr_logits * (1 - self.args.alpha),
            desattr_logits,
            distattr_logits,
        )


# ---------------------- Utils ----------------------
def consistency_loss(logits_s, probs_w, weight, T=0.5, use_hard_labels=True):
    """
    logits_s: strong logits [B,C]
    logits_w: weak  logits [B,C]  (teacher)
    weight  : sample-wise weight [B] in [0,1]
    """
    # cross-entropy with hard or soft labels
    if use_hard_labels:
        pseudo = probs_w
        yhat = pseudo.argmax(dim=-1)
        logp = F.log_softmax(logits_s, dim=-1)
        loss = F.nll_loss(logp, yhat, reduction="none")  # [B]
        return (loss * weight).mean(), weight
    else:
        pseudo = probs_w
        logp = F.log_softmax(logits_s, dim=-1)
        nll = torch.sum(-pseudo * logp, dim=1)  # [B]
        return (nll * weight).mean(), weight


class SoftMatchStat:
    def __init__(self, num_classes, device, margin_init, ema_p=0.999):
        self.num_classes = num_classes
        self.device = device
        self.ema_p = ema_p
        self.lb_prob_t = torch.ones(num_classes, device=device) / num_classes
        self.ulb_prob_t = torch.ones(num_classes, device=device) / num_classes
        self.prob_max_mu_t = 1.0 / num_classes
        self.prob_max_var_t = 1.0

        # self.ub_margin_mu_t = margin_init
        self.ub_margin_mu_t = 0
        self.ub_margin_var_t = 1.0

        # self.lb_margin_mu_t = margin_init
        self.lb_margin_mu_t = 0
        self.lb_margin_var_t = 1.0

    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t
        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

        max_probs, _ = ulb_probs.max(dim=-1)
        mu = torch.mean(max_probs).item()
        var = (
            torch.var(max_probs, unbiased=True).item() if max_probs.numel() > 1 else 1.0
        )
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * mu
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * var

    @torch.no_grad()
    def update_margin_t(self, lb_margin, ulb_margin):
        mu = ulb_margin.mean().item()
        var = torch.var(ulb_margin, unbiased=True).item() if ulb_margin.numel() > 1 else 0.0
        if self.ub_margin_mu_t != 0 and self.ub_margin_var_t != 0:
            self.ub_margin_mu_t = (
                self.ema_p * self.ub_margin_mu_t + (1 - self.ema_p) * mu
            )
            self.ub_margin_var_t = (
                self.ema_p * self.ub_margin_var_t + (1 - self.ema_p) * var
            )
        else:
            self.ub_margin_mu_t = mu
            self.ub_margin_var_t = var

        l_mu = lb_margin.mean().item()
        l_var = torch.var(lb_margin, unbiased=True).item() if lb_margin.numel() > 1 else 0.0
        if self.lb_margin_mu_t != 0 and self.lb_margin_var_t != 0:
            self.lb_margin_mu_t = (
                self.ema_p * self.lb_margin_mu_t + (1 - self.ema_p) * l_mu
            )
            self.lb_margin_var_t = (
                self.ema_p * self.lb_margin_var_t + (1 - self.ema_p) * l_var
            )
        else:
            self.lb_margin_mu_t = l_mu
            self.lb_margin_var_t = l_var

    @torch.no_grad()
    def distribution_alignment_probe(self, probs):
        ratio = self.lb_prob_t / (self.ulb_prob_t + 1e-8)
        ratio = ratio.clamp(0.1, 10)
        probs = probs * ratio
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @torch.no_grad()
    def distribution_alignment_margin(self, margin):
        """Return the raw top-1 minus top-2 probability margin unchanged."""
        return margin

    @torch.no_grad()
    def calculate_mask(self, probs, margin):
        max_probs, _ = probs.max(dim=-1)
        mu = float(self.prob_max_mu_t)
        var = max(float(self.prob_max_var_t), 1e-4)
        neg_part = torch.clamp(max_probs - mu, max=0.0)
        mask = torch.exp(-(neg_part**2) / (2.0 * (var / 4.0)))

        if margin is not None:
            mg_mu, mg_var = self.ub_margin_mu_t, self.ub_margin_var_t
            mg_var = max(float(mg_var), 1e-12)
            neg_part_margin = torch.clamp(margin - mg_mu, max=0.0)
            mask_margin = torch.exp(-(neg_part_margin**2) / (2.0 * (mg_var / 4.0)))
        else:
            mask_margin = torch.ones_like(mask)

        mask = (mask + mask_margin) / 2

        return max_probs.detach(), mask.detach()


# ---------------- Attribute-aware pieces ----------------
@torch.no_grad()
def js_divergence(p, q, eps=1e-8):
    """Jensen–Shannon divergence between distributions p, q. p,q:[B,C]"""
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
    js = 0.5 * (kl_pm + kl_qm)  # [B]
    return js


def build_attr_weight(probs_des, probs_dist, probs_fused, use_js=True, beta=1.0):

    # 伪标签来自融合分布
    yhat = probs_fused.argmax(dim=-1)  # [B]
    bidx = torch.arange(yhat.size(0), device=yhat.device)
    conf_geo = torch.sqrt(probs_des[bidx, yhat] * probs_dist[bidx, yhat])  # [B]

    if use_js:
        js = js_divergence(probs_des, probs_dist)  # [B]
        js = torch.clamp(js, 0.0, 1.0)  # 归一化上限1（经验）
        agree = 1.0 - js  # 越一致越接近1
    else:
        agree = torch.ones_like(conf_geo)

    w_attr = torch.clamp(conf_geo * agree, 0.0, 1.0)
    if beta != 1.0:
        w_attr = torch.clamp(w_attr**beta, 0.0, 1.0)
    return w_attr  # [B], in [0,1]


# =========================================================


# Persistence and provenance helpers do not sample training RNGs.
def write_json(path, value):
    """Atomically replace a JSON artifact in its destination directory."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    """Keep the previous checkpoint intact until the new file is complete."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cpu_state(value):
    """Copy state to CPU using only tensors and weights_only-safe containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return value


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def recorded_attributes(classes, model, path, num_attr, processor):
    """Record sampled descriptions without advancing the original Python RNG."""
    data = json.loads(Path(path).read_text())
    local_rng = random.Random()
    local_rng.setstate(random.getstate())
    chosen = {}
    for classname in classes:
        valid = [sentence for sentence in data[classname] if len(sentence) >= 20]
        chosen[classname] = (
            local_rng.sample(valid, num_attr)
            if num_attr < len(valid)
            else local_rng.choices(valid, k=num_attr)
        )
    features = clip_attr_classifier(
        classes, model, path, num_attr=num_attr, processor=processor
    )
    if local_rng.getstate() != random.getstate():
        raise RuntimeError("Attribute sampler differs from the recorded algorithm")
    return features, chosen


def record_data(labeled, unlabeled, test, classes, args, output):
    """Record loader membership without iterating a randomized DataLoader."""

    def rows(loader, unlabeled=False):
        return [
            {
                # Keep dataset-directory symlinks in the recorded relative path.
                "path": str(Path(os.path.abspath(item.impath)).relative_to(Path(DOWNSTREAM_PATH).absolute())),
                "label": int(item.label),
                "classname": classes[item.label] if unlabeled else item.classname,
            }
            for item in loader.dataset.data_source
        ]

    train_rows = rows(labeled)
    test_rows = rows(test)
    unlabeled_rows = rows(unlabeled, unlabeled=True)
    groups = {"train": train_rows, "test": test_rows, "unlabeled": unlabeled_rows}
    path_sets = {key: {row["path"] for row in value} for key, value in groups.items()}
    for key, value in groups.items():
        if len(path_sets[key]) != len(value):
            raise ValueError(f"Duplicate images in {key}")
    if (
        path_sets["train"] & path_sets["test"]
        or path_sets["unlabeled"] & path_sets["test"]
    ):
        raise ValueError("Training pool overlaps with the test set")
    counts = collections.Counter(row["label"] for row in train_rows)
    if set(counts) != set(range(len(classes))) or any(
        n != args.shot for n in counts.values()
    ):
        raise ValueError("Unexpected labeled class counts")
    info = {
        "num_classes": len(classes),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "unlabeled_pool_size": len(unlabeled_rows),
        "labeled_unlabeled_overlap": len(path_sets["train"] & path_sets["unlabeled"]),
        "steps_per_epoch": len(labeled),
        "unlabeled_batch_size": args.bs * args.unlabeled_ratio,
        "train_hash": fingerprint(sorted(train_rows, key=lambda row: row["path"])),
        "test_hash": fingerprint(sorted(test_rows, key=lambda row: row["path"])),
        "unlabeled_hash": fingerprint(unlabeled_rows),
        "train_order_hash": fingerprint(train_rows),
        "test_order_hash": fingerprint(test_rows),
        "class_counts": dict(counts),
    }
    reference_path = ROOT / "reference_splits.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())[
            f"{args.dataset}:{args.seed}"
        ]
        for key in ("train_hash", "test_hash", "num_classes"):
            if info[key] != reference[key]:
                raise ValueError(f"Split differs from previous AttrVR suite: {key}")
        info["reference_splits_match"] = True
    write_json(output / "samples.json", {**groups, "classes": classes})
    write_json(output / "data_audit.json", info)
    return info


def restore_inference(checkpoint, network):
    """Restore every non-frozen component used by the forward pass."""
    network.visual_reprogram.load_state_dict(checkpoint["visual_prompt_dict"])
    network.text_adapter.load_state_dict(checkpoint["text_adapter_dict"])
    with torch.no_grad():
        network.txt_emb_desattr.copy_(checkpoint["txt_emb_desattr"])
        network.txt_emb_distattr.copy_(checkpoint["txt_emb_distattr"])
        network.pre_model.logit_scale.copy_(checkpoint["logit_scale"])
        if "logit_bias" in checkpoint:
            network.pre_model.logit_bias.copy_(checkpoint["logit_bias"])


@torch.no_grad()
def evaluate_saved(network, loader, device):
    network.eval()
    correct = total = 0
    for images, labels in loader:
        logits, _, _ = network(images.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite checkpoint evaluation logits")
        correct += logits.argmax(1).eq(labels.to(device)).sum().item()
        total += labels.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dataset",
        choices=[
            "caltech101",
            "dtd",
            "eurosat",
            "fgvc",
            "food101",
            "oxford_flowers",
            "oxford_pets",
            "stanford_cars",
            "sun397",
            "ucf101",
            "resisc45",
            "imagenet",
        ],
        default="caltech101",
    )
    # AttrVR
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num_attr", type=int, default=2)
    p.add_argument("--k", type=int, default=2)
    # Train
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=40)
    p.add_argument("--linear_lr", type=float, default=3e-4)
    p.add_argument("--input_size", type=int, default=192)
    p.add_argument("--shot", type=int, default=2)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--workers", type=int, default=2, help="Data-loader workers; 0 uses the main process")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # SoftMatch
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--unlabeled_ratio", type=int, default=7)
    p.add_argument("--ema_p", type=float, default=0.999)
    p.add_argument("--margin_init", type=float, default=0.5)
    p.add_argument("--hard_label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dist_align", action=argparse.BooleanOptionalAction, default=True)
    # Attribute-aware weights
    p.add_argument(
        "--pre_model", type=str, default="clip", choices=["clip", "siglip", "siglip2"]
    )
    p.add_argument("--backbone", type=str, default="ViT-B/16")

    p.add_argument(
        "--attr_beta", type=float, default=1.0, help="属性权重幂次，>1更保守"
    )
    p.add_argument(
        "--use_js",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用JS作为分布一致性因子",
    )
    p.add_argument(
        "--consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用属性一致权重",
    )
    p.add_argument("--confidence_thresh", type=float, default=0.95, help="置信度阈值")
    p.add_argument(
        "--use_gaussian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用高斯分布对齐",
    )
    p.add_argument(
        "--top_margin", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--attr_path", type=str, default="gpt3", help="属性路径")

    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone-weights", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None, help="SigLIP/SigLIP2 model ID or local directory")
    p.add_argument(
        "--evaluate", type=Path, help="Reevaluate a complete fs_cure checkpoint"
    )
    p.add_argument(
        "--stop-after", type=int, help="Epoch-boundary stop for verification only"
    )
    args = p.parse_args()
    evaluation_checkpoint = None
    if args.evaluate:
        local_backbone = args.backbone_weights
        local_model_name = args.model_name
        evaluation_checkpoint = torch.load(
            args.evaluate, map_location="cpu", weights_only=True
        )
        for key, value in evaluation_checkpoint["training_args"].items():
            if key not in ("workers", "device"):
                setattr(args, key, value)
        if local_backbone:
            args.backbone_weights = local_backbone
        if local_model_name:
            args.model_name = local_model_name
    if not 0 < args.k <= args.num_attr:
        p.error("Require 0 < k <= num_attr")
    if min(args.epoch, args.shot, args.bs, args.unlabeled_ratio) < 1 or args.workers < 0:
        p.error("epoch, shot, bs, unlabeled_ratio must be positive; workers must be nonnegative")
    if not 0 < args.input_size <= 224:
        p.error("input_size must be between 1 and 224")
    if args.stop_after is not None and args.stop_after < 1:
        p.error("stop-after must be positive")
    if not Path(DOWNSTREAM_PATH).is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {DOWNSTREAM_PATH}. Set CURE_DATA_ROOT to the prepared dataset directory.")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; check the NVIDIA driver or use --device cpu for a small run")
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    # Load first so an automatically downloaded checkpoint also has a hash.
    model, processor = get_model(args, device)
    training_args = {
        key: value
        for key, value in vars(args).items()
        if key not in ("output", "evaluate", "stop_after")
    }
    weights_path = (
        Path(args.backbone_weights)
        if args.backbone_weights
        else Path.home() / ".cache/clip" / (args.backbone.replace("/", "-") + ".pt")
    )
    backbone_reference = {
        "model": args.pre_model,
        "backbone": args.backbone,
        "path": str(weights_path.resolve()),
        "sha256": (
            hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if weights_path.exists()
            else None
        ),
    }
    if (
        evaluation_checkpoint
        and any(evaluation_checkpoint["backbone_reference"][key] != backbone_reference[key]
                for key in ("model", "backbone", "sha256"))
    ):
        raise ValueError("Frozen backbone identity differs from saved checkpoint")
    config = {
        **training_args,
        "output": str(output),
        "schema_version": 1,
        "release_version": CURE_RELEASE_VERSION,
        "margin_mode": "raw_top1_minus_top2",
        "selection": "best_test_accuracy",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0,
        "amp": False,
        "scheduler": "CosineAnnealingLR",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_used": str(device),
        "backbone_reference": backbone_reference,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stop_after": args.stop_after,
    }
    write_json(output / "config.json", config)

    weak_process = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )

    strong_process = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(p=0.5),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
            RandomErasing(p=1, scale=(0.02, 0.4), ratio=(1, 1), value="random"),
        ]
    )
    test_process = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(args.input_size),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )

    # Data
    labeled_loader, unlabeled_loader, testloader, classes = build_ssl_dataloaders(
        args.dataset,
        DOWNSTREAM_PATH,
        weak_process,
        weak_process,
        strong_process,
        test_process,
        batch_size=args.bs,
        shot=args.shot,
        mu=args.unlabeled_ratio,
        seed=args.seed,
        num_workers=args.workers,
    )
    softmatch_loader = FixMatchDataLoader(labeled_loader, unlabeled_loader)

    data_info = record_data(
        labeled_loader, unlabeled_loader, testloader, classes, args, output
    )
    model_for_text = model.module if isinstance(model, torch.nn.DataParallel) else model
    txt_emb_desattr, des_text = recorded_attributes(
        classes,
        model_for_text,
        ROOT / f"attributes/{args.attr_path}/{args.dataset}_des.json",
        args.num_attr,
        processor,
    )
    txt_emb_distattr, dist_text = recorded_attributes(
        classes,
        model_for_text,
        ROOT / f"attributes/{args.attr_path}/{args.dataset}_dist.json",
        args.num_attr,
        processor,
    )
    selected_attributes = {"des": des_text, "dist": dist_text}
    write_json(output / "selected_attributes.json", selected_attributes)

    print(f"Shape of attribute {txt_emb_desattr.shape}, Class Num:{len(classes)}")
    visual_reprogram = PaddingVR(224, args.input_size, model=args.pre_model).to(device)
    if args.pre_model == "clip":
        if "ViT" in args.backbone:
            if "ViT-L" in args.backbone:
                textual_dim = 768
            else:
                textual_dim = 512
        else:
            if "RN101" == args.backbone:
                textual_dim = 512
            else:
                textual_dim = 1024
    else:
        textual_dim = 768
    text_linear = TextMLP(textual_dim)
    attrvr = AttrVRModule(
        model, visual_reprogram, text_linear, txt_emb_desattr, txt_emb_distattr, args
    )

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        attrvr_multi = torch.nn.DataParallel(attrvr).to(device)
    else:
        attrvr_multi = attrvr.to(device)

    if evaluation_checkpoint is not None:
        if classes != evaluation_checkpoint["classes"]:
            raise ValueError("Checkpoint class order differs from dataset")
        if data_info["test_hash"] != evaluation_checkpoint["data_audit"]["test_hash"]:
            raise ValueError("Checkpoint test split differs from dataset")
        restore_inference(evaluation_checkpoint, attrvr)
        replay = evaluate_saved(attrvr_multi, testloader, device)
        expected = evaluation_checkpoint["test_acc"]
        if abs(replay - expected) > 1e-12:
            raise ValueError(f"Checkpoint replay differs: {replay} vs {expected}")
        write_json(
            output / "evaluation.json",
            {
                "accuracy": replay,
                "expected_accuracy": expected,
                "verified": True,
                "checkpoint": str(args.evaluate.resolve()),
                "test_size": data_info["test_size"],
            },
        )
        print("CHECKPOINT_REPLAY_OK", replay, flush=True)
        return

    def network(x):
        return attrvr(x)

    def network_multi(x):
        return attrvr_multi(x)

    opt_group = [
        {"params": visual_reprogram.parameters(), "lr": args.lr, "momentum": 0.9},
        {"params": text_linear.parameters(), "lr": args.linear_lr, "momentum": 0.9},
    ]

    # Optim & sched (warmup + cosine)
    optimizer = torch.optim.SGD(opt_group)
    t_max = args.epoch * len(softmatch_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # SoftMatch stat
    sm_stat = SoftMatchStat(
        num_classes=len(classes),
        device=device,
        margin_init=args.margin_init,
        ema_p=args.ema_p,
    )

    # Train
    # visual_reprogram.train()
    best_acc = 0.0
    best_epoch = 0
    started = time.time()
    progress_bar = tqdm(
        total=args.epoch, desc="Training SoftMatch + AttrDA", leave=True
    )

    def _sync_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for epoch in range(args.epoch):
        _sync_cuda()
        epoch_t0 = time.perf_counter()
        total_num = true_num = 0
        loss_sum = sup_sum = unsup_sum = 0.0
        avg_mask = avg_max_prob = avg_attr_w = avg_margin = 0.0

        attrvr.train()
        attrvr_multi.train()

        for labeled_batch, unlabeled_batch in softmatch_loader:
            # labeled
            labeled_x, labeled_y = labeled_batch
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)

            # unlabeled
            ulb_w, ulb_s, _ = unlabeled_batch
            ulb_w, ulb_s = ulb_w.to(device), ulb_s.to(device)

            optimizer.zero_grad()

            # --- supervised ---
            logits_lb, _, _ = network_multi(labeled_x)
            if args.pre_model == "clip":
                sup_loss = F.cross_entropy(logits_lb, labeled_y, reduction="mean")
            elif args.pre_model == "siglip" or args.pre_model == "siglip2":
                # loss = F.cross_entropy(fx, y, reduction='mean')
                y_onehot = torch.zeros_like(logits_lb).scatter_(
                    1, labeled_y.view(-1, 1), 1.0
                )
                m1 = -torch.ones_like(logits_lb) + 2 * y_onehot
                loglik = torch.nn.functional.logsigmoid(m1 * logits_lb)
                nll = -torch.sum(loglik, dim=-1)
                sup_loss = nll.mean()
            else:
                raise ValueError(f"Unknown pre-model: {args.pre_model}")

            # --- weak branch ---
            # Weak predictions are only used to build pseudo labels/weights,
            # so this branch does not need gradients.
            with torch.no_grad():
                fused_w, des_w, dist_w = network_multi(ulb_w)
                probs_ulb_w_raw = torch.softmax(fused_w, dim=-1)  # 原始弱增强预测

            # 是否使用 top-margin 作为辅助统计
            if args.top_margin:
                margin_ulb_w_raw = top2_margin(probs_ulb_w_raw)
                margin_lb = top2_margin(torch.softmax(logits_lb.detach(), dim=-1))
            else:
                margin_ulb_w_raw = None

            # 分布对齐（类别频次）
            if args.dist_align:
                # 动量统计
                sm_stat.update_prob_t(
                    torch.softmax(logits_lb.detach(), dim=-1), probs_ulb_w_raw
                )

                if args.top_margin:
                    sm_stat.update_margin_t(margin_lb, margin_ulb_w_raw)

                probs_ulb_w = sm_stat.distribution_alignment_probe(probs_ulb_w_raw)
                margin_ulb_w = sm_stat.distribution_alignment_margin(margin_ulb_w_raw)
            else:
                probs_ulb_w = probs_ulb_w_raw
                margin_ulb_w = margin_ulb_w_raw

            max_probs, pseudo_label = probs_ulb_w.max(dim=-1)

            # SoftMatch 高斯权重
            if args.use_gaussian:
                max_probs, mask = sm_stat.calculate_mask(
                    probs_ulb_w, margin_ulb_w
                )  # [B]
            else:
                mask = (max_probs >= args.confidence_thresh).float()

            if args.consistency:
                # ---- 属性一致权重 ----
                probs_des = torch.softmax(des_w, dim=-1)
                probs_dist = torch.softmax(dist_w, dim=-1)

                attr_w = build_attr_weight(
                    probs_des,
                    probs_dist,
                    probs_ulb_w,
                    use_js=args.use_js,
                    beta=args.attr_beta,
                )  # [B] in [0,1]
                final_weight = torch.clamp(mask * attr_w, 0.0, 1.0)
            else:
                attr_w = torch.ones_like(mask)
                final_weight = mask

            logits_ulb_s, _, _ = network_multi(ulb_s)
            unsup_loss, _ = consistency_loss(
                logits_ulb_s, probs_ulb_w, final_weight, use_hard_labels=args.hard_label
            )

            total_loss = sup_loss + args.lambda_u * unsup_loss
            if not torch.isfinite(total_loss):
                raise ValueError(f"Nonfinite loss at epoch {epoch + 1}")
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # stat
            total_num += labeled_y.size(0)
            true_num += torch.argmax(logits_lb, 1).eq(labeled_y).float().sum().item()
            loss_sum += total_loss.item()
            sup_sum += sup_loss.item()
            unsup_sum += unsup_loss.item()
            avg_mask += mask.mean().item()
            avg_max_prob += max_probs.mean().item()

            if margin_ulb_w is not None:
                avg_margin += margin_ulb_w.mean().item()
            else:
                avg_margin += 0.0

            avg_attr_w += attr_w.mean().item()

            # _clip_core = model
            attrvr.pre_model.logit_scale.data = torch.clamp(
                attrvr.pre_model.logit_scale.data, 0, 4.6052
            )

        _sync_cuda()
        epoch_train_time_s = time.perf_counter() - epoch_t0

        # epoch logs
        n_iter = max(1, len(softmatch_loader))
        train_acc = true_num / total_num if total_num > 0 else 0.0
        log_avg = {
            "avg_loss": loss_sum / n_iter,
            "avg_sup_loss": sup_sum / n_iter,
            "avg_unsup_loss": unsup_sum / n_iter,
            "avg_mask": avg_mask / n_iter,
            "avg_attr_w": avg_attr_w / n_iter,
            "avg_max_prob": avg_max_prob / n_iter,
            "avg_ulb_prob_t": float(sm_stat.ulb_prob_t.mean().item()),
            "avg_lb_prob_t": float(sm_stat.lb_prob_t.mean().item()),
            "prob_mu": float(sm_stat.prob_max_mu_t),
            "prob_var": float(sm_stat.prob_max_var_t),
            "margin_mu": float(sm_stat.ub_margin_mu_t),
            "margin_var": float(sm_stat.ub_margin_var_t),
            "epoch_train_time_s": epoch_train_time_s,
        }

        progress_bar.set_postfix(
            {
                "Epoch": epoch + 1,
                "Train Acc": f"{train_acc:.4f}",
                "Loss": f'{log_avg["avg_loss"]:.4f}',
                "Sup Loss": f'{log_avg["avg_sup_loss"]:.4f}',
                "Unsup Loss": f'{log_avg["avg_unsup_loss"]:.4f}',
                "Mask": f'{log_avg["avg_mask"]:.3f}',
                "AttrW": f'{log_avg["avg_attr_w"]:.3f}',
                "p_mu": f'{log_avg["prob_mu"]:.3f}',
                "p_var": f'{log_avg["prob_var"]:.3f}',
                "m_mu": f'{log_avg["margin_mu"]:.3f}',
                "m_var": f'{log_avg["margin_var"]:.3f}',
                "t_tr": f"{epoch_train_time_s:.1f}s",
            }
        )

        # ===== Eval =====
        attrvr.eval()
        attrvr_multi.eval()
        _sync_cuda()
        eval_t0 = time.perf_counter()
        test_total = test_true = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                fx, _, _ = network_multi(x)
                test_total += y.size(0)
                test_true += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = test_true / test_total if test_total > 0 else 0.0

        _sync_cuda()
        epoch_eval_time_s = time.perf_counter() - eval_t0
        epoch_time_s = epoch_train_time_s + epoch_eval_time_s
        log_avg["epoch_eval_time_s"] = epoch_eval_time_s
        log_avg["epoch_time_s"] = epoch_time_s

        progress_bar.set_postfix(
            {
                "Epoch": epoch + 1,
                "Train Acc": f"{train_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}",
                "Best Acc": f"{best_acc:.4f}",
                "Mask": f'{log_avg["avg_mask"]:.3f}',
                "AttrW": f'{log_avg["avg_attr_w"]:.3f}',
                "t_ep": f"{epoch_time_s:.1f}s",
            },
            refresh=False,
        )

        is_best = test_acc > best_acc or best_epoch == 0
        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1
        state_dict = {
            "schema_version": 1,
            "visual_prompt_dict": cpu_state(visual_reprogram.state_dict()),
            "text_adapter_dict": cpu_state(text_linear.state_dict()),
            "txt_emb_desattr": cpu_state(attrvr.txt_emb_desattr),
            "txt_emb_distattr": cpu_state(attrvr.txt_emb_distattr),
            "logit_scale": cpu_state(attrvr.pre_model.logit_scale),
            "optimizer": cpu_state(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "softmatch_state": cpu_state(
                {key: value for key, value in vars(sm_stat).items() if key != "device"}
            ),
            "rng_state": rng_state(),
            "training_args": training_args,
            "config": config,
            "backbone_reference": backbone_reference,
            "classes": classes,
            "selected_attributes": selected_attributes,
            "data_audit": data_info,
            "epoch": epoch,
            "epoch_completed": epoch + 1,
            "global_step": (epoch + 1) * len(softmatch_loader),
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "train_acc": train_acc,
            "test_acc": test_acc,
            **log_avg,
        }
        if hasattr(attrvr.pre_model, "logit_bias"):
            state_dict["logit_bias"] = cpu_state(attrvr.pre_model.logit_bias)
        if is_best:
            save_checkpoint(output / "best.pth", state_dict)
        save_checkpoint(output / "last.pth", state_dict)
        if (epoch + 1) % 50 == 0:
            save_checkpoint(output / f"checkpoint_epoch_{epoch + 1}.pth", state_dict)
        metric = {
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.time() - started,
            **log_avg,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metric, allow_nan=False) + "\n")

        visual_reprogram.train()
        progress_bar.update(1)
        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            break

    progress_bar.close()
    print("\nTraining completed!")
    print(f"Final best test accuracy: {best_acc:.4f}")
    print(f"Results saved to: {output}")
    saved_best = torch.load(output / "best.pth", map_location="cpu", weights_only=True)
    restore_inference(saved_best, attrvr)
    replay = evaluate_saved(attrvr_multi, testloader, device)
    if abs(replay - best_acc) > 1e-12:
        raise ValueError(f"Best checkpoint replay differs: {replay} vs {best_acc}")
    write_json(
        output / "summary.json",
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "best_acc": best_acc,
            "final_acc": test_acc,
            "best_epoch": best_epoch,
            "replay_best_acc": replay,
            "checkpoint_verified": True,
            "epochs_completed": epoch + 1,
            "requested_epochs": args.epoch,
            "complete": epoch + 1 == args.epoch,
            "elapsed_seconds": time.time() - started,
            "data_audit": data_info,
            "training_args": training_args,
        },
    )
    print("BEST_CHECKPOINT_REPLAY_OK", replay, flush=True)


if __name__ == "__main__":
    main()
