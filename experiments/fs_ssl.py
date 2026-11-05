"""Train AttrVR with one pinned, official SSL objective.

This entry point deliberately owns the data wrappers and training loop.  It keeps
the AttrVR model/protocol common across methods and gives FlexMatch/CGMatch a
stable unlabeled-sample index, which the historical wrappers did not expose.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clip
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Lambda,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from cfg import DOWNSTREAM_PATH
from datasets import build_dataset, dataset_list
from methods.vp import PaddingVR
from methods.ssl_core import METHODS, SSLController
from tools import convert_models_to_fp32, getLogitsCLIP, set_seed


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest_samples(samples) -> str:
    digest = hashlib.sha256()
    for item in sorted(samples, key=lambda x: (x.impath, x.label)):
        digest.update(f"{Path(item.impath).resolve()}\t{item.label}\n".encode())
    return digest.hexdigest()


class LabeledDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with Image.open(item.impath) as image:
            image = image.convert("RGB")
            return self.transform(image), item.label


class UnlabeledDataset(Dataset):
    def __init__(self, samples, weak_transform, strong_transform):
        self.samples = list(samples)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with Image.open(item.impath) as image:
            image = image.convert("RGB")
            weak = self.weak_transform(image)
            strong = self.strong_transform(image)
        return index, weak, strong, item.label


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loaders(args, output: Path):
    dataset = build_dataset(args.dataset, args.data_root, args.shot, args.seed)
    labeled = list(dataset.train_x)
    unlabeled = list(dataset.train_u)
    test = list(dataset.test)
    if not unlabeled:
        raise ValueError(f"{args.dataset} exposes no train_u pool")

    counts = Counter(item.label for item in labeled)
    expected_labels = set(range(len(dataset.classnames)))
    if set(counts) != expected_labels or any(value != args.shot for value in counts.values()):
        raise ValueError(f"Few-shot split is not exactly {args.shot}/class: {counts}")
    sets = {
        "labeled": {str(Path(item.impath).resolve()) for item in labeled},
        "unlabeled": {str(Path(item.impath).resolve()) for item in unlabeled},
        "test": {str(Path(item.impath).resolve()) for item in test},
    }
    overlaps = {
        "labeled_unlabeled": len(sets["labeled"] & sets["unlabeled"]),
        "labeled_test": len(sets["labeled"] & sets["test"]),
        "unlabeled_test": len(sets["unlabeled"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Data partitions overlap: {overlaps}")
    for sample in labeled + unlabeled + test:
        if not Path(sample.impath).is_file():
            raise FileNotFoundError(sample.impath)

    rgb = Lambda(lambda image: image.convert("RGB"))
    weak = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            rgb,
            ToTensor(),
        ]
    )
    strong = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=10),
            rgb,
            ToTensor(),
        ]
    )
    test_transform = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(args.input_size),
            rgb,
            ToTensor(),
        ]
    )
    labeled_dataset = LabeledDataset(labeled, weak)
    unlabeled_dataset = UnlabeledDataset(unlabeled, weak, strong)
    test_dataset = LabeledDataset(test, test_transform)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
    )
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.bs,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.bs * args.unlabeled_ratio,
        sampler=RandomSampler(
            unlabeled_dataset,
            replacement=True,
            num_samples=args.bs * args.unlabeled_ratio * len(labeled_loader),
            generator=generator,
        ),
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_bs,
        shuffle=False,
        **loader_options,
    )
    audit = {
        "dataset": args.dataset,
        "seed": args.seed,
        "shot": args.shot,
        "num_classes": len(dataset.classnames),
        "labeled_count": len(labeled),
        "unlabeled_count": len(unlabeled),
        "test_count": len(test),
        "labeled_per_class": dict(sorted(counts.items())),
        "labeled_hash": digest_samples(labeled),
        "unlabeled_hash": digest_samples(unlabeled),
        "test_hash": digest_samples(test),
        "overlap": overlaps,
    }
    write_json(output / "data_audit.json", audit)
    return labeled_loader, unlabeled_loader, test_loader, dataset.classnames, audit


@torch.no_grad()
def build_attribute_embeddings(args, classnames, model, output: Path):
    selected = {"des": {}, "dist": {}}
    embeddings = {}
    rng = random.Random(args.seed)
    for kind in ("des", "dist"):
        attributes = json.loads(
            (ROOT / f"attributes/gpt3/{args.dataset}_{kind}.json").read_text()
        )
        class_embeddings = []
        for classname in classnames:
            candidates = [text for text in attributes[classname] if len(text) >= 20]
            if not candidates:
                raise ValueError(f"No valid {kind} attribute for {classname}")
            texts = (
                rng.sample(candidates, args.num_attr)
                if len(candidates) >= args.num_attr
                else rng.choices(candidates, k=args.num_attr)
            )
            selected[kind][classname] = texts
            class_embeddings.append(model.encode_text(clip.tokenize(texts).to(args.device)))
        embeddings[kind] = torch.stack(class_embeddings, dim=1).to(args.device)
    write_json(output / "selected_attributes.json", selected)
    return embeddings["des"], embeddings["dist"]


def make_network(args, clip_model, prompt, des_embeddings, dist_embeddings):
    def network(images):
        features = clip_model.encode_image(prompt(images))
        features = features / features.norm(dim=-1, keepdim=True)
        scale = clip_model.logit_scale.exp()
        des = getLogitsCLIP(args.num_attr, scale, features, des_embeddings, k=args.k)
        dist = getLogitsCLIP(args.num_attr, scale, features, dist_embeddings, k=args.k)
        return args.alpha * des + (1.0 - args.alpha) * dist

    return network


@torch.no_grad()
def evaluate(network, loader, device) -> float:
    correct = total = 0
    for images, labels in loader:
        logits = network(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def scheduler_factor(step: int, total_steps: int, warmup_steps: int, min_factor: float):
    if step < warmup_steps:
        # Match the historical SequentialLR/LinearLR schedule: the optimizer is
        # initialized at 1e-6 * base_lr, then reaches base_lr after five epochs.
        return 1e-6 + (1.0 - 1e-6) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path, args, prompt, controller, epoch, best_acc, best_epoch):
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "prompt": prompt.state_dict(),
            "ssl_state": controller.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "training_args": serialized_args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=dataset_list, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(DOWNSTREAM_PATH))
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-attr", "--num_attr", dest="num_attr", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=40.0)
    parser.add_argument("--input-size", "--input_size", dest="input_size", type=int, default=192)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lambda-u", "--lambda_u", dest="lambda_u", type=float, default=1.0)
    parser.add_argument("--unlabeled-ratio", "--unlabeled_ratio", dest="unlabeled_ratio", type=int, default=7)
    parser.add_argument("--p-cutoff", "--p_cutoff", dest="p_cutoff", type=float, default=0.95)
    parser.add_argument("--ema-p", "--ema_p", dest="ema_p", type=float, default=0.999)
    parser.add_argument("--n-sigma", "--n_sigma", dest="n_sigma", type=float, default=2.0)
    parser.add_argument("--entropy-ratio", "--entropy_ratio", dest="entropy_ratio", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--test-bs", "--test_bs", dest="test_bs", type=int, default=128)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if any((args.output / name).exists() for name in ("config.json", "metrics.jsonl", "best.pth")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    if not (0 < args.k <= args.num_attr) or args.epoch < 1 or args.shot < 1:
        parser.error("Require 0 < k <= num_attr, epoch >= 1, shot >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    set_seed(args.seed)
    device = torch.device(args.device)

    labeled_loader, unlabeled_loader, test_loader, classes, audit = build_loaders(
        args, args.output
    )
    clip_model, _ = clip.load(str(args.backbone_weights) if args.backbone_weights else "ViT-B/16", device=device, jit=False)
    convert_models_to_fp32(clip_model)
    clip_model.eval().requires_grad_(False)
    des, dist = build_attribute_embeddings(args, classes, clip_model, args.output)
    prompt = PaddingVR(224, args.input_size).to(device)
    network = make_network(args, clip_model, prompt, des, dist)

    steps_per_epoch = len(labeled_loader)
    total_steps = args.epoch * steps_per_epoch
    expected_visits = total_steps * args.bs * args.unlabeled_ratio / len(unlabeled_loader.dataset)
    cg_queue = max(2, min(1000, round(expected_visits * 0.106)))
    cg_warmup = max(1, round(total_steps / 512))
    controller = SSLController(
        args.method,
        len(classes),
        len(unlabeled_loader.dataset),
        device,
        total_steps,
        p_cutoff=args.p_cutoff,
        ema_p=args.ema_p,
        n_sigma=args.n_sigma,
        entropy_ratio=args.entropy_ratio,
        cg_queue_size=cg_queue,
        cg_warmup_steps=cg_warmup,
    )
    optimizer = torch.optim.SGD(
        prompt.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    warmup_steps = min(total_steps, 5 * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(step, total_steps, warmup_steps, 1e-4 / args.lr),
    )
    recipe = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("device", "output")
    }
    recipe.update(
        {
            "method": args.method,
            "cg_queue_size_adapted": cg_queue,
            "cg_warmup_steps_adapted": cg_warmup,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "strong_augmentation": "torchvision RandAugment(n=2,m=10)",
        }
    )
    write_json(args.output / "config.json", recipe)

    best_acc = -1.0
    best_epoch = 0
    final_acc = 0.0
    global_step = 0
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(1, args.epoch + 1):
        started = time.time()
        prompt.train()
        totals = Counter()
        unlabeled_iterator = iter(unlabeled_loader)
        for images_l, labels_l in labeled_loader:
            indices, images_w, images_s, _ = next(unlabeled_iterator)
            images_l = images_l.to(device, non_blocking=True)
            labels_l = labels_l.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_l = network(images_l)
            if args.method == "cgmatch" and global_step < cg_warmup:
                logits_w = logits_s = None
            else:
                with torch.no_grad():
                    logits_w = network(images_w.to(device, non_blocking=True))
                logits_s = network(images_s.to(device, non_blocking=True))
            loss, details = controller.loss(
                global_step,
                logits_l,
                labels_l,
                logits_w,
                logits_s,
                indices,
                args.lambda_u,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["train_correct"] += logits_l.argmax(1).eq(labels_l).sum().item()
            totals["train_total"] += labels_l.numel()
            for key, value in details.items():
                totals[key] += value
            global_step += 1

        prompt.eval()
        final_acc = evaluate(network, test_loader, device)
        if final_acc > best_acc:
            best_acc, best_epoch = final_acc, epoch
            save_checkpoint(
                args.output / "best.pth",
                args,
                prompt,
                controller,
                epoch,
                best_acc,
                best_epoch,
            )
        row = {
            "epoch": epoch,
            "test_acc": final_acc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "train_acc": totals["train_correct"] / totals["train_total"],
            "loss": totals["loss"] / steps_per_epoch,
            "supervised_loss": totals["supervised_loss"] / steps_per_epoch,
            "unsupervised_loss": totals["unsupervised_loss"] / steps_per_epoch,
            "mask": totals["mask"] / steps_per_epoch,
            "ambiguous_mask": totals["ambiguous_mask"] / steps_per_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.time() - started,
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    save_checkpoint(
        args.output / "last.pth",
        args,
        prompt,
        controller,
        args.epoch,
        best_acc,
        best_epoch,
    )
    checkpoint = torch.load(args.output / "best.pth", map_location=device)
    prompt.load_state_dict(checkpoint["prompt"])
    prompt.eval()
    replay = evaluate(network, test_loader, device)
    summary = {
        "complete": True,
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs_completed": args.epoch,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "replay_best_acc": replay,
        "best_epoch": best_epoch,
        "checkpoint_verified": abs(best_acc - replay) < 1e-12,
        "training_args": recipe,
        "data_audit": audit,
    }
    write_json(args.output / "summary.json", summary)
    if not summary["checkpoint_verified"]:
        raise RuntimeError("Best checkpoint replay does not reproduce best accuracy")


if __name__ == "__main__":
    main()
