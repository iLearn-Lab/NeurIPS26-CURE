"""Check the runtime, bundled metadata, and optionally prepared images."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Also verify that this dataset's released seed-0 images exist")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    import torch
    import torchvision
    import clip
    from methods.vp import PaddingVR
    from methods.ssl_core import METHODS, SSLController

    for package in ("torch", "torchvision", "numpy", "scipy", "Pillow", "clip"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    index = json.loads((ROOT / "splits/index.json").read_text())
    print(index)
    for key, info in index.items():
        path = ROOT / info["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Changed split file: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            split = json.load(stream)
        counts = Counter(label for _, label in split["train"])
        if set(counts) != set(range(info["num_classes"])) or set(counts.values()) != {2}:
            raise ValueError(f"Invalid 2-shot counts: {key}")
        groups = [{row[0] for row in split[name]} for name in ("train", "unlabeled", "test")]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError(f"Overlapping data partitions: {key}")
        for kind in ("des", "dist"):
            attributes = json.loads((ROOT / f"attributes/gpt3/{split['dataset']}_{kind}.json").read_text())
            for classname in split["classes"]:
                if not any(len(text) >= 20 for text in attributes[classname]):
                    raise ValueError(f"Missing {kind} attributes for {classname}")
    print(f"Bundled splits and attribute coverage: OK ({len(index)} runs)")

    prompt = PaddingVR(224, 192)
    image = prompt(torch.zeros(1, 3, 192, 192))
    image.sum().backward()
    if image.shape != (1, 3, 224, 224) or not torch.isfinite(prompt.program.grad).all():
        raise ValueError("Visual prompt gradient check failed")
    for method in METHODS:
        controller = SSLController(method, 3, 8, torch.device("cpu"), 10, cg_queue_size=2)
        labeled = torch.randn(2, 3, requires_grad=True)
        weak = torch.randn(4, 3)
        strong = torch.randn(4, 3, requires_grad=True)
        loss, _ = controller.loss(1, labeled, torch.tensor([0, 1]), weak, strong, torch.arange(4), 1.0)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(labeled.grad).all() or not torch.isfinite(strong.grad).all():
            raise ValueError(f"Nonfinite SSL loss or gradient: {method}")
    print("Visual prompt and five SSL loss/gradient checks: OK")
    if args.dataset:
        from cfg import DOWNSTREAM_PATH
        from datasets import build_dataset
        dataset = build_dataset(args.dataset, args.data_root or DOWNSTREAM_PATH, shot=2, seed=0)
        print(f"Dataset files: OK (labeled={len(dataset.train_x)}, unlabeled={len(dataset.train_u)}, test={len(dataset.test)})")


if __name__ == "__main__":
    main()
