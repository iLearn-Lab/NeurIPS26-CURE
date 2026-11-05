"""Run independent benchmark jobs sequentially on the visible GPU."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("caltech101", "dtd", "eurosat", "fgvc", "food101", "oxford_flowers",
            "oxford_pets", "stanford_cars", "sun397", "ucf101", "resisc45", "imagenet")
SSL_METHODS = ("fixmatch", "softmatch", "cgmatch", "flexmatch", "freematch")
METHODS = ("cure", "vr", "ar", "attrvr", "ar_fixmatch", "ar_flexmatch",
           "ar_attrfixmatch", *SSL_METHODS, "mixmatch", "refixmatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=["cure"])
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 2025])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/benchmark")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading data or starting training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    env = os.environ.copy()
    if args.data_root:
        env["CURE_DATA_ROOT"] = str(args.data_root.expanduser().resolve())
    for method in args.methods:
        for dataset in args.datasets:
            for seed in args.seeds:
                if method in SSL_METHODS:
                    entry, extra = "fs_ssl", ["--method", method]
                else:
                    entry = {"vr": "fs_vp", "mixmatch": "fs_attrvr_mixmatch",
                             "refixmatch": "fs_attrvr_refixmatch"}.get(method, f"fs_{method}")
                    extra = []
                if method in ("vr", "ar", "attrvr"):
                    extra += ["--pre_model", "clip", "--backbone", "ViT-B/16", "--bs", "128"]
                else:
                    extra += ["--bs", "16"]
                if method in ("attrvr", "ar_attrfixmatch", "mixmatch", "refixmatch"):
                    extra += ["--num_attr", "2", "--k", "2", "--alpha", "0.5"]
                command = [sys.executable, "-u", str(ROOT / f"experiments/{entry}.py"),
                           "--dataset", dataset, "--seed", str(seed), "--shot", "2",
                           "--epoch", str(args.epochs), "--output",
                           str(args.output_root.resolve() / method / dataset / f"seed_{seed}"), *extra]
                if args.backbone_weights:
                    command += ["--backbone-weights", str(args.backbone_weights.expanduser().resolve())]
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
