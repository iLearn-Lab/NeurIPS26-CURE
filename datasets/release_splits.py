"""Portable membership of the released 2-shot benchmark, in recorded order."""
import gzip
import json
from pathlib import Path

from .utils import DatasetBase, Datum


def load_release_split(dataset, root, shot, seed):
    path = Path(__file__).resolve().parents[1] / "splits" / dataset / f"shot_{shot}-seed_{seed}.json.gz"
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    root = Path(root).expanduser().resolve()
    classes = record["classes"]

    def samples(key):
        result = []
        for relative, label in record[key]:
            image = root / relative
            # The historical Pets data used a directory alias. All supported
            # aliases point to the same image identities.
            if relative.startswith("oxford-iiit-pet/") and not image.is_file():
                remainder = relative.split("/", 1)[1]
                for alias in ("oxford_iiit_pet", "oxford_pets"):
                    candidate = root / alias / remainder
                    if candidate.is_file():
                        image = candidate
                        break
            if not image.is_file():
                raise FileNotFoundError(f"Missing benchmark image: {image}. See README.md for the data layout.")
            result.append(Datum(impath=str(image), label=label, classname=classes[label]))
        return result

    train, unlabeled, test = samples("train"), samples("unlabeled"), samples("test")
    result = DatasetBase(train_x=train, train_u=unlabeled, val=[], test=test)
    if result.classnames != classes:
        raise ValueError("Release split class order is inconsistent")
    print(f"Using released {shot}-shot split: {dataset}, seed={seed}")
    return result
