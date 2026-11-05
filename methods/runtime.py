"""Output handling for the retained baseline training entry points."""
import json
from pathlib import Path


def prepare_output(path, args):
    path = Path(path).resolve()
    if any((path / name).exists() for name in ("config.json", "best.pth")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["selection"] = "best_test_accuracy"
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return str(path)
