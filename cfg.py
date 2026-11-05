"""Dataset location shared by the public training entry points."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOWNSTREAM_PATH = str(Path(os.environ.get("CURE_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve())
