import sys
from pathlib import Path

# run_detect / detector / loaders are top-level modules in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
