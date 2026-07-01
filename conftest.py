"""Make the repo root importable so tests can `from src... import ...` from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
