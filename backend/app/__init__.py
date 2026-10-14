"""Industrial Defect Detection API package."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo = str(_REPO_ROOT)
if _repo not in sys.path:
    sys.path.insert(0, _repo)
