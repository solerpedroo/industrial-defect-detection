"""Repository paths shared by the API and the ML tree."""

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the monorepo root and make ``ml`` importable."""
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def dataset_root() -> Path:
    return repo_root() / "data" / "mvtec-ad"


def config_dir() -> Path:
    return repo_root() / "ml" / "configs"
