"""Ship gates: git and the API image stay free of datasets and checkpoints."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pt", ".ckpt", ".pth"}


def _blocked(path: str) -> bool:
    if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    if path.startswith(("checkpoints/", "runs/", "artifacts/")):
        return True
    if path.startswith("data/") and path != "data/README.md":
        return True
    if path.startswith("ml/data/") and path != "ml/data/README.md":
        return True
    return path.startswith("sample_outputs/") and path != "sample_outputs/README.md"


def test_git_does_not_track_images_or_checkpoints() -> None:
    completed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = [item for item in completed.stdout.decode().split("\0") if item]
    blocked = [path for path in tracked if _blocked(path)]
    assert blocked == []


def test_api_image_does_not_copy_dataset_or_checkpoints() -> None:
    dockerfile = (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8")
    copies = "\n".join(line for line in dockerfile.splitlines() if line.strip().startswith("COPY"))
    for token in ("data/", "checkpoints", "runs/", "sample_outputs", ".pt"):
        assert token not in copies
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    for token in ("data/", "checkpoints", "sample_outputs", ".pt"):
        assert token not in compose
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    for line in ("data/*", "!data/README.md", "checkpoints/", "*.pt", "sample_outputs/*"):
        assert line in ignore.splitlines()


def test_readme_names_the_reproduction_path() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for snippet in (
        "scripts/train_autoencoder.py",
        "scripts/build_patchcore.py",
        "scripts/evaluate_experiment.py",
        "scripts/render_report.py",
        "exp-a-autoencoder-vs-patchcore",
        "exp-b-robustness",
        "exp-c-dataset-size",
        "INFERENCE_MANIFEST",
        "not physical severity",
    ):
        assert snippet in readme
