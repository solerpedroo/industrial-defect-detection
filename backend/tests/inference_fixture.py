"""Synthetic real-model serving fixture; never a product training HTTP path."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def prepare_serving(root: Path):
    import torch
    from PIL import Image

    from ml.datasets import create_split_manifest
    from ml.evaluation import evaluate, load_evaluation_config
    from ml.models.patchcore import build_patchcore_memory, load_patchcore_config
    from ml.training.autoencoder import train_autoencoder
    from ml.training.config import load_autoencoder_config

    torch.set_num_threads(1)
    dataset, manifests = root / "dataset", root / "splits"
    for partition, values in (
        ("train/good", range(40, 46)),
        ("test/good", [100]),
        ("test/broken", [180]),
    ):
        directory = dataset / "bottle" / partition
        directory.mkdir(parents=True)
        for index, value in enumerate(values):
            Image.new("RGB", (32, 32), (value, value + 1, value + 2)).save(
                directory / f"{index:03}.png"
            )
    masks = dataset / "bottle/ground_truth/broken"
    masks.mkdir(parents=True)
    Image.new("L", (32, 32), 255).save(masks / "000_mask.png")
    split = create_split_manifest(
        dataset, "bottle", seed=42, validation_fraction=0.33, manifest_dir=manifests
    )
    ae = load_autoencoder_config("bottle")
    ae = replace(
        ae,
        preprocess=replace(ae.preprocess, image_size=32),
        channels=(2, 4, 8),
        max_epochs=1,
        batch_size=2,
    )
    ae_run = train_autoencoder(dataset, "bottle", root / "ae", manifest_dir=manifests, config=ae)
    pc = load_patchcore_config("bottle")
    pc = replace(
        pc,
        preprocess=replace(pc.preprocess, image_size=32),
        pre_trained=False,
        coreset_sampling_ratio=0.5,
        batch_size=2,
    )
    pc_run = build_patchcore_memory(
        dataset, "bottle", root / "pc", manifest_dir=manifests, config=pc
    )
    evaluate(
        dataset,
        {"bottle": {"autoencoder": ae_run.checkpoint_path, "patchcore": pc_run.checkpoint_path}},
        root / "evaluation",
        manifest_dir=manifests,
        data_kind="synthetic",
        config=replace(
            load_evaluation_config(), categories=("bottle",), image_size=32, warmup_images=1
        ),
    )
    manifest = root / "serving.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trusted_root": str(root),
                "device": "cpu",
                "entries": [
                    {
                        "category_id": "bottle",
                        "model_id": model,
                        "checkpoint": path,
                        "calibration": "evaluation/calibration.json",
                    }
                    for model, path in (
                        ("autoencoder", "ae/best.pt"),
                        ("patchcore", "pc/memory.pt"),
                    )
                ],
            }
        )
    )
    return SimpleNamespace(
        root=root,
        dataset=dataset,
        split=split,
        manifest=manifest,
        calibration=root / "evaluation/calibration.json",
    )


if __name__ == "__main__":
    import argparse
    import os
    import tempfile

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="industrial-inference-e2e-") as directory:
        prepared = prepare_serving(Path(directory))
        os.environ["INFERENCE_MANIFEST"] = str(prepared.manifest)
        os.environ["INFERENCE_ALLOW_SYNTHETIC"] = "1"
        os.environ["ALLOWED_ORIGINS"] = os.environ.get(
            "PLAYWRIGHT_WEB_URL", "http://localhost:3000"
        )
        from app.main import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
