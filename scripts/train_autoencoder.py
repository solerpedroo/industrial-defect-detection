"""Train on normal manifest IDs offline; never create a split or tune on test."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.splits import DEFAULT_MANIFEST_DIR  # noqa: E402
from ml.training.autoencoder import train_autoencoder  # noqa: E402
from ml.training.config import load_autoencoder_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    catalog = yaml.safe_load((REPO_ROOT / "ml/configs/categories.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", required=True, choices=[entry["id"] for entry in catalog["categories"]]
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / catalog["dataset"]["storage_root"]
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New run directory, never overwritten."
    )
    parser.add_argument("--device", default="cpu", help="cpu or an available cuda[:index] device.")
    for name, kind in (
        ("epochs", int),
        ("batch-size", int),
        ("image-size", int),
        ("seed", int),
        ("patience", int),
        ("learning-rate", float),
    ):
        parser.add_argument(f"--{name}", type=kind, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_autoencoder_config(args.category)
        if args.image_size is not None:
            config = replace(
                config, preprocess=replace(config.preprocess, image_size=args.image_size)
            )
        overrides = {
            field: getattr(args, argument)
            for field, argument in (
                ("max_epochs", "epochs"),
                ("batch_size", "batch_size"),
                ("seed", "seed"),
                ("early_stopping_patience", "patience"),
                ("learning_rate", "learning_rate"),
            )
            if getattr(args, argument) is not None
        }
        config = replace(config, **overrides)
        result = train_autoencoder(
            args.dataset_root,
            args.category,
            args.output_dir,
            manifest_dir=args.manifest_dir,
            config=config,
            device=args.device,
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "history": str(result.history_path),
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "epochs_completed": result.epochs_completed,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
