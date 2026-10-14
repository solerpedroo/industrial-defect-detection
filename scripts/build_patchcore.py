"""Build a frozen normal PatchCore memory from an existing validated manifest."""

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
from ml.models.patchcore import build_patchcore_memory, load_patchcore_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    catalog = yaml.safe_load((REPO_ROOT / "ml/configs/categories.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", required=True, choices=[item["id"] for item in catalog["categories"]]
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / catalog["dataset"]["storage_root"]
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Explicit random-weight regression mode; not a pretrained run.",
    )
    for name, kind in (
        ("image-size", int),
        ("batch-size", int),
        ("seed", int),
        ("max-patches", int),
        ("num-neighbors", int),
        ("coreset-ratio", float),
    ):
        parser.add_argument(f"--{name}", type=kind, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_patchcore_config(args.category)
        if args.image_size is not None:
            config = replace(
                config, preprocess=replace(config.preprocess, image_size=args.image_size)
            )
        values = {
            key: getattr(args, argument)
            for key, argument in (
                ("batch_size", "batch_size"),
                ("seed", "seed"),
                ("max_patches", "max_patches"),
                ("num_neighbors", "num_neighbors"),
                ("coreset_sampling_ratio", "coreset_ratio"),
            )
            if getattr(args, argument) is not None
        }
        if args.no_pretrained:
            values["pre_trained"] = False
        config = replace(config, **values)
        result = build_patchcore_memory(
            args.dataset_root,
            args.category,
            args.output_dir,
            manifest_dir=args.manifest_dir,
            config=config,
            device=args.device,
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Memory build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "metadata": str(result.metadata_path),
                "patch_count": result.patch_count,
                "memory_count": result.memory_count,
                "pre_trained": config.pre_trained,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
