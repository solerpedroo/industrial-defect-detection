"""Freeze normal validation thresholds and report experiment A on local test images."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation import evaluate, load_evaluation_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    config = load_evaluation_config()
    catalog = yaml.safe_load((REPO_ROOT / "ml/configs/categories.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", choices=config.categories)
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / catalog["dataset"]["storage_root"]
    )
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-kind", choices=("synthetic", "mvtec_ad"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protocol", type=Path, help="Explicit experiment A YAML override.")
    for model_id in config.models:
        parser.add_argument(
            f"--{model_id}",
            action="append",
            required=True,
            metavar="CATEGORY=TRUSTED_CHECKPOINT",
            help="Repeat for each category.",
        )
    args = parser.parse_args(argv)
    try:
        if args.protocol is not None:
            config = load_evaluation_config(args.protocol)
        if args.category is not None:
            config = replace(config, categories=tuple(args.category))
        artifacts = {category: {} for category in config.categories}
        for model_id in config.models:
            for entry in getattr(args, model_id):
                category, separator, path = entry.partition("=")
                if (
                    not separator
                    or not path
                    or category not in artifacts
                    or model_id in artifacts[category]
                ):
                    raise ValueError(
                        "Supply exactly one CATEGORY=TRUSTED_CHECKPOINT per model/category."
                    )
                artifacts[category][model_id] = Path(path)
        result = evaluate(
            args.dataset_root,
            artifacts,
            args.output_dir,
            data_kind=args.data_kind,
            manifest_dir=args.manifest_dir,
            config=config,
            device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"results": str(result), "data_kind": args.data_kind}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
