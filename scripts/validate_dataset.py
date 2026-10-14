"""Audit a local MVTec copy; optionally write a normal-only holdout manifest."""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets import create_split_manifest, inspect_category, summarize_category  # noqa: E402
from ml.datasets.splits import DEFAULT_MANIFEST_DIR, manifest_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    config_root = REPO_ROOT / "ml" / "configs"
    catalog = yaml.safe_load((config_root / "categories.yaml").read_text(encoding="utf-8"))
    defaults = yaml.safe_load((config_root / "training_defaults.yaml").read_text(encoding="utf-8"))
    category_ids = [item["id"] for item in catalog["categories"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / catalog["dataset"]["storage_root"]
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=category_ids,
        help="Repeat to select categories; defaults to all catalog categories.",
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Required for --create-split; no hidden split default.",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing manifests after a successful audit.",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    args = parser.parse_args(argv)
    if args.create_split and args.validation_fraction is None:
        parser.error("--create-split requires --validation-fraction.")
    if not args.create_split and (args.overwrite or args.validation_fraction is not None):
        parser.error("--overwrite and --validation-fraction require --create-split.")
    if args.create_split and (
        not math.isfinite(args.validation_fraction)
        or not 0 < args.validation_fraction < 1
        or not 0 <= args.seed < 2**32
    ):
        parser.error("Use a finite fraction in (0, 1) and a seed in [0, 2**32).")
    selected = list(dict.fromkeys(args.category or category_ids))
    reports = [inspect_category(args.dataset_root, category_id) for category_id in selected]
    errors: list[str] = []
    if args.create_split and all(report.valid for report in reports):
        for report in reports:
            count = len(report.train_checksums)
            if not 0 < math.ceil(count * args.validation_fraction) < count:
                errors.append(
                    f"{report.category_id}: split needs nonempty fit and validation sets."
                )
        existing = [
            category_id
            for category_id in selected
            if manifest_path(category_id, args.manifest_dir).exists()
        ]
        if existing and not args.overwrite:
            errors.append(f"Existing manifests for {', '.join(existing)}; use --overwrite.")
        if not errors:
            for category_id in selected:
                try:
                    create_split_manifest(
                        args.dataset_root,
                        category_id,
                        seed=args.seed,
                        validation_fraction=args.validation_fraction,
                        manifest_dir=args.manifest_dir,
                        overwrite=args.overwrite,
                    )
                except (ValueError, OSError) as error:
                    errors.append(str(error))
                    break
    categories = [
        {
            "id": report.category_id,
            **summarize_category(
                args.dataset_root, report.category_id, manifest_dir=args.manifest_dir
            ),
            "integrity": report.to_dict(),
        }
        for report in reports
    ]
    for category in categories:
        if category["validation_split"] == "invalid":
            errors.append(f"{category['id']}: invalid or stale split manifest.")
    payload = {"dataset_id": catalog["dataset"]["id"], "categories": categories, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for category in categories:
            print(
                f"{category['id']}: {category['availability']} "
                f"train={category['train_normal']} validation={category['validation_normal']} "
                f"test_normal={category['test_normal']} "
                f"test_anomalous={category['test_anomalous']} "
                f"masks={category['mask_count']} split={category['validation_split']}"
            )
            for issue in category["integrity"]["issues"]:
                print(f"  {issue['code']}: {issue['path']} — {issue['detail']}")
            for group in category["integrity"]["duplicates"]:
                print(f"  duplicate_content: {', '.join(group)}")
        for error in errors:
            print(f"Error: {error}")
    return 1 if errors or any(not report.valid for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
