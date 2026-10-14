"""Write a protocol report package. Does not train or download data."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments import PROTOCOL_IDS
from ml.experiments.render import blocked_document, render_report

DATASET_SIZE_BLOCK = (
    "Retraining on normal-only subsets is blocked. The split loader requires the "
    "full normal inventory and the recorded validation partition, so a fit subset "
    "cannot be persisted without a new manifest contract. Any listed ids are a "
    "draw plan, not measured metrics."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown/CSV/JSON/PNG report package.")
    parser.add_argument("--protocol", required=True, choices=PROTOCOL_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--train-ids-file", type=Path)
    args = parser.parse_args()
    if args.results is not None:
        document = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        train_ids = None
        if args.train_ids_file is not None:
            loaded = json.loads(args.train_ids_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("Train ids file must be a JSON list.")
            train_ids = loaded
        reason = (
            DATASET_SIZE_BLOCK
            if args.protocol == "exp-c-dataset-size"
            else "No completed measurement file was provided for this protocol."
        )
        document = blocked_document(args.protocol, reason, train_ids=train_ids)
    path = render_report(document, args.output, protocol_id=args.protocol)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
