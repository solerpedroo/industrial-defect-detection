"""Experiment protocol helpers.

Wave 8 renders Markdown, CSV, JSON and PNG packages and scores experiment B
at thresholds frozen by experiment A. Experiment C records subset membership.
Measured retraining of those subsets stays blocked until the manifest contract
can persist a smaller fit set without dropping checksum coverage.
"""

PROTOCOL_IDS = (
    "exp-a-autoencoder-vs-patchcore",
    "exp-b-robustness",
    "exp-c-dataset-size",
)
