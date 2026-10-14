# Architecture decisions

This log is the durable decision record for people and agents. Add an entry
before changing a cross-cutting convention. Accepted decisions supersede informal
chat context; implementation details belong in task cards.

## ADR-001 — Agentic foundation precedes model training

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Establish agents, skills, rules, task cards, handoffs, and a
  workboard before training code. Wave 0 may scaffold the API and the dashboard
  shell. It does not train a model.
- **Why:** The project spans dataset leakage rules, PyTorch, an external
  anomaly library, and a web contract. A coordination protocol limits drift.
- **Consequence:** Product training starts from a Wave 3 or Wave 4 task card.

## ADR-002 — Catalog-led model and category contract

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** `ml/configs/categories.yaml`, `ml/configs/models.yaml`, and
  `ml/configs/training_defaults.yaml` are the canonical ids, English labels,
  and starting hyperparameters.
- **Consequence:** API schemas and UI controls consume those catalogs. A second
  hardcoded list of categories or learning rates is a defect.

## ADR-003 — Normal-only training and sealed test protocol

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Models fit on normal images only. Anomalous images and masks are
  evaluation data. Thresholds are chosen on a validation split carved from
  normal training images, or from a documented target false-positive rate on
  that split. The official MVTec test split is used once for reporting.
- **Why:** Tuning on the test split, or training on defects, makes the
  comparison unusable for the research question.
- **Consequence:** A task card that reads test masks to pick a threshold is
  rejected. Validation manifests record the image ids they hold out.

## ADR-004 — Canonical in-memory image and transport convention

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** The API canonicalizes decoded uploads to a contiguous
  three-channel RGB `numpy.ndarray` of `uint8` values in `[0, 255]`. Alpha is
  discarded at decode. Resize and channel normalization happen later inside
  `ml/preprocessing/` and are recorded in the experiment config. Image-bearing
  routes use `multipart/form-data`.
- **Why:** RGB matches the browser, an integer boundary keeps metadata
  unambiguous, and multipart avoids base64 request overhead.
- **Consequence:** Inference loaders convert to tensors at the model boundary
  and convert maps back to PNG only in the HTTP adapter.

## ADR-005 — Upload safety envelope

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** The API accepts byte-verified PNG and JPEG uploads up to 10 MiB
  and 4096 × 4096 pixels. Upload bytes exist only for the request lifetime.
  Routes never execute uploaded content and never train in-process.
- **Why:** These limits constrain malformed input without a database.
- **Consequence:** Invalid uploads return `400`, `413`, or `415`. A `planned`
  model returns `501` after validation succeeds.

## ADR-006 — Anomaly score is not physical severity

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Model scores, heatmap intensity, and mask area are algorithm
  outputs. They are not millimetres, defect grades, or financial loss. Bounding
  boxes, when drawn, are contours of the thresholded anomaly mask.
- **Consequence:** UI copy and reports use "anomaly score" and "affected area".
  They do not label a score as severity in engineering units.

## ADR-007 — Dataset files stay out of Git

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** MVTec AD is CC BY-NC-SA 4.0. Images, masks, checkpoints, and
  feature memories are local or artifact-store files. The repository documents
  the expected directory layout and the license.
- **Consequence:** Tests use synthetic arrays. A pull request that adds dataset
  images is rejected.

## ADR-008 — Audited, deterministic normal-only split manifests

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Wave 1 stores versioned JSON split manifests under
  `ml/data/splits/<category>.json`. IDs are dataset-root-relative POSIX paths.
  The seed and validation fraction select only sorted `train/good` IDs using
  SHA-256 ranking of seed and ID; validation size is `ceil(n * fraction)`.
  Both partitions must be nonempty. Every normal training image has a raw-file
  SHA-256 checksum. No absolute dataset path, image bytes, or test labels enter
  the manifest. Creation requires a successful integrity audit first.
- **Integrity:** Decode images fully; check mask pairing, size and binary values.
  Compare RGB pixel hashes (including spatial dimensions) to catch images copied
  between official train/test or repeated within train even under a different
  encoding. Repeated test images do not influence the normal split. Audit reads
  of test pixels/masks are exclusively integrity checks, never threshold tuning.
  Symlinks and unexpected tree entries are rejected. Source files are never moved
  or deleted. Atomic writes refuse an existing manifest unless explicitly replaced.
- **API consequence:** Existing dataset summaries add `validation_normal` and
  allow `validation_split` values `not_created`, `created`, and `invalid`.
  With a valid manifest, `train_normal` counts only retained fit images. Otherwise
  it counts available official training images, with zero validation images.
  A malformed, mismatched, or stale training manifest is `invalid`; HTTP reads
  never create or repair manifests. Dataset availability remains a directory
  presence signal, not an integrity certificate.
- **Scope:** No new endpoint, dependency, tensor preprocessing, or model. The
  existing typed summary and dataset view change with the API contract.

## ADR-009 — Recorded preprocessing shared across stages

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Wave 2 provides one `apply_preprocess` function in `ml/` for
  train, validation, test and inference. Canonical input is RGB HWC `uint8`
  in [0, 255]; output is contiguous CHW `float32`, using `(pixel / 255 - mean)
  divided by std per channel. Output is not bounded to [0, 1]. Spatial size and
  normalization are loaded from `training_defaults.yaml`, without copied numeric
  defaults in Python. PyTorch conversion remains a Wave 3 model-boundary concern.
- **Geometry:** Images resize to the configured square with Pillow bilinear
  resampling. Optional grayscale uses Pillow L conversion, replicated to RGB
  before per-channel normalization. Binary H×W uint8 0/255 masks resize with
  nearest-neighbor sampling and follow exactly the image's horizontal flip.
- **Augmentation:** Only train may use horizontal flipping. It must be explicitly
  permitted in the canonical category entry and enabled by experiment probability.
  Every initial category denies it until its orientation assumptions are reviewed.
  Enabled train augmentation requires an explicit integer seed; local RNG draws
  do not change global randomness. Later training loops must supply per-sample,
  per-epoch seeds. Validation/test/inference stay deterministic and unaugmented.
- **Provenance:** Each result records effective configuration, versioned config
  SHA-256, source/output shapes, stage, seed, sampled decisions, operations and
  NumPy/Pillow versions as JSON-safe metadata. Before/after previews use display
  RGB pixels, not the normalized model array. Inputs are never modified.
- **Boundary:** The user explicitly authorized Wave 2 while the real-MVTec
  acceptance gate is open. Synthetic evidence may complete engineering work;
  real-data preparation and actual model consumers remain unverified until their
  own gates. No new package, HTTP route, training loop or model readiness change.
