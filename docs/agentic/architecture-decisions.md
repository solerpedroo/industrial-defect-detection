# Architecture decisions

This log is the durable decision record for people and agents. Add an entry
before changing a cross-cutting convention. Accepted decisions supersede informal
chat context; implementation details belong in task cards.

## ADR-015 — Offline report packages and sealed experiment B/C

- **Status:** accepted
- **Date:** 2026-10-14
- **Decision:** Wave 8 writes report packages under `sample_outputs/<protocol_id>/`
  (or `REPORTS_ROOT`) as `report.md`, `results.json`, `results.csv` and an
  optional `image_auroc.png`. `GET /api/reports/{id}` reads that markdown and
  lists PNG names. It does not train, score, calibrate or return figure bytes.
- **Experiment B:** Noise, blur and brightness use the intensities in
  `experiments/robustness/config.yaml`, plus an undegraded original copy.
  The same image ids are scored in every condition. Thresholds are copied from
  a completed experiment A report (`normal_validation`). Degraded pixels stay
  in memory and are not written back to the dataset.
- **Experiment C:** Subset membership for 25/50/75/100 percent and three repeats
  is a seeded plan over normal train ids only. It is not a measured AUROC.
  The split loader still requires the full normal inventory, so those draws
  cannot be persisted as fit manifests. Reports say that retraining is blocked.
- **Claims:** Packages repeat that scores, heatmaps and mask area are not
  physical severity. A missing measurement is an explicit blocked section, not
  a fabricated metric. Real MVTec and CUDA gates stay open.
- **Safety:** Report ids must be catalog ids. The reader rejects symlinks and
  paths that leave the reports root. Markdown larger than 1 MiB returns 413.

## ADR-014 — Local saved-report and live-model comparison

- **Status:** accepted
- **Date:** 2026-10-14
- **Scope:** User authorized Wave 7 engineering; no new endpoint or dependency.
  The existing compare endpoint serves two installed distinct models on one upload.
  The UI consumes catalogs and clears/aborts obsolete or failed requests.
- **Saved reports:** Explicit local selection of Wave 5 `results.json` (up to
  16 MiB), held only in browser memory. Validate completed schema/phase, catalog
  membership, row coverage, numbers/null reasons, calibration source and identical
  per-category test IDs/labels before displaying. Preserve recorded hashes and
  provenance; structural checks do not authenticate a report or recompute metrics.
  No local paths, filesystem API, persistence or report upload endpoint.
- **Interpretation:** Compare model columns within one run and category only.
  Display test split, synthetic/partial scope, resolution, seed, hashes, timing
  and approximate-memory conventions. Saved metrics do not describe a live upload
  or necessarily the currently installed checkpoints. Raw scores and heat colors
  remain model-specific; do not rank them on a shared numerical scale.
- **Export:** Client CSV contains the visible category's recorded values, explicit
  null reasons and provenance. Quote/escape fields and neutralize spreadsheet
  formula prefixes. Never export invented zeros for undefined metrics.
- **Boundary:** Real MVTec experiment A, CUDA, experiment B/C and report generation
  remain separate gates. Browser fixtures use generated synthetic real-model
  reports, not fabricated benchmark claims.

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

## ADR-010 — Offline normal-only autoencoder and trusted checkpoints

- **Status:** accepted
- **Date:** 2026-10-14
- **Dependency:** PyTorch `>=2.10,<2.11` is a separate `training` extra. CPU wheels
  are used for local regression evidence and backend CI; the API/e2e shell does
  not require torch. GPU users choose the official compatible CUDA wheel.
- **Model:** Three stride-2 convolution/ReLU encoder blocks and mirrored
  transposed-convolution decoder blocks ending in sigmoid. Channel widths live
  in YAML. Spatial size is divisible by eight; all weights are trainable, with
  no pretrained or frozen backbone. Input/output are BCHW RGB float32 [0,1].
  Autoencoder-specific mean [0,0,0] and std [1,1,1] live in training defaults;
  the model consumes the shared Wave 2 function in both fit and prediction.
- **Fit:** Validated Wave 1 normal train/validation IDs are the only sources.
  Each read fingerprints the exact decoded bytes; a final manifest recheck
  rejects drift. Adam minimizes MSE. Epoch losses are sample-weighted; strict
  validation improvement publishes the best checkpoint. Patience stops training
  after consecutive non-improvements. Official test images/masks are never opened.
- **Reproducibility:** Local loader generators, explicit initialization seed and
  per-sample/per-epoch SHA-256 augmentation seeds; deterministic torch algorithms
  with caller RNG/backend flags restored. Metadata records config hash, full
  normal split provenance, history, library versions, device and duration. Same
  environment reproduction is tested; cross-device/version bit identity is not promised.
- **Artifacts:** New run directories only, under ignored repository artifact
  roots or explicit external directories outside the dataset. Best weights and
  loss history use atomic file replacement within the owned run. Completed weights
  are staged, their exact checksum goes into final history, and `best.pt` is
  published last as the readiness marker; history alone does not prove completion.
  Internal `.tmp` staging files cannot be loaded as prepared `.pt` checkpoints.
  Checkpoints hold
  state_dict plus primitive metadata, not pickled modules or optimizer resume state.
  Loading requires an explicit trusted root, refuses escaping/symlink/special
  paths, uses CPU `weights_only=True`, validates schema/config/state and then
  transfers the model to the requested CPU/CUDA device.
- **Prediction:** eval plus inference_mode returns square HWC reconstruction
  float32 [0,1], HW channel-mean squared error float32 [0,1], and its spatial mean
  as image score. No threshold, label, original-resolution localization or severity.
  The API catalog remains `planned`; HTTP serving and calibrated decisions are later gates.
- **Boundary:** The user authorized Wave 3 without closing real-MVTec acceptance.
  Engineering evidence uses synthetic images only. No dataset or checkpoint is committed.
- **References:** [PyTorch installation](https://pytorch.org/get-started/previous-versions/),
  [restricted loading](https://docs.pytorch.org/docs/2.10/generated/torch.load.html),
  [reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

## ADR-011 — Frozen PatchCore memory through Anomalib

- **Status:** accepted
- **Date:** 2026-10-14
- **Dependencies:** Optional `patchcore` extra pins Anomalib 2.7.0, torchvision
  0.25.x and timm 1.0.x, retaining torch 2.10.x. CPU wheels are used for regression
  evidence/CI. API import paths stay independent of these optional libraries.
- **Adapter:** Use Anomalib's low-level PatchcoreModel, not its Lightning trainer,
  postprocessing, automatic thresholds or dataset splitting. The project owns
  validated manifest selection and recorded Wave 2 preprocessing. Input is RGB
  HWC uint8 [0,255], resized and ImageNet-normalized to CHW float32. Output is a
  nonnegative finite square HW raw feature-distance map, with no upper bound.
- **Backbone:** ResNet-18 layer2/layer3, pretrained by default, all parameters
  frozen and feature extractor held in eval. No optimizer/backpropagation or
  batchnorm updates during memory construction. Random-weight mode is explicit,
  recorded and intended only for isolated offline regression tests, never passed
  off as pretrained evidence. Initial pretrained weights may populate local cache;
  artifacts include the extracted backbone state so reload never downloads them.
- **Memory:** Only retained normal fit IDs are decoded; validation pixels and
  test/masks stay sealed. Manifest checks hash normal validation bytes for drift
  without decoding them. Read hashes match the decoded fit bytes. Anomalib extracts/aggregates
  features and samples a seeded k-center coreset using the YAML ratio. Require
  `floor(patches * ratio) >= num_neighbors` to avoid empty/degenerate support.
  Total patch count is bounded explicitly; no silent reservoir or ratio change.
- **Score:** Return Anomalib's raw weighted maximum-patch nearest-neighbor score
  (plain maximum when neighbors=1), not a spatial mean or an Autoencoder-scale
  equivalent. AnomalyPrediction/AnomalyPredictor document common shapes and
  differing numerical scales. No score calibration, threshold, label or severity.
- **Artifacts:** New ignored/external run only, outside sources. Restricted
  CPU weights_only serialization/load of complete model state plus primitive
  provenance; trusted-root/symlink/special-path and schema/state checks. A staged
  prepared `.pt` publishes last, after JSON metadata; reload freezes/evals the
  stored backbone and dynamically restores memory with no pretrained retrieval.
- **Boundary:** User authorized Wave 4 engineering before real-MVTec acceptance.
  CPU synthetic and pretrained smoke evidence are distinct from real benchmark,
  CUDA or HTTP serving. Model catalogs stay `planned` until the serving gate.
- **Evidence:** 213 Python tests pass with the actual pretrained smoke enabled;
  independent review found no blockers. Scoped RNG, float32 dtype and backend
  settings are restored. Upstream TorchScript/Python 3.14 and deprecated-import
  warnings are retained, not evidence of cross-version or CUDA compatibility.
- **References:** [Anomalib](https://pypi.org/project/anomalib/2.7.0/),
  [PatchCore model](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/patchcore/torch_model.py),
  [k-center sampler](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/components/sampling/k_center_greedy.py).

## ADR-012 — Validation-frozen offline evaluation

- **Status:** accepted
- **Date:** 2026-10-14
- **Scope:** Wave 5 numerical metrics/calibration and offline experiment A only;
  no new dependency, HTTP readiness, UI, robustness or subset study.
- **Calibration:** Per category/model, use normal held-out image scores and pooled
  normal map pixels, separately. Threshold is NumPy's higher empirical quantile
  at `1 - target_fpr`; anomaly means strictly `value > threshold`. Record the
  target, quantile/method/comparator, observed calibration exceedance, counts,
  source IDs/checksums and model/config hash. This controls empirical calibration
  exceedance conservatively, not population FPR; spatial pixels are correlated.
  Small validation splits can only support coarse operating points. Never pick
  thresholds/normalization/hyperparameters from official test scores or masks.
- **Order:** Validate trusted artifacts against current normal split provenance,
  freeze and persist all requested category/model thresholds before any official
  test audit/decoding. Both models must share test IDs, resolution and geometry;
  family-specific normalization/scores remain distinct. Test masks resize with
  nearest-neighbor; report at model resolution, not original pixel geometry.
- **Metrics:** NumPy exact grouped tied-score ROC trapezoids and non-interpolated
  average precision (`aupr` fields mean AP, not trapezoidal PR area). Global pixel
  AUROC/AP, micro IoU/Dice; image precision/recall/F1 and TP/FP/TN/FN with frozen
  image threshold. Undefined ratios/rank metrics are JSON null with explicit
  reasons, never invented zero. AUPRO averages 8-connected regions equally,
  integrates the grouped threshold curve to configured background FPR limit with
  linear endpoint interpolation and divides by that limit. No per-region or
  per-image renormalization of raw scores.
  Accepted score conversion to float64 must be lossless; integers above 2**53
  are explicitly rejected rather than becoming fabricated ties.
- **Compute/provenance:** Warmup precedes timed per-image end-to-end prediction
  (preprocessing included; file decode/load excluded). CUDA synchronizes around
  timing and records allocator peak separately. CPU peak is explicitly an
  approximate process high-water RSS, not isolated model allocation. Record
  artifact bytes/hash, preparation duration, versions, hardware, config/seed,
  test image/mask byte hashes, timings and both thresholds. Pixel count budgets
  fail explicitly; there is no hidden subsampling. Completed JSON publishes last
  after CSV; interrupted runs are not completed reports.
  Autoencoder duration is read from regular complete `history.json`, bound to
  checkpoint SHA/config/split, with sidecar drift checks. Missing sidecar means
  null duration with an explicit source; invalid provenance is rejected.
- **Boundary:** User authorized Wave 5 despite the real-data gate. Synthetic
  integration evidence is not experiment A's real-MVTec result. Catalogs stay
  `planned`. Protocol config is fixed before the final official test report;
  code cannot prevent a user deliberately rerunning/tuning separate experiments.
- **References:** [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
  [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
  [Anomalib metric definitions](https://anomalib.readthedocs.io/en/lib-v2.7.0/markdown/guides/reference/metrics/index.html).

## ADR-013 — Calibration-bound ephemeral inference

- **Status:** accepted
- **Date:** 2026-10-14
- **Scope:** Wave 6 upload-to-result slice using existing endpoints/libraries.
  Keep FastAPI/Next.js deployment; no new service, training endpoint or dependency.
- **Readiness:** `INFERENCE_MANIFEST` optionally points to an operator-owned local
  JSON manifest with explicit trusted root, device and category/model checkpoint
  and Wave 5 `calibration.json` paths. Reject unsafe paths, hashes, mismatched
  config/preprocessing/normal split/validation membership or quantile conventions.
  Load/freeze/cache/warm predictors once at startup; serialize prediction use.
  Default catalog YAML stays `planned`; API overlays `ready` only for validated
  installed categories, with `ready_categories` and `synthetic_categories`.
  Invalid configured serving fails startup; uninstalled pairs retain 501.
- **Calibration:** Serve frozen normal-validation thresholds only, strict `>`.
  Startup may read trusted artifacts/provenance, never dataset/test pixels, train
  or recalibrate. A Wave 5 frozen calibration is sufficient independently of a
  completed test report. Bind its bytes/hash and recorded model/source/config.
  Synthetic calibrations require `INFERENCE_ALLOW_SYNTHETIC=1`; mark response/UI
  as synthetic demonstration, never industrial readiness. Real-data/CUDA gates
  remain open even though the HTTP engineering path is implemented.
- **Transport:** Preserve upload checks; bound application reads to limit+1,
  close UploadFile on every path and keep payload/results request-scoped. Multipart
  infrastructure may spool temporary bytes; no application persistence/logging.
  A pre-parse ASGI envelope bounds inference bodies to 10 MiB plus 64 KiB of
  multipart overhead, including chunked requests, before FastAPI parses files.
  Offload blocking prediction/PNG work from async handlers. Optional model extras
  import only when configured. Health/catalog errors expose no host paths.
- **Geometry/display:** All output PNGs share the recorded model square (not
  original localization). Return prepared RGB, raw-distance heatmap, 50% overlay,
  binary pixel mask and optional reconstruction as base64 PNG, never raw tensors.
  Heat color uses fixed [0,2*pixel_threshold] per installed calibration, not
  upload min/max; zero threshold uses a zero/nonzero display convention. Color
  saturation is display-only. No boxes or physical severity claim.
- **Timing/UI:** `inference_ms` is synchronized prediction/preprocessing only,
  excluding upload decode and PNG encode; request_ms includes queue/transport work
  before JSON serialization. UI submits explicitly, clears results on input
  changes/failure and aborts/ignores obsolete responses. Opacity is client-only;
  PNG downloads use current ephemeral response, with clear resolution and
  calibration/score-scale limitations. No saved-experiment dashboard in this wave.
- **References:** [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
  [lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/),
  [Starlette thread pool](https://www.starlette.io/threadpool/).
