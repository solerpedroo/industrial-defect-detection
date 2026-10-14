# Generated data manifests

Split manifests, checksums, and preprocessing records created by the training
pipeline belong here. They are gitignored. Category definitions stay in
`ml/configs/`.

Wave 1 creates `splits/<category>.json` only after a successful local integrity
audit. Each version-1 manifest records `category_id`, `seed`,
`validation_fraction`, `train_ids`, `validation_ids` and `train_checksums`.
All IDs refer to normal images under `<category>/train/good/`, relative to the
dataset root. Do not commit these files. Changing training files invalidates the
manifest; normal partition consumers raise an error instead of fitting on a
fallback partition. See [`docs/architecture.md`](../../docs/architecture.md).
