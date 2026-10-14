# Demo storyboard

Record this against the current UI. Target length: 2–3 minutes. Do not commit
the video, the dataset, or a checkpoint. A session started with
`INFERENCE_ALLOW_SYNTHETIC=1` is enough for the engineering demo and must be
described as synthetic.

1. **Overview** — name Bottle, Leather, and Metal Nut, and the two model families. Say that training stays offline.
2. **Datasets** — show the license note. If the local MVTec copy is absent, say that the counts are not a prepared dataset.
3. **Inference Lab** — upload one image, select an installed category and model, and show the score, heatmap, overlay, and mask download. Say the score is not physical severity.
4. **Comparison** — use the same upload for the autoencoder and PatchCore. Say the heatmap scales are not a shared ranking.
5. **Saved metrics** — open a completed local `results.json`, filter one category, and export CSV. If a metric is null, show the recorded reason.
6. **Report package** — open `report.md` for protocol A, or a blocked package for protocol B or C. Read the limitation sentence. Say experiment C is a membership plan, not an AUROC.
7. **Close** — say that thresholds came from normal validation, and that real MVTec and CUDA are outside this recording.

Do not pan across the dataset directory. Link the official MVTec AD page instead of redistributing images.
