# Adding a model

Use this when a wave introduces an algorithm such as PaDiM. Read `AGENTS.md` and the anomaly-model skill first.

1. Add a catalog entry in `ml/configs/models.yaml` with `status: planned` until inference works.
2. Implement the algorithm under `ml/models/<name>/` behind the shared prediction dataclass in `ml/models/base.py`.
3. Keep training in `ml/training/` or a CLI script. Do not start it from FastAPI.
4. Fit on normal images only. Write checkpoints outside Git.
5. Add an adapter in `backend/app/inference/` that loads one artifact and returns score plus an anomaly map.
6. Flip `status` to `ready` in the same change that serves `POST /api/inference`.
7. Extend evaluation with the metrics in `ml/evaluation/metrics.py`. Choose thresholds on validation data.
8. Add tests with synthetic arrays. Do not commit MVTec images.

Parameter defaults live in the YAML catalogs. The UI reads `/api/models` and does not hard-code a second list.
