"""Wave 8 report packages, degradations and subset plans."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.datasets import SplitManifest
from ml.experiments.degrade import apply_degradation
from ml.experiments.protocols import (
    degradation_conditions,
    load_dataset_size_config,
    load_robustness_config,
)
from ml.experiments.render import LIMITATION, blocked_document, render_report
from ml.experiments.robustness import score_degraded_split, thresholds_from_experiment_a
from ml.experiments.subsets import (
    assert_subset_not_loadable,
    draw_normal_subset,
    plan_normal_subsets,
)


def test_brightness_and_blur_match_hand_values_without_mutating_the_source() -> None:
    source = np.full((4, 4, 3), 100, dtype=np.uint8)
    source[0, 0] = 250
    original = source.copy()
    dimmed = apply_degradation(source, "brightness", 0.6, seed=42, image_id="a")
    bright = apply_degradation(source, "brightness", 1.2, seed=42, image_id="a")
    blurred = apply_degradation(
        np.full((4, 4, 3), 40, dtype=np.uint8),
        "gaussian_blur",
        1,
        seed=42,
        image_id="a",
    )

    assert np.array_equal(source, original)
    assert dimmed[1, 1, 0] == 60
    assert bright[0, 0, 0] == 255
    assert np.all(blurred == 40)


def test_noise_is_reproducible_per_image_and_stays_in_8_bit() -> None:
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image_id = "bottle/test/good/000.png"
    first = apply_degradation(source, "gaussian_noise", 10, seed=42, image_id=image_id)
    second = apply_degradation(source, "gaussian_noise", 10, seed=42, image_id=image_id)
    other = apply_degradation(source, "gaussian_noise", 10, seed=42, image_id="other")

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, source)


def test_subset_draws_follow_the_recorded_fractions_and_seed() -> None:
    ids = tuple(f"train/{index}.png" for index in range(8))
    assert draw_normal_subset(ids, 1.0, 42, 0) == tuple(sorted(ids))
    assert draw_normal_subset(ids, 1.0, 42, 2) == draw_normal_subset(ids, 1.0, 42, 0)
    assert len(draw_normal_subset(ids, 0.25, 42, 0)) == 2
    assert len(draw_normal_subset(ids, 0.5, 42, 0)) == 4
    assert draw_normal_subset(ids, 0.25, 42, 0) != draw_normal_subset(ids, 0.25, 42, 1)
    assert draw_normal_subset(ids, 0.25, 42, 0) == draw_normal_subset(ids, 0.25, 42, 0)
    config = load_dataset_size_config()
    plan = plan_normal_subsets(
        ids,
        fractions=tuple(config["normal_train_fractions"]),
        seed=config["seed"],
        repeats=config["repeats"],
    )
    assert len(plan) == 12
    assert load_robustness_config()["degradations"]["gaussian_noise_sigma"] == [10, 20, 30]


def test_subset_plan_cannot_replace_the_validation_manifest() -> None:
    manifest = SplitManifest(
        schema_version=1,
        category_id="bottle",
        seed=42,
        validation_fraction=0.25,
        train_ids=("a", "b", "c", "d"),
        validation_ids=("e",),
        train_checksums={key: "ab" * 32 for key in "abcde"},
    )
    subset = draw_normal_subset(manifest.train_ids, 0.5, 42, 0)
    reason = assert_subset_not_loadable(manifest, subset)
    assert "Blocked" in reason
    assert set(subset).isdisjoint(manifest.validation_ids)


def test_degraded_scores_keep_ids_and_frozen_thresholds() -> None:
    records = []
    for index, label in enumerate((False, True)):
        image = np.full((8, 8, 3), 20 if label else 200, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        if label:
            mask[2:6, 2:6] = True
        records.append({"image_id": f"id-{index}", "image": image, "label": label, "mask": mask})

    def predict(image):
        score = float(image.mean())
        return score, np.full(image.shape[:2], score, dtype=np.float32)

    rows = score_degraded_split(
        records, predict, image_threshold=30.0, pixel_threshold=40.0, seed=42
    )
    assert [item[0] for item in degradation_conditions()] == [row["condition"] for row in rows]
    assert all(row["test_ids"] == ["id-0", "id-1"] for row in rows)
    assert all(row["image_threshold"] == 30.0 and row["pixel_threshold"] == 40.0 for row in rows)
    assert all(row["threshold_source"] == "frozen_normal_validation" for row in rows)
    assert rows[0]["metrics"]["image_auroc"] is not None
    assert rows[1]["condition"] == "gaussian_noise"


def test_thresholds_come_only_from_a_completed_experiment_a_report() -> None:
    document = {
        "complete": True,
        "phase": "official_test_report",
        "config": {"protocol_id": "exp-a-autoencoder-vs-patchcore"},
        "results": [
            {
                "category": "bottle",
                "model_id": "autoencoder",
                "thresholds": {
                    "image": {"threshold": 1.5, "source": "normal_validation"},
                    "pixel": {"threshold": 0.25, "source": "normal_validation"},
                },
            }
        ],
    }
    assert thresholds_from_experiment_a(document, "bottle", "autoencoder") == (1.5, 0.25)
    document["complete"] = False
    with pytest.raises(ValueError, match="completed"):
        thresholds_from_experiment_a(document, "bottle", "autoencoder")


def test_render_writes_package_and_states_the_severity_limitation(tmp_path) -> None:
    document = {
        "complete": True,
        "data_kind": "synthetic",
        "config": {"protocol_id": "exp-a-autoencoder-vs-patchcore"},
        "results": [
            {
                "category": "bottle",
                "model_id": "autoencoder",
                "metrics": {"image_auroc": 0.75, "pixel_auroc": 0.6},
                "thresholds": {"image": {"source": "normal_validation"}},
            }
        ],
    }
    report = render_report(
        document, tmp_path / "exp-a", protocol_id="exp-a-autoencoder-vs-patchcore"
    )
    text = report.read_text(encoding="utf-8")
    assert LIMITATION in text
    assert "0.75" in text
    assert "physical severity" in text
    saved = json.loads((tmp_path / "exp-a" / "results.json").read_text(encoding="utf-8"))
    assert saved["complete"] is True
    assert "image_auroc" in (tmp_path / "exp-a" / "results.csv").read_text(encoding="utf-8")
    with Image.open(tmp_path / "exp-a" / "image_auroc.png") as figure:
        assert figure.format == "PNG"


def test_blocked_protocol_c_has_a_plan_and_no_invented_auroc(tmp_path) -> None:
    document = blocked_document(
        "exp-c-dataset-size",
        "Retraining is blocked until the manifest contract allows a fit subset.",
        train_ids=["n0", "n1", "n2", "n3"],
    )
    report = render_report(document, tmp_path / "exp-c", protocol_id="exp-c-dataset-size")
    text = report.read_text(encoding="utf-8")
    assert "Blocked:" in text
    assert "fraction 0.25" in text
    assert "Image AUROC" not in text
    assert not (tmp_path / "exp-c" / "image_auroc.png").exists()
    saved = json.loads((tmp_path / "exp-c" / "results.json").read_text(encoding="utf-8"))
    assert saved["results"] == []
    assert saved["complete"] is False


def test_report_endpoint_returns_markdown_when_the_package_exists(tmp_path, monkeypatch) -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    from app.main import create_app

    monkeypatch.setenv("REPORTS_ROOT", str(tmp_path))
    document = blocked_document("exp-b-robustness", "No measurement file was provided.")
    render_report(document, tmp_path / "exp-b-robustness", protocol_id="exp-b-robustness")

    async def fetch(path: str):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://testserver"
        ) as client:
            response = await client.get(path)
        return response.status_code, response.json()

    status, payload = asyncio.run(fetch("/api/reports/exp-b-robustness"))
    assert status == 200
    assert LIMITATION in payload["markdown"]
    assert payload["figures"] == []
    missing, detail = asyncio.run(fetch("/api/reports/exp-c-dataset-size"))
    assert missing == 404
    assert "exp-c-dataset-size" in detail["detail"]


def test_report_endpoint_rejects_a_symlinked_package(tmp_path, monkeypatch) -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    from app.main import create_app

    package = tmp_path / "real"
    document = blocked_document("exp-a-autoencoder-vs-patchcore", "Missing measurement.")
    render_report(document, package, protocol_id="exp-a-autoencoder-vs-patchcore")
    root = tmp_path / "reports"
    root.mkdir()
    (root / "exp-a-autoencoder-vs-patchcore").symlink_to(package, target_is_directory=True)
    monkeypatch.setenv("REPORTS_ROOT", str(root))

    async def fetch():
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://testserver"
        ) as client:
            return await client.get("/api/reports/exp-a-autoencoder-vs-patchcore")

    response = asyncio.run(fetch())
    assert response.status_code == 404
    assert Path(package / "report.md").is_file()
