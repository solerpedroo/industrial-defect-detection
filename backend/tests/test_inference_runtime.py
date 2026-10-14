"""Wave 6 trusted startup, actual models and ephemeral PNG/API regression evidence."""

import asyncio
import base64
import builtins
import json
import os
import threading
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from inference_fixture import prepare_serving
from PIL import Image

from app.core.upload_limit import MAX_INFERENCE_BODY, InferenceUploadLimit
from app.inference import load_registry
from app.main import create_app
from app.visualization import render_prediction


@pytest.fixture(autouse=True)
def preserve_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def prepared(tmp_path):
    return prepare_serving(tmp_path)


def _upload(width=21, height=35):
    image = Image.new("RGB", (width, height), (42, 43, 44))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _png(encoded):
    image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
    assert image.format == "PNG"
    image.load()
    return np.asarray(image)


def _client(prepared, monkeypatch):
    monkeypatch.setenv("INFERENCE_MANIFEST", str(prepared.manifest))
    monkeypatch.setenv("INFERENCE_ALLOW_SYNTHETIC", "1")
    return TestClient(create_app())


def test_real_cached_models_catalog_inference_and_compare(prepared, monkeypatch):
    with _client(prepared, monkeypatch) as client:
        cards = client.get("/api/models").json()["models"]
        assert all(
            card["status"] == "ready" and card["ready_categories"] == ["bottle"] for card in cards
        )
        assert all(card["synthetic_categories"] == ["bottle"] for card in cards)
        # Startup is complete. HTTP must not load/train/calibrate/read dataset files.
        monkeypatch.setattr(
            "ml.models.autoencoder.load_autoencoder", lambda *_a, **_k: pytest.fail("reload")
        )
        monkeypatch.setattr(
            "ml.models.patchcore.load_patchcore", lambda *_a, **_k: pytest.fail("reload")
        )
        monkeypatch.setattr(
            "ml.evaluation.calibrate_normal", lambda *_a, **_k: pytest.fail("calibration")
        )
        monkeypatch.setattr(
            "app.services.catalog.dataset_root", lambda: pytest.fail("dataset read")
        )
        for model in ("autoencoder", "patchcore"):
            response = client.post(
                "/api/inference",
                data={"category_id": "bottle", "model_id": model},
                files={"image": ("sample.png", _upload(), "image/png")},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["data_kind"] == "synthetic"
            assert result["map_width"] == result["map_height"] == 32
            assert result["label"] == (
                "anomalous" if result["score"] > result["threshold"] else "normal"
            )
            assert result["request_ms"] >= result["inference_ms"] >= 0
            assert len(result["artifact_sha256"]) == len(result["calibration_sha256"]) == 64
            for key in ("input_png", "anomaly_map_png", "overlay_png"):
                assert _png(result[key]).shape == (32, 32, 3)
            assert set(np.unique(_png(result["mask_png"]))) <= {0, 255}
            assert (result["reconstruction_png"] is not None) == (model == "autoencoder")
        response = client.post(
            "/api/inference/compare",
            data={"category_id": "bottle", "model_a": "autoencoder", "model_b": "patchcore"},
            files={"image": ("sample.png", _upload(), "image/png")},
        )
        assert response.status_code == 200, response.text
        assert [result["model_id"] for result in response.json()["results"]] == [
            "autoencoder",
            "patchcore",
        ]


def test_uninstalled_category_stays_501_and_bad_inputs_stay_4xx(prepared, monkeypatch):
    with _client(prepared, monkeypatch) as client:
        for category, model, content, expected in (
            ("leather", "autoencoder", _upload(), 501),
            ("unknown", "autoencoder", _upload(), 422),
            ("bottle", "unknown", _upload(), 422),
            ("bottle", "autoencoder", b"bad", 415),
            ("bottle", "autoencoder", b"", 400),
            ("bottle", "autoencoder", b"x" * (10 * 1024 * 1024 + 1), 413),
        ):
            response = client.post(
                "/api/inference",
                data={"category_id": category, "model_id": model},
                files={"image": ("sample.png", content, "image/png")},
            )
            assert response.status_code == expected


def test_unconfigured_lifespan_is_planned(monkeypatch):
    monkeypatch.delenv("INFERENCE_MANIFEST", raising=False)
    with TestClient(create_app()) as client:
        assert all(
            card["status"] == "planned" and not card["ready_categories"]
            for card in client.get("/api/models").json()["models"]
        )


def test_synthetic_requires_explicit_startup_opt_in(prepared):
    with pytest.raises(ValueError, match="Invalid serving"):
        load_registry(prepared.manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_sha256", "0" * 64),
        ("model_config_hash", "0" * 64),
        ("preprocessing_config_hash", "0" * 64),
        ("validation_ids", []),
        ("validation_checksums", {}),
    ],
)
def test_calibration_binding_failures(prepared, field, value):
    document = json.loads(prepared.calibration.read_text())
    document["calibration"]["bottle"]["autoencoder"][field] = value
    prepared.calibration.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_registry(prepared.manifest, allow_synthetic=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold", -1),
        ("threshold", float("nan")),
        ("threshold", True),
        ("source", "official_test"),
        ("comparator", ">="),
        ("method", "linear"),
        ("quantile", 0.5),
        ("sample_count", True),
        ("observed_fpr", 0.9),
    ],
)
def test_invalid_threshold_provenance(prepared, field, value):
    document = json.loads(prepared.calibration.read_text())
    document["calibration"]["bottle"]["autoencoder"]["image"][field] = value
    prepared.calibration.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_registry(prepared.manifest, allow_synthetic=True)


@pytest.mark.parametrize(
    "change", ["complete", "phase", "config", "split", "duplicate", "escape", "symlink", "fifo"]
)
def test_invalid_installations(prepared, change):
    manifest = json.loads(prepared.manifest.read_text())
    document = json.loads(prepared.calibration.read_text())
    if change == "complete":
        document["complete"] = True
    elif change == "phase":
        document["phase"] = "official_test_report"
    elif change == "config":
        document["config_hash"] = "0" * 64
    elif change == "split":
        document["normal_splits"]["bottle"]["train_ids"] = []
    elif change == "duplicate":
        manifest["entries"].append(manifest["entries"][0])
    elif change == "escape":
        manifest["entries"][0]["checkpoint"] = "../outside.pt"
    elif change == "symlink":
        link = prepared.root / "linked.pt"
        link.symlink_to(prepared.root / "ae/best.pt")
        manifest["entries"][0]["checkpoint"] = "linked.pt"
    elif change == "fifo":
        fifo = prepared.root / "pipe.json"
        os.mkfifo(fifo)
        manifest["entries"][0]["calibration"] = "pipe.json"
    prepared.manifest.write_text(json.dumps(manifest))
    prepared.calibration.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_registry(prepared.manifest, allow_synthetic=True)


def test_runtime_model_failure_is_sanitized_503(prepared, monkeypatch):
    with _client(prepared, monkeypatch) as client:
        registry = client.app.state.inference

        def broken(*_):
            raise RuntimeError("private /host/path")

        monkeypatch.setattr(registry, "predict", broken)
        response = client.post(
            "/api/inference",
            data={"category_id": "bottle", "model_id": "autoencoder"},
            files={"image": ("sample.png", _upload(), "image/png")},
        )
        assert response.status_code == 503
        assert "private" not in response.text


def test_display_threshold_equality_fixed_scale_and_overlay_pixels():
    prediction = SimpleNamespace(
        score=1.0,
        anomaly_map=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32),
        input_rgb=np.full((2, 2, 3), 100, dtype=np.uint8),
    )
    result = render_prediction(prediction, size=2, pixel_threshold=1.0)
    assert _png(result["mask_png"]).tolist() == [[0, 0], [255, 255]]
    assert result["display_max"] == 2
    heat = _png(result["anomaly_map_png"])
    assert np.array_equal(heat[1, 0], heat[1, 1])  # fixed calibrated saturation.
    assert np.array_equal(
        _png(result["overlay_png"]), np.rint(0.5 * 100 + 0.5 * heat).astype(np.uint8)
    )
    prediction.anomaly_map[1, 1] = 100
    assert np.array_equal(
        _png(render_prediction(prediction, size=2, pixel_threshold=1.0)["anomaly_map_png"]), heat
    )
    result = render_prediction(prediction, size=2, pixel_threshold=0.0)
    assert _png(result["mask_png"]).tolist() == [[0, 255], [255, 255]]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_display_threshold(value):
    with pytest.raises(ValueError):
        render_prediction(SimpleNamespace(), size=2, pixel_threshold=value)


def test_ae_only_installation_does_not_import_patchcore(prepared, monkeypatch):
    document = json.loads(prepared.manifest.read_text())
    document["entries"] = document["entries"][:1]
    prepared.manifest.write_text(json.dumps(document))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("anomalib", "ml.models.patchcore")):
            pytest.fail("AE-only serving imported optional PatchCore libraries")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    registry = load_registry(prepared.manifest, allow_synthetic=True)
    assert registry.categories("autoencoder") == ["bottle"]
    assert registry.categories("patchcore") == []


def test_serving_loads_without_dataset_and_uses_cached_snapshots(prepared, monkeypatch):
    prepared.dataset.rename(prepared.root / "offline-dataset")
    with _client(prepared, monkeypatch) as client:
        monkeypatch.setattr(
            "app.inference.runtime.file_checksum", lambda *_: pytest.fail("request disk read")
        )
        response = client.post(
            "/api/inference",
            data={"category_id": "bottle", "model_id": "autoencoder"},
            files={"image": ("sample.png", _upload(), "image/png")},
        )
        assert response.status_code == 200, response.text


def test_decode_is_off_event_loop_and_upload_closed(monkeypatch):
    from starlette.datastructures import UploadFile

    from app.api import routes

    monkeypatch.delenv("INFERENCE_MANIFEST", raising=False)
    original_decode, original_close = routes.decode_upload, UploadFile.close
    thread_ids, closed = [], []

    def decode(raw):
        thread_ids.append(threading.get_ident())
        return original_decode(raw)

    async def close(upload):
        await original_close(upload)
        closed.append(upload.file.closed)

    monkeypatch.setattr(routes, "decode_upload", decode)
    monkeypatch.setattr(UploadFile, "close", close)
    with TestClient(create_app()) as client:
        loop_thread = client.portal.call(threading.get_ident)
        response = client.post(
            "/api/inference",
            data={"category_id": "bottle", "model_id": "autoencoder"},
            files={"image": ("sample.png", _upload(), "image/png")},
        )
        assert response.status_code == 501
    assert thread_ids and all(identity != loop_thread for identity in thread_ids)
    assert closed and all(closed)


@pytest.mark.parametrize("declared", [None, str(MAX_INFERENCE_BODY + 1), "bad"])
def test_preparse_envelope_limits_declared_and_chunked_bodies(declared):
    async def run():
        calls, sent = [], []

        async def downstream(*_):
            calls.append("parsed")

        async def receive():
            return {"type": "http.request", "body": b"x" * (1024 * 1024), "more_body": True}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/api/inference", "headers": []}
        if declared is not None:
            scope["headers"] = [(b"content-length", declared.encode())]
        await InferenceUploadLimit(downstream)(scope, receive, send)
        assert not calls
        assert sent[0]["status"] == 413

    asyncio.run(run())
