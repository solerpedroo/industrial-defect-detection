import asyncio
from io import BytesIO

from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import create_app


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _client_get(path: str) -> tuple[int, dict]:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        response = await client.get(path)
    return response.status_code, response.json()


async def _client_post(
    path: str, data: dict, content: bytes, name: str = "sample.png"
) -> tuple[int, dict]:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        response = await client.post(path, data=data, files={"image": (name, content, "image/png")})
    return response.status_code, response.json()


def test_datasets_list_the_three_starting_categories() -> None:
    status_code, payload = asyncio.run(_client_get("/api/datasets"))

    assert status_code == 200
    assert payload["id"] == "mvtec_ad"
    assert payload["commercial_use"] is False
    assert payload["storage_root"] == "data/mvtec-ad"
    assert [item["id"] for item in payload["categories"]] == ["bottle", "leather", "metal_nut"]
    assert payload["categories"][0]["validation_split"] == "not_created"


def test_unknown_category_summary_is_not_found() -> None:
    status_code, payload = asyncio.run(_client_get("/api/datasets/unknown/summary"))

    assert status_code == 404
    assert "unknown" in payload["detail"]


def test_models_stay_planned_until_a_later_wave() -> None:
    status_code, payload = asyncio.run(_client_get("/api/models"))

    assert status_code == 200
    assert [item["id"] for item in payload["models"]] == ["autoencoder", "patchcore"]
    assert {item["status"] for item in payload["models"]} == {"planned"}


def test_experiments_expose_three_protocols_without_metrics_files() -> None:
    status_code, payload = asyncio.run(_client_get("/api/experiments"))

    assert status_code == 200
    assert [item["id"] for item in payload["experiments"]] == [
        "exp-a-autoencoder-vs-patchcore",
        "exp-b-robustness",
        "exp-c-dataset-size",
    ]
    assert all(item["status"] == "planned" for item in payload["experiments"])


def test_missing_report_names_the_protocol() -> None:
    status_code, payload = asyncio.run(_client_get("/api/reports/exp-a-autoencoder-vs-patchcore"))

    assert status_code == 404
    assert "exp-a-autoencoder-vs-patchcore" in payload["detail"]


def test_inference_rejects_unknown_category_before_scoring() -> None:
    status_code, payload = asyncio.run(
        _client_post(
            "/api/inference",
            {"category_id": "cable", "model_id": "autoencoder"},
            _png_bytes(),
        )
    )

    assert status_code == 422
    assert "cable" in payload["detail"]


def test_inference_rejects_non_image_bytes() -> None:
    status_code, payload = asyncio.run(
        _client_post(
            "/api/inference",
            {"category_id": "bottle", "model_id": "autoencoder"},
            b"not-an-image",
            name="notes.txt",
        )
    )

    assert status_code == 415


def test_inference_of_a_planned_model_is_not_implemented() -> None:
    status_code, payload = asyncio.run(
        _client_post(
            "/api/inference",
            {"category_id": "bottle", "model_id": "autoencoder"},
            _png_bytes(),
        )
    )

    assert status_code == 501
    assert "artifact" in payload["detail"]


def test_compare_requires_two_different_models() -> None:
    status_code, payload = asyncio.run(
        _client_post(
            "/api/inference/compare",
            {"category_id": "bottle", "model_a": "autoencoder", "model_b": "autoencoder"},
            _png_bytes(),
        )
    )

    assert status_code == 422


def test_local_category_counts_when_the_tree_exists(tmp_path, monkeypatch) -> None:
    bottle = tmp_path / "bottle"
    (bottle / "train" / "good").mkdir(parents=True)
    (bottle / "test" / "good").mkdir(parents=True)
    (bottle / "test" / "broken_large").mkdir(parents=True)
    (bottle / "ground_truth" / "broken_large").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(bottle / "train" / "good" / "a.png")
    Image.new("RGB", (2, 2)).save(bottle / "test" / "good" / "b.png")
    Image.new("RGB", (2, 2)).save(bottle / "test" / "broken_large" / "c.png")
    Image.new("L", (2, 2)).save(bottle / "ground_truth" / "broken_large" / "c_mask.png")

    monkeypatch.setattr("app.services.catalog.dataset_root", lambda: tmp_path)
    from app.services.catalog import category_summary

    summary = category_summary("bottle")
    assert summary is not None
    assert summary.availability == "present"
    assert summary.train_normal == 1
    assert summary.test_normal == 1
    assert summary.test_anomalous == 1
    assert summary.mask_count == 1
    assert summary.defect_types[0].id == "broken_large"
