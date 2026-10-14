"""Wave 1 regression evidence: tiny uint8 RGB/L trees, normal holdout seed 42."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import create_app
from ml.datasets import (
    DatasetIntegrityError,
    ManifestError,
    create_split_manifest,
    inspect_category,
    load_split_manifest,
    partition_samples,
    summarize_category,
)
from ml.datasets.splits import manifest_path


async def _client_get(path: str) -> tuple[int, dict]:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        response = await client.get(path)
    return response.status_code, response.json()


def _image(path: Path, value: int, *, mode: str = "RGB", size=(4, 3)) -> None:
    """Synthetic 8-bit pixels in [0, 255]; each image has a distinct constant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (value, value + 1, value + 2) if mode == "RGB" else value
    Image.new(mode, size, color).save(path)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "mvtec-ad"
    manifests = tmp_path / "manifests"
    for index in range(5):
        _image(root / "bottle" / "train" / "good" / f"{index:03}.png", index * 10)
    _image(root / "bottle" / "test" / "good" / "000.png", 100)
    _image(root / "bottle" / "test" / "broken" / "000.png", 150)
    _image(root / "bottle" / "ground_truth" / "broken" / "000_mask.png", 255, mode="L")
    monkeypatch.setattr("ml.datasets.splits.DEFAULT_MANIFEST_DIR", manifests)
    monkeypatch.setattr("app.services.catalog.dataset_root", lambda: root)
    return root, manifests


def _create(dataset, **kwargs):
    root, manifests = dataset
    return create_split_manifest(
        root, "bottle", seed=42, validation_fraction=0.4, manifest_dir=manifests, **kwargs
    )


def test_audit_pairs_masks_and_keeps_source_files(dataset):
    root, _ = dataset
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report = inspect_category(root, "bottle")
    assert report.valid
    assert len(report.samples) == 7
    assert len(report.train_checksums) == 5
    anomaly = next(sample for sample in report.samples if sample.is_anomalous)
    assert anomaly.split == "test"
    assert anomaly.mask_id == "bottle/ground_truth/broken/000_mask.png"
    assert all(sample.mask_id is None for sample in report.samples if not sample.is_anomalous)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_summary_without_manifest_and_missing_category(dataset):
    root, _ = dataset
    summary = summarize_category(root, "bottle")
    assert summary["train_normal"] == 5
    assert summary["validation_normal"] == 0
    assert summary["test_normal"] == summary["test_anomalous"] == summary["mask_count"] == 1
    assert summary["validation_split"] == "not_created"
    missing = summarize_category(root, "leather")
    assert missing["availability"] == "missing"
    assert missing["train_normal"] == missing["validation_normal"] == 0
    assert not inspect_category(root, "leather").valid


def test_seeded_membership_is_disjoint_reproducible_and_normal_only(dataset):
    root, manifests = dataset
    manifest = _create(dataset)
    # Literal expected IDs pin the seed-42 protocol independently of the helper.
    assert manifest.validation_ids == ("bottle/train/good/001.png", "bottle/train/good/002.png")
    assert len(manifest.train_ids) == 3
    assert set(manifest.train_ids).isdisjoint(manifest.validation_ids)
    assert set(manifest.train_ids) | set(manifest.validation_ids) == set(manifest.train_checksums)
    assert all("/train/good/" in image_id for image_id in manifest.train_checksums)
    assert load_split_manifest(root, "bottle", manifest_dir=manifests) == manifest
    original = manifest_path("bottle", manifests).read_bytes()
    _create(dataset, overwrite=True)
    assert manifest_path("bottle", manifests).read_bytes() == original
    train = partition_samples(root, "bottle", "train")
    validation = partition_samples(root, "bottle", "validation")
    test = partition_samples(root, "bottle", "test")
    assert len(train) == 3 and len(validation) == 2 and len(test) == 2
    assert all(not sample.is_anomalous and sample.mask_id is None for sample in train + validation)
    assert {sample.image_id for sample in test}.isdisjoint(manifest.train_checksums)


def test_test_content_does_not_choose_validation_membership(dataset):
    root, manifests = dataset
    original = _create(dataset)
    _image(root / "bottle" / "test" / "good" / "000.png", 120)
    _image(root / "bottle" / "test" / "broken" / "000.png", 170)
    changed = create_split_manifest(
        root, "bottle", seed=42, validation_fraction=0.4, manifest_dir=manifests / "second"
    )
    assert changed == original


def test_train_consumer_and_summary_do_not_decode_official_test(dataset, monkeypatch):
    root, _ = dataset
    _create(dataset)

    def forbidden(*args, **kwargs):
        raise AssertionError("Train consumer or summary attempted to decode an image.")

    monkeypatch.setattr("PIL.Image.open", forbidden)
    assert len(partition_samples(root, "bottle", "validation")) == 2
    summary = summarize_category(root, "bottle")
    assert summary["train_normal"] == 3
    assert summary["validation_normal"] == 2
    assert summary["validation_split"] == "created"


@pytest.mark.parametrize("target", ["test/good/copy.png", "train/good/copy.png"])
def test_duplicate_content_rejects_manifest_build_even_with_new_encoding(dataset, target):
    root, manifests = dataset
    destination = root / "bottle" / target
    with Image.open(root / "bottle" / "train" / "good" / "000.png") as image:
        image.save(destination, compress_level=0)
    report = inspect_category(root, "bottle")
    assert not report.valid and report.duplicates
    assert destination.relative_to(root).as_posix() in report.duplicates[0]
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)
    assert not manifests.exists()
    assert destination.is_file()


def test_repeated_test_pixels_do_not_change_the_holdout(dataset):
    root, _ = dataset
    shutil.copyfile(root / "bottle/test/good/000.png", root / "bottle/test/good/copy.png")
    assert inspect_category(root, "bottle").valid
    assert len(_create(dataset).validation_ids) == 2


@pytest.mark.parametrize(
    "relative",
    [
        "train/good/000.png",
        "test/good/000.png",
        "ground_truth/broken/000_mask.png",
    ],
)
def test_corrupt_files_are_reported_and_preserved(dataset, relative):
    root, manifests = dataset
    path = root / "bottle" / relative
    path.write_bytes(b"corrupt-image")
    report = inspect_category(root, "bottle")
    assert any(issue.code == "unreadable_image" for issue in report.issues)
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)
    assert path.read_bytes() == b"corrupt-image"
    assert not manifests.exists()


@pytest.mark.parametrize(
    "problem, code",
    [
        ("missing", "missing_mask"),
        ("orphan", "orphan_mask"),
        ("size", "mask_size"),
        ("values", "nonbinary_mask"),
        ("channels", "nonbinary_mask"),
        ("name", "mask_name"),
    ],
)
def test_mask_integrity_errors(dataset, problem, code):
    root, _ = dataset
    mask = root / "bottle/ground_truth/broken/000_mask.png"
    if problem == "missing":
        mask.unlink()
    elif problem == "orphan":
        _image(mask.with_name("orphan_mask.png"), 0, mode="L")
    elif problem == "size":
        _image(mask, 255, mode="L", size=(2, 2))
    elif problem == "values":
        _image(mask, 127, mode="L")
    elif problem == "channels":
        _image(mask, 0)
    else:
        mask.rename(mask.with_name("wrong.png"))
    assert code in {issue.code for issue in inspect_category(root, "bottle").issues}
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)


@pytest.mark.parametrize("relative", ["train/defect/x.png", "train/good/nested/x.png"])
def test_unexpected_train_tree_is_rejected(dataset, relative):
    root, _ = dataset
    _image(root / "bottle" / relative, 200)
    report = inspect_category(root, "bottle")
    assert any(issue.code == "unexpected_file" for issue in report.issues)
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)


def test_symlink_is_not_followed(dataset):
    root, _ = dataset
    target = root / "bottle/train/good/000.png"
    (target.parent / "linked.png").symlink_to(target)
    assert "symlink" in {issue.code for issue in inspect_category(root, "bottle").issues}
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 0.99, float("nan"), float("inf")])
def test_invalid_or_empty_split_fails_without_writing(dataset, fraction):
    root, manifests = dataset
    with pytest.raises(ManifestError):
        create_split_manifest(root, "bottle", seed=42, validation_fraction=fraction)
    assert not manifests.exists()


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seed_fails_without_writing(dataset, seed):
    root, manifests = dataset
    with pytest.raises(ManifestError):
        create_split_manifest(root, "bottle", seed=seed, validation_fraction=0.4)
    assert not manifests.exists()


def test_existing_manifest_requires_explicit_overwrite(dataset):
    _, manifests = dataset
    _create(dataset)
    path = manifest_path("bottle", manifests)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _create(dataset)
    assert path.read_bytes() == before
    assert list(manifests.iterdir()) == [path]


def test_source_replacement_during_decode_cannot_publish_unaudited_bytes(dataset, monkeypatch):
    root, manifests = dataset
    source = root / "bottle/train/good/000.png"
    original = source.read_bytes()
    image_open = Image.open
    changed = False

    def replace_during_decode(stream, *args, **kwargs):
        nonlocal changed
        image = image_open(stream, *args, **kwargs)
        if not changed and stream.getvalue() == original:
            source.write_bytes(b"not-an-image")
            changed = True
        return image

    monkeypatch.setattr("ml.datasets.reader.Image.open", replace_during_decode)
    with pytest.raises(DatasetIntegrityError, match="changed during"):
        _create(dataset)
    assert changed
    assert not manifests.exists()


@pytest.mark.parametrize("entry", ["symlink", "notes", "directory", "fifo"])
def test_new_unexpected_training_entries_invalidate_manifest(dataset, entry):
    root, _ = dataset
    _create(dataset)
    directory = root / "bottle/train/good"
    if entry == "symlink":
        (directory / "linked.png").symlink_to(directory / "000.png")
    elif entry == "notes":
        (directory / "notes.txt").write_text("unexpected")
    elif entry == "directory":
        (root / "bottle/train/defect").mkdir()
    else:
        os.mkfifo(directory / "pipe.png")
    with pytest.raises(ManifestError):
        partition_samples(root, "bottle", "train")
    assert summarize_category(root, "bottle")["validation_split"] == "invalid"


def test_nonregular_image_entries_are_reported_without_opening_them(dataset):
    root, _ = dataset
    path = root / "bottle/train/good/pipe.png"
    os.mkfifo(path)
    report = inspect_category(root, "bottle")
    assert "nonregular_file" in {issue.code for issue in report.issues}
    with pytest.raises(DatasetIntegrityError):
        _create(dataset)


@pytest.mark.parametrize("problem", ["huge_fraction", "nested_json"])
def test_extreme_malformed_manifest_does_not_crash_summary_api(dataset, problem):
    _, manifests = dataset
    _create(dataset)
    path = manifest_path("bottle", manifests)
    if problem == "huge_fraction":
        document = json.loads(path.read_text())
        document["validation_fraction"] = 10**400
        path.write_text(json.dumps(document))
    else:
        path.write_text("[" * 2000 + "0" + "]" * 2000)
    status, payload = asyncio.run(_client_get("/api/datasets/bottle/summary"))
    assert status == 200
    assert payload["validation_split"] == "invalid"


def test_nonregular_manifest_is_invalid_without_blocking(dataset):
    _, manifests = dataset
    manifests.mkdir()
    os.mkfifo(manifest_path("bottle", manifests))
    result = _cli(dataset, "--category", "bottle")
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["categories"][0]["validation_split"] == "invalid"
    with pytest.raises(ManifestError):
        partition_samples(dataset[0], "bottle", "train")


def test_failed_atomic_publish_cleans_temporary_file_without_replacing_target(dataset, monkeypatch):
    _, manifests = dataset
    target = manifest_path("bottle", manifests)

    def concurrent_writer(source, destination):
        target.write_text("concurrent manifest")
        raise FileExistsError("Concurrent writer won.")

    monkeypatch.setattr("ml.datasets.splits.os.link", concurrent_writer)
    with pytest.raises(FileExistsError):
        _create(dataset)
    assert target.read_text() == "concurrent manifest"
    assert list(manifests.iterdir()) == [target]


@pytest.mark.parametrize("mutation", ["content", "add", "remove"])
def test_changed_train_inventory_or_content_invalidates_split(dataset, mutation):
    root, _ = dataset
    _create(dataset)
    image = root / "bottle/train/good/000.png"
    if mutation == "content":
        _image(image, 230)
    elif mutation == "add":
        _image(image.with_name("new.png"), 240)
    else:
        image.unlink()
    with pytest.raises(ManifestError):
        partition_samples(root, "bottle", "train")
    summary = summarize_category(root, "bottle")
    assert summary["validation_split"] == "invalid"
    assert summary["validation_normal"] == 0


@pytest.mark.parametrize("mutation", ["malformed", "category", "version", "overlap", "test_id"])
def test_invalid_manifest_never_exposes_a_valid_partition(dataset, mutation):
    root, manifests = dataset
    _create(dataset)
    path = manifest_path("bottle", manifests)
    document = json.loads(path.read_text())
    if mutation == "malformed":
        path.write_text("{invalid-json")
    else:
        if mutation == "category":
            document["category_id"] = "leather"
        elif mutation == "version":
            document["schema_version"] = 99
        elif mutation == "overlap":
            document["validation_ids"] = document["train_ids"]
        else:
            document["validation_ids"][0] = "bottle/test/good/000.png"
        path.write_text(json.dumps(document))
    with pytest.raises(ManifestError):
        partition_samples(root, "bottle", "train")
    assert summarize_category(root, "bottle")["validation_split"] == "invalid"


def test_missing_manifest_cannot_fall_back_to_official_train(dataset):
    root, _ = dataset
    with pytest.raises(FileNotFoundError):
        partition_samples(root, "bottle", "train")


@pytest.mark.parametrize("category_id", ["../bottle", "/tmp/bottle", "bottle/x"])
def test_category_ids_cannot_be_paths(dataset, category_id):
    root, _ = dataset
    with pytest.raises(ValueError):
        inspect_category(root, category_id)


def test_summary_endpoints_expose_real_partition_counts(dataset):
    _create(dataset)
    for endpoint in ("/api/datasets", "/api/datasets/bottle/summary"):
        status, payload = asyncio.run(_client_get(endpoint))
        summary = payload["categories"][0] if "categories" in payload else payload
        assert status == 200
        assert summary["train_normal"] == 3
        assert summary["validation_normal"] == 2
        assert summary["validation_split"] == "created"
        assert summary["test_normal"] == summary["test_anomalous"] == 1
        assert "train_ids" not in summary and "train_checksums" not in summary


def test_summary_endpoint_reports_invalid_manifest(dataset):
    _, manifests = dataset
    _create(dataset)
    manifest_path("bottle", manifests).write_text("null")
    status, payload = asyncio.run(_client_get("/api/datasets/bottle/summary"))
    assert status == 200
    assert payload["validation_split"] == "invalid"
    assert payload["validation_normal"] == 0 and payload["train_normal"] == 5


def _cli(dataset, *args):
    root, manifests = dataset
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/validate_dataset.py"),
            "--dataset-root",
            str(root),
            "--manifest-dir",
            str(manifests),
            "--json",
            *args,
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_cli_audit_is_read_only_and_create_is_explicit(dataset):
    _, manifests = dataset
    audit = _cli(dataset, "--category", "bottle")
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout)["categories"][0]["integrity"]["valid"]
    assert not manifests.exists()
    create = _cli(dataset, "--category", "bottle", "--create-split", "--validation-fraction", "0.4")
    assert create.returncode == 0, create.stderr
    assert json.loads(create.stdout)["categories"][0]["validation_normal"] == 2
    again = _cli(dataset, "--category", "bottle", "--create-split", "--validation-fraction", "0.4")
    assert again.returncode == 1
    assert json.loads(again.stdout)["errors"]


def test_cli_uses_all_catalog_categories_and_writes_nothing_when_any_missing(dataset):
    _, manifests = dataset
    result = _cli(dataset, "--create-split", "--validation-fraction", "0.4")
    assert result.returncode == 1
    assert [item["id"] for item in json.loads(result.stdout)["categories"]] == [
        "bottle",
        "leather",
        "metal_nut",
    ]
    assert not manifests.exists()


def test_cli_prepares_all_three_synthetic_categories(dataset):
    root, manifests = dataset
    for category_id in ("leather", "metal_nut"):
        shutil.copytree(root / "bottle", root / category_id)
    result = _cli(dataset, "--create-split", "--validation-fraction", "0.25")
    assert result.returncode == 0, result.stderr
    categories = json.loads(result.stdout)["categories"]
    assert len(categories) == 3
    for summary in categories:
        assert summary["integrity"]["valid"]
        assert summary["validation_split"] == "created"
        assert summary["train_normal"] == 3 and summary["validation_normal"] == 2
        manifest = load_split_manifest(root, summary["id"], manifest_dir=manifests)
        assert all(
            image_id.startswith(f"{summary['id']}/train/good/")
            for image_id in manifest.validation_ids
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--create-split"],
        ["--overwrite"],
        ["--category", "../bottle"],
        ["--create-split", "--validation-fraction", "nan"],
    ],
)
def test_cli_rejects_invalid_arguments(dataset, args):
    _, manifests = dataset
    result = _cli(dataset, "--category", "bottle", *args)
    assert result.returncode == 2
    assert not manifests.exists()
