"""Wave 2: synthetic RGB/mask uint8 [0, 255], fixed seeds, explicit stages."""

import json
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
import yaml
from PIL import Image

from app.core.images import decode_upload
from ml.preprocessing import (
    apply_preprocess,
    load_preprocess_config,
    preprocess_preview,
)
from ml.preprocessing.config import CONFIG_DIR, PreprocessConfig


@pytest.fixture(autouse=True)
def preserve_global_numpy_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def config():
    return replace(
        load_preprocess_config("bottle"),
        image_size=2,
        mean=(0, 0, 0),
        std=(1, 1, 1),
    )


@pytest.fixture
def image():
    return np.array(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )


def test_config_reads_size_statistics_and_policies_from_canonical_yaml():
    defaults = yaml.safe_load((CONFIG_DIR / "training_defaults.yaml").read_text())
    categories = yaml.safe_load((CONFIG_DIR / "categories.yaml").read_text())["categories"]
    assert [entry["id"] for entry in categories] == ["bottle", "leather", "metal_nut"]
    for entry in categories:
        config = load_preprocess_config(entry["id"])
        assert config.image_size == defaults["image_size"]
        assert config.mean == tuple(defaults["normalization"]["mean"])
        assert config.std == tuple(defaults["normalization"]["std"])
        assert config.grayscale is defaults["preprocessing"]["grayscale"]
        assert config.horizontal_flip_probability == 0
        assert config.allow_horizontal_flip is False


def test_hand_computed_rgb_channel_normalization(config):
    source = np.array([[[0, 127, 255]]], dtype=np.uint8)
    config = replace(config, image_size=1, mean=(0.25, 0.5, 0.75), std=(0.5, 0.25, 0.125))
    result = apply_preprocess(source, config)
    expected = np.array([[[-0.5]], [[(127 / 255 - 0.5) / 0.25]], [[2]]], dtype=np.float32)
    np.testing.assert_allclose(result.model_input, expected, rtol=1e-5, atol=1e-7)
    assert result.model_input.shape == (3, 1, 1)
    assert result.model_input.dtype == np.float32 and result.model_input.flags.c_contiguous
    np.testing.assert_array_equal(result.display_rgb, source)
    assert result.model_input.min() < 0 and result.model_input.max() > 1


def test_stages_share_numerics_when_augmentation_is_disabled(config, image):
    results = [
        apply_preprocess(image, config, stage=stage)
        for stage in ("train", "validation", "test", "inference")
    ]
    for result in results:
        np.testing.assert_array_equal(
            result.model_input, image.transpose(2, 0, 1) / np.float32(255)
        )
        assert result.record["config_hash"] == config.config_hash
        assert result.record["operations"][1]["applied"] is False


def test_bilinear_image_resize_and_nearest_mask_preserve_distinct_semantics(config):
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[1, 1] = 255
    mask = np.array([[0, 0], [0, 255]], dtype=np.uint8)
    result = apply_preprocess(pixels, replace(config, image_size=3), mask=mask, stage="test")
    assert result.display_rgb.shape == (3, 3, 3)
    assert result.display_rgb[1, 1, 0] == 64  # Average of the four source corners, rounded.
    assert result.mask.shape == (3, 3) and result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) == {0, 255}
    assert result.mask[1, 1] == 255


def test_grayscale_replication_has_hand_computed_luma(config):
    source = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    result = apply_preprocess(source, replace(config, image_size=3, grayscale=True))
    # Pillow L uses rounded 0.299 R + 0.587 G + 0.114 B for these primaries.
    np.testing.assert_array_equal(result.display_rgb[0, :, 0], [76, 150, 29])
    for channel in (1, 2):
        np.testing.assert_array_equal(
            result.display_rgb[:, :, channel], result.display_rgb[:, :, 0]
        )
        np.testing.assert_array_equal(result.model_input[channel], result.model_input[0])


def test_category_policy_denies_flipping_for_initial_categories():
    for category_id in ("bottle", "leather", "metal_nut"):
        with pytest.raises(ValueError, match="not permitted"):
            load_preprocess_config(category_id, horizontal_flip_probability=0.5)


def test_explicit_category_opt_in_is_read_from_catalog(tmp_path):
    for name in ("training_defaults.yaml", "categories.yaml"):
        (tmp_path / name).write_bytes((CONFIG_DIR / name).read_bytes())
    path = tmp_path / "categories.yaml"
    catalog = yaml.safe_load(path.read_text())
    catalog["categories"][0]["preprocessing"]["allow_horizontal_flip"] = True
    path.write_text(yaml.safe_dump(catalog))
    config = load_preprocess_config("bottle", config_dir=tmp_path, horizontal_flip_probability=0.5)
    assert config.allow_horizontal_flip and config.horizontal_flip_probability == 0.5
    with pytest.raises(ValueError, match="not permitted"):
        load_preprocess_config("leather", config_dir=tmp_path, horizontal_flip_probability=0.5)


def test_seeded_flip_is_paired_reproducible_and_leaves_inputs_and_rng_unchanged(config, image):
    config = replace(config, allow_horizontal_flip=True, horizontal_flip_probability=0.5)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    original_image, original_mask = image.copy(), mask.copy()
    np.random.seed(123)
    before = np.random.get_state()
    flipped = apply_preprocess(image, config, stage="train", seed=2, mask=mask)
    repeated = apply_preprocess(image, config, stage="train", seed=2, mask=mask)
    unchanged = apply_preprocess(image, config, stage="train", seed=42, mask=mask)
    after = np.random.get_state()
    np.testing.assert_array_equal(flipped.display_rgb, image[:, ::-1])
    np.testing.assert_array_equal(flipped.mask, mask[:, ::-1])
    np.testing.assert_array_equal(flipped.model_input, repeated.model_input)
    np.testing.assert_array_equal(unchanged.display_rgb, image)
    assert flipped.record["operations"][1]["applied"] is True
    assert unchanged.record["operations"][1]["applied"] is False
    assert repeated.record == flipped.record
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(mask, original_mask)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("stage", ["validation", "test", "inference"])
def test_nontraining_stages_never_augment_even_with_enabled_config(config, image, stage):
    config = replace(config, allow_horizontal_flip=True, horizontal_flip_probability=1)
    result = apply_preprocess(image, config, stage=stage)
    np.testing.assert_array_equal(result.display_rgb, image)
    assert result.record["operations"][1]["enabled"] is False
    assert result.record["seed"] is None
    with pytest.raises(ValueError, match="only accepted"):
        apply_preprocess(image, config, stage=stage, seed=42)


def test_preview_uses_display_pixels_and_independent_copies(config, image):
    config = replace(config, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), image_size=3)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    preview = preprocess_preview(image, config, mask=mask)
    result = apply_preprocess(image, config, mask=mask)
    assert preview.before_rgb.shape == (2, 2, 3)
    assert preview.after_rgb.shape == (3, 3, 3) and preview.after_rgb.dtype == np.uint8
    np.testing.assert_array_equal(preview.after_rgb, result.display_rgb)
    np.testing.assert_array_equal(preview.after_mask, result.mask)
    assert preview.record == result.record
    preview.before_rgb[:] = 255
    preview.before_mask[:] = 0
    assert image[0, 0, 0] == 0 and mask[0, 0] == 255


def test_json_record_tracks_effective_config_operations_versions_and_seed(config, image):
    result = apply_preprocess(image, config, stage="train", seed=42)
    record = json.loads(json.dumps(result.record, allow_nan=False))
    assert record == result.record
    assert record["source_shape"] == [2, 2, 3] and record["output_shape"] == [3, 2, 2]
    assert record["source_dtype"] == "uint8" and record["output_dtype"] == "float32"
    assert record["channel_order"] == "RGB" and record["seed"] == 42
    assert len(record["config_hash"]) == 64
    assert record["versions"]["numpy"] and record["versions"]["pillow"]
    assert [step["operation"] for step in record["operations"]] == [
        "resize",
        "horizontal_flip",
        "grayscale",
        "scale",
        "normalize",
        "layout",
    ]
    recipe = dict(record["config"])
    recipe.pop("schema_version")
    reconstructed = PreprocessConfig(**recipe)
    assert reconstructed.config_hash == config.config_hash
    np.testing.assert_array_equal(
        apply_preprocess(image, reconstructed).model_input, result.model_input
    )
    assert replace(config, image_size=3).config_hash != config.config_hash
    assert replace(config, grayscale=True).config_hash != config.config_hash


def test_upload_boundary_can_use_the_same_numerical_entrypoint(config, image):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    decoded = decode_upload(buffer.getvalue())
    training = apply_preprocess(image, config, stage="train")
    inference = apply_preprocess(decoded.image, config, stage="inference")
    np.testing.assert_array_equal(training.model_input, inference.model_input)


def test_noncontiguous_input_is_supported_without_aliasing(config, image):
    source = image[:, ::-1]
    result = apply_preprocess(source, config)
    assert result.model_input.flags.c_contiguous and result.display_rgb.flags.c_contiguous
    assert not np.shares_memory(source, result.display_rgb)
    assert not np.shares_memory(source, result.model_input)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.uint16),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        [[[0, 0, 0]]],
    ],
)
def test_invalid_image_input_fails_explicitly(config, source):
    with pytest.raises(ValueError):
        apply_preprocess(source, config)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2), dtype=bool),
        np.zeros((3, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
    ],
)
def test_invalid_mask_fails_explicitly(config, image, mask):
    with pytest.raises(ValueError, match="Mask"):
        apply_preprocess(image, config, mask=mask)


@pytest.mark.parametrize(
    "changes",
    [
        {"image_size": 0},
        {"image_size": -1},
        {"image_size": True},
        {"mean": (0, 0)},
        {"mean": (float("nan"), 0, 0)},
        {"mean": (10**400, 0, 0)},
        {"mean": (True, 0, 0)},
        {"std": (0, 1, 1)},
        {"std": (-1, 1, 1)},
        {"std": (float("inf"), 1, 1)},
        {"std": (1e-100, 1, 1)},
        {"grayscale": "false"},
        {"allow_horizontal_flip": 1},
        {"horizontal_flip_probability": float("nan")},
        {"horizontal_flip_probability": 1.1},
        {"horizontal_flip_probability": True},
        {"horizontal_flip_probability": 0.5},
    ],
)
def test_invalid_configuration_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


@pytest.mark.parametrize("seed", [None, -1, 2**32, True, 1.5])
def test_enabled_augmentation_requires_valid_explicit_seed(config, image, seed):
    config = replace(config, allow_horizontal_flip=True, horizontal_flip_probability=0.5)
    with pytest.raises(ValueError, match="[Ss]eed"):
        apply_preprocess(image, config, stage="train", seed=seed)


def test_invalid_stage_and_nonfinite_normalization_fail(config, image):
    with pytest.raises(ValueError, match="Stage"):
        apply_preprocess(image, config, stage="unknown")
    with pytest.raises(ValueError, match="non-finite"):
        apply_preprocess(image, replace(config, mean=(1, 1, 1), std=(1e-40, 1e-40, 1e-40)))


def test_unknown_category_and_malformed_catalogs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown category"):
        load_preprocess_config("unknown")
    for name in ("training_defaults.yaml", "categories.yaml"):
        (tmp_path / name).write_bytes((CONFIG_DIR / name).read_bytes())
    path = tmp_path / "training_defaults.yaml"
    path.write_text("image_size: 2\npreprocessing: []\nnormalization: {}\n")
    with pytest.raises(ValueError, match="Malformed"):
        load_preprocess_config("bottle", config_dir=tmp_path)
    path.write_text("[invalid yaml")
    with pytest.raises(ValueError, match="Cannot read"):
        load_preprocess_config("bottle", config_dir=tmp_path)
