"""Hand-computed metric/calibration evidence on tiny binary synthetic arrays."""

from dataclasses import replace

import numpy as np
import pytest

from ml.evaluation import (
    aupro,
    calibrate_normal,
    confusion_metrics,
    evaluate_predictions,
    load_evaluation_config,
    ranking_metrics,
)
from ml.evaluation.metrics import _region_weights


def test_hand_computed_ranking_and_confusion():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    assert ranking_metrics(labels, scores) == pytest.approx({"auroc": 0.75, "aupr": 5 / 6})
    result = confusion_metrics(labels, scores, 0.36)
    assert {key: result[key] for key in ("tp", "fp", "tn", "fn")} == dict.fromkeys(
        ("tp", "fp", "tn", "fn"), 1
    )
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["false_positive_rate"] == result["false_negative_rate"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["dice"] == 0.5


@pytest.mark.parametrize(
    "labels,scores,auc,ap",
    [
        ([0, 1], [0, 1], 1, 1),
        ([0, 1], [1, 0], 0, 0.5),
        ([0, 1], [0.5, 0.5], 0.5, 0.5),
        ([0, 1, 1, 0], [1, 1, 1, 1], 0.5, 0.5),
        ([1, 1], [0, 1], None, 1),
        ([0, 0], [0, 1], None, None),
    ],
)
def test_ties_and_single_classes(labels, scores, auc, ap):
    assert ranking_metrics(labels, scores) == {"auroc": auc, "aupr": ap}


def test_undefined_ratios_are_not_fake_zero():
    result = confusion_metrics([0, 0], [0, 0], 0)
    assert result["precision"] is result["recall"] is result["f1"] is None
    assert result["iou"] is result["dice"] is None
    assert result["false_positive_rate"] == 0
    missed = confusion_metrics([1], [0], 0)
    assert missed["precision"] is None
    assert missed["recall"] == missed["f1"] == 0


@pytest.mark.parametrize(
    "target,score",
    [
        ([], []),
        ([0], [0, 1]),
        ([2], [0]),
        ([np.nan], [0]),
        ([0], [np.nan]),
        ([0], [np.inf]),
        ([0], [-1]),
        (["0"], [0]),
        ([0], ["1"]),
        ([0], [True]),
    ],
)
def test_invalid_binary_inputs(target, score):
    with pytest.raises(ValueError):
        ranking_metrics(target, score)


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), True])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        confusion_metrics([0], [0], threshold)


def test_normal_quantile_higher_and_strict_comparator():
    threshold = calibrate_normal(np.array([1, 2, 3, 4, 5], dtype=np.float32), target_fpr=0.25)
    assert threshold.threshold == 4
    assert threshold.quantile == 0.75
    assert threshold.sample_count == 5
    assert threshold.observed_fpr == 0.2
    assert threshold.to_dict()["source"] == "normal_validation"
    assert threshold.method == "higher" and threshold.comparator == ">"
    assert confusion_metrics([0], [4], threshold.threshold)["fp"] == 0
    assert calibrate_normal([2, 2, 2], target_fpr=0.05).observed_fpr == 0


def test_calibration_empirical_exceedance_is_conservative():
    rng = np.random.default_rng(42)
    for count in (1, 2, 3, 10, 100):
        for alpha in (0.01, 0.05, 0.2, 0.8):
            result = calibrate_normal(rng.integers(0, 10, size=count), target_fpr=alpha)
            assert result.observed_fpr <= alpha


@pytest.mark.parametrize(
    "values,alpha",
    [
        ([], 0.05),
        ([np.nan], 0.05),
        ([np.inf], 0.05),
        ([-1], 0.05),
        ([True], 0.05),
        ([0], 0),
        ([0], 1),
        ([0], np.nan),
        ([0], True),
    ],
)
def test_invalid_calibration(values, alpha):
    with pytest.raises(ValueError):
        calibrate_normal(values, target_fpr=alpha)


def test_aupro_perfect_reverse_tied_and_clipped_endpoint():
    mask = np.array([[[1, 0]]], dtype=np.uint8)
    assert aupro(mask, [[[2, 1]]], fpr_limit=0.3) == 1
    assert aupro(mask, [[[1, 2]]], fpr_limit=0.3) == 0
    # Tied foreground/background: PRO=FPR diagonal, normalized area L/2.
    assert aupro(mask, [[[1, 1]]], fpr_limit=0.3) == pytest.approx(0.15)
    assert aupro(mask, [[[1, 1]]], fpr_limit=1) == 0.5


def test_aupro_equal_region_weight_not_pixel_area():
    mask = np.array([[[1, 0, 1, 1]]], dtype=np.uint8)
    assert aupro(mask, [[[3, 2, 1, 1]]], fpr_limit=0.3) == 0.5
    assert aupro(mask, [[[1, 2, 3, 3]]], fpr_limit=0.3) == 0.5


def test_region_connectivity_and_image_boundaries():
    masks = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=bool)
    weights, count = _region_weights(masks)
    assert count == 2  # diagonal neighbors connect, separate images never connect.
    assert weights.sum() == 1
    assert weights[masks].tolist() == [0.25] * 4


def test_aupro_missing_region_or_background():
    assert aupro([[[0, 0]]], [[[0, 1]]], fpr_limit=0.3) is None
    assert aupro([[[1, 1]]], [[[0, 1]]], fpr_limit=0.3) is None
    with pytest.raises(ValueError):
        aupro([[0, 1]], [[0, 1]], fpr_limit=0.3)


@pytest.mark.parametrize("limit", [0, -1, 1.1, float("nan"), True])
def test_invalid_aupro_limit(limit):
    with pytest.raises(ValueError):
        aupro([[[1, 0]]], [[[2, 1]]], fpr_limit=limit)


def test_combined_metrics_at_frozen_thresholds_and_null_reasons():
    result = evaluate_predictions(
        [0, 1],
        [0.0, 2.0],
        [[[0, 0]], [[1, 0]]],
        [[[0.0, 0.0]], [[2.0, 0.0]]],
        image_threshold=1.0,
        pixel_threshold=1.0,
        fpr_limit=0.3,
    )
    for key in ("image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr", "iou", "dice", "aupro"):
        assert result[key] == 1
    assert result["undefined"] == {}
    result = evaluate_predictions(
        [0], [0.0], [[[0]]], [[[0.0]]], image_threshold=0.0, pixel_threshold=0.0, fpr_limit=0.3
    )
    assert result["image_auroc"] is None
    assert "image_auroc" in result["undefined"]


def test_wrong_image_pixel_batch_alignment():
    with pytest.raises(ValueError):
        evaluate_predictions(
            [0],
            [0.0],
            [[[0]], [[0]]],
            [[[0.0]], [[0.0]]],
            image_threshold=0.0,
            pixel_threshold=0.0,
            fpr_limit=0.3,
        )


def test_rank_metrics_match_independent_sklearn_oracle():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(42)
    for _ in range(50):
        labels = np.r_[0, 1, rng.integers(0, 2, 30)]
        scores = rng.integers(0, 5, 32).astype(np.float32)
        actual = ranking_metrics(labels, scores)
        assert actual["auroc"] == pytest.approx(metrics.roc_auc_score(labels, scores))
        assert actual["aupr"] == pytest.approx(metrics.average_precision_score(labels, scores))


def test_large_integer_scores_cannot_silently_become_ties():
    scores = np.array([2**53, 2**53 + 1], dtype=np.uint64)
    with pytest.raises(ValueError, match="lossless"):
        ranking_metrics([0, 1], scores)
    with pytest.raises(ValueError, match="lossless"):
        calibrate_normal(scores, target_fpr=0.05)


def test_protocol_defaults_hash_and_invalid_overrides():
    config = load_evaluation_config()
    assert config.categories == ("bottle", "leather", "metal_nut")
    assert config.models == ("autoencoder", "patchcore")
    assert config.target_image_fpr == 0.05
    assert config.target_pixel_fpr == 0.01
    assert len(config.config_hash) == 64
    assert replace(config, target_image_fpr=0.1).config_hash != config.config_hash
    for values in (
        {"image_size": 33},
        {"seed": True},
        {"max_pixels": 0},
        {"warmup_images": 0},
        {"target_pixel_fpr": np.nan},
        {"aupro_fpr_limit": 0},
        {"categories": ("unknown",)},
        {"categories": ("bottle", "bottle")},
        {"models": ("autoencoder",)},
    ):
        with pytest.raises(ValueError):
            replace(config, **values)
