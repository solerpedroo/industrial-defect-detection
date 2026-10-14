"""Exact raw-score metrics, grouped ties; undefined quantities are None.

Binary targets are 0/1. Scores are finite nonnegative distances, not probabilities.
Pixel masks/maps are matching NHW arrays at the recorded model resolution.
AP is non-interpolated average precision, not trapezoidal precision-recall area.
"""

import numpy as np


def _finite_scores(values) -> np.ndarray:
    values = np.asarray(values)
    if (
        not values.size
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
        or (values < 0).any()
        or values.dtype.kind in "iu"
        and (values > 2**53).any()
    ):
        raise ValueError("Expected finite nonnegative scores with lossless float64 representation.")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(np.float64, copy=False)
    if not np.isfinite(converted).all() or (
        values.dtype.kind == "f"
        and values.dtype.itemsize > 8
        and not np.array_equal(converted.astype(values.dtype), values)
    ):
        raise ValueError("Score conversion to float64 would lose precision.")
    return converted


def binary_inputs(target, score) -> tuple[np.ndarray, np.ndarray]:
    target, score = np.asarray(target), np.asarray(score)
    if (
        target.shape != score.shape
        or not target.size
        or target.dtype.kind not in "biuf"
        or score.dtype.kind not in "iuf"
        or not np.isin(target, [0, 1]).all()
        or not np.isfinite(score).all()
        or (score < 0).any()
    ):
        raise ValueError("Expected matching nonempty binary targets and finite nonnegative scores.")
    return target.astype(bool, copy=False), _finite_scores(score)


def _curve(target: np.ndarray, score: np.ndarray):
    target, score = target.ravel(), score.ravel()
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(order) - 1]
    tp = np.r_[0, np.cumsum(target[order], dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    return tp, fp


def ranking_metrics(target, score) -> dict:
    """Pooled ROC AUC/AP; tied groups count atomically; no threshold tuning."""
    target, score = binary_inputs(target, score)
    positives, negatives = int(target.sum()), int((~target).sum())
    tp, fp = _curve(target, score)
    auroc = float(np.trapezoid(tp / positives, fp / negatives)) if positives and negatives else None
    ap = float(np.sum(np.diff(tp / positives) * tp[1:] / (tp[1:] + fp[1:]))) if positives else None
    return {"auroc": auroc, "aupr": ap}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def confusion_metrics(target, score, threshold: float) -> dict:
    """Strict greater-than decisions; undefined denominator returns None."""
    target, score = binary_inputs(target, score)
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    predicted = score > threshold
    tp, fp = int((predicted & target).sum()), int((predicted & ~target).sum())
    fn, tn = int((~predicted & target).sum()), int((~predicted & ~target).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "iou": _ratio(tp, tp + fp + fn),
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _region_weights(masks: np.ndarray) -> tuple[np.ndarray, int]:
    """Equal region weights with 8-connectivity, independent between images."""
    weights = np.zeros(masks.shape, dtype=np.float64)
    regions = 0
    for index, mask in enumerate(masks):
        seen = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        for y, x in zip(*np.nonzero(mask), strict=True):
            if seen[y, x]:
                continue
            seen[y, x] = True
            pending, component = [(int(y), int(x))], []
            while pending:
                cy, cx = pending.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            pending.append((ny, nx))
            ys, xs = zip(*component, strict=True)
            weights[index, ys, xs] = 1 / len(component)
            regions += 1
    if regions:
        weights /= regions
    return weights, regions


def aupro(masks, maps, *, fpr_limit: float) -> float | None:
    """Normalized region-overlap area on [0,fpr_limit], exact score ties.

    Regions are equally weighted 8-connected foreground components. FPR pools
    all background pixels. Linear interpolation clips the endpoint, preserving
    vertical segments. No regions or no background makes AUPRO undefined.
    """
    masks, maps = binary_inputs(masks, maps)
    if masks.ndim != 3:
        raise ValueError("AUPRO requires matching NHW masks/maps.")
    if type(fpr_limit) not in (int, float) or not 0 < fpr_limit <= 1:
        raise ValueError("AUPRO FPR limit must be in (0,1].")
    weights, count = _region_weights(masks)
    background = int((~masks).sum())
    if not count or not background:
        return None
    score = maps.ravel()
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(order) - 1]
    fpr = np.r_[0.0, np.cumsum((~masks).ravel()[order])[ends] / background]
    pro = np.r_[0.0, np.cumsum(weights.ravel()[order])[ends]]
    stop = np.searchsorted(fpr, fpr_limit, side="right")
    x, y = fpr[:stop], pro[:stop]
    if x[-1] < fpr_limit:
        endpoint = y[-1] + (pro[stop] - y[-1]) * (fpr_limit - x[-1]) / (fpr[stop] - x[-1])
        x, y = np.r_[x, fpr_limit], np.r_[y, endpoint]
    return float(np.clip(np.trapezoid(y, x) / fpr_limit, 0, 1))


def evaluate_predictions(
    labels, scores, masks, maps, *, image_threshold: float, pixel_threshold: float, fpr_limit: float
) -> dict:
    """Image/global-pixel metrics at frozen thresholds; no fitting/calibration."""
    labels, scores = binary_inputs(labels, scores)
    masks, maps = binary_inputs(masks, maps)
    if labels.ndim != 1 or masks.ndim != 3 or len(labels) != len(masks):
        raise ValueError("Expected N image scores and NHW masks/maps.")
    image_rank, pixel_rank = ranking_metrics(labels, scores), ranking_metrics(masks, maps)
    image = confusion_metrics(labels, scores, image_threshold)
    pixel = confusion_metrics(masks, maps, pixel_threshold)
    result = {
        "image_auroc": image_rank["auroc"],
        "image_aupr": image_rank["aupr"],
        **{
            key: image[key]
            for key in (
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_negative_rate",
                "tp",
                "fp",
                "tn",
                "fn",
            )
        },
        "pixel_auroc": pixel_rank["auroc"],
        "pixel_aupr": pixel_rank["aupr"],
        "iou": pixel["iou"],
        "dice": pixel["dice"],
        "pixel_confusion": {key: pixel[key] for key in ("tp", "fp", "tn", "fn")},
        "aupro": aupro(masks, maps, fpr_limit=fpr_limit),
    }
    result["undefined"] = {
        key: "Required class/region or ratio denominator is absent."
        for key, value in result.items()
        if value is None
    }
    return result
