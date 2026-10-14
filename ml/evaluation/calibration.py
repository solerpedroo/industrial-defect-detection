"""Normal-validation empirical quantile calibration; no test input accepted."""

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import _finite_scores


@dataclass(frozen=True)
class FrozenThreshold:
    threshold: float
    target_fpr: float
    quantile: float
    sample_count: int
    observed_fpr: float
    source: str = "normal_validation"
    method: str = "higher"
    comparator: str = ">"

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_normal(values, *, target_fpr: float) -> FrozenThreshold:
    """Higher quantile of finite nonnegative normal values; strict > decisions.

    Exceedance is conservative on this calibration sample only, not a guarantee
    for unseen data. Caller must select held-out normals before any test read.
    Pooled pixel calibration is a pixel FPR, not per-image FPR.
    """
    values = _finite_scores(values)
    if type(target_fpr) not in (int, float) or not 0 < target_fpr < 1:
        raise ValueError("Target FPR must be finite and strictly in (0,1).")
    quantile = 1 - target_fpr
    threshold = float(np.quantile(values.astype(np.float64), quantile, method="higher"))
    return FrozenThreshold(
        threshold,
        float(target_fpr),
        quantile,
        int(values.size),
        float(np.count_nonzero(values > threshold) / values.size),
    )
