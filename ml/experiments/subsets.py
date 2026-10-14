"""Normal-training subset draws for experiment C.

Draws use only caller-supplied normal train ids. They never read official test
ids. A 100% draw returns every train id. Smaller draws keep a deterministic
prefix of a seeded ranking.

The current split loader refuses a manifest whose train ids are not the full
complement of the validation holdout, because every normal file must remain
checksummed. These draws are therefore a plan, not a fit manifest, until that
contract is extended. No metric is invented from the plan.
"""

import hashlib
import math

from ml.datasets import SplitManifest


def draw_normal_subset(
    train_ids: tuple[str, ...] | list[str],
    fraction: float,
    seed: int,
    repeat: int,
) -> tuple[str, ...]:
    """Return a stable subset of normal train ids for one fraction and repeat."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    if type(repeat) is not int or repeat < 0:
        raise ValueError("Repeat index must be a nonnegative integer.")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Subset fraction must be in (0, 1].")
    ids = tuple(train_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not isinstance(item, str) or not item for item in ids)
    ):
        raise ValueError("Train ids must be unique nonempty strings.")
    if float(fraction) == 1.0:
        return tuple(sorted(ids))
    count = math.floor(len(ids) * float(fraction))
    if count < 1:
        raise ValueError("Subset fraction leaves no normal training image.")
    ranked = sorted(
        ids,
        key=lambda image_id: (
            hashlib.sha256(f"{seed}:{repeat}:{image_id}".encode()).hexdigest(),
            image_id,
        ),
    )
    return tuple(sorted(ranked[:count]))


def plan_normal_subsets(
    train_ids: tuple[str, ...] | list[str],
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
    repeats: int = 3,
) -> tuple[dict, ...]:
    """List membership for every recorded fraction and repeat. No images are read."""
    plan = []
    for fraction in fractions:
        for repeat in range(repeats):
            chosen = draw_normal_subset(train_ids, float(fraction), seed, repeat)
            plan.append(
                {
                    "fraction": float(fraction),
                    "repeat": repeat,
                    "train_ids": list(chosen),
                    "count": len(chosen),
                }
            )
    return tuple(plan)


def assert_subset_not_loadable(manifest: SplitManifest, subset: tuple[str, ...]) -> str:
    """Explain why this draw cannot replace the fit manifest.

    Validation ids stay on the original manifest. The subset is not written.
    """
    if not set(subset) <= set(manifest.train_ids):
        raise ValueError("Subset must come from the manifest fit ids.")
    if set(subset) & set(manifest.validation_ids):
        raise ValueError("Subset must not include validation ids.")
    if set(subset) != set(manifest.train_ids):
        return (
            "Blocked: the split loader requires the full normal inventory and the "
            "recorded validation partition, so this fit subset cannot be persisted "
            "without a new manifest contract."
        )
    return "Full normal fit set; no subset reduction."
