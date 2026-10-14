"""PatchCore adapter.

Wave 4 wraps Anomalib. The backbone stays frozen while the normal-patch memory
is built. This module must not import FastAPI.
"""


def build_memory(*_args: object, **_kwargs: object) -> None:
    """Build a normal feature memory. Not available in Wave 0."""
    raise NotImplementedError("PatchCore memory construction is implemented in Wave 4.")


def predict(*_args: object, **_kwargs: object) -> None:
    """Score one image against the memory. Not available in Wave 0."""
    raise NotImplementedError("PatchCore inference is implemented in Wave 4.")
