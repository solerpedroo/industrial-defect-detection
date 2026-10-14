"""Inference adapters.

Wave 6 loads prepared checkpoints here. Wave 0 keeps the package so routes have
a stable import boundary and still refuse to score images.
"""


def predict(*_args: object, **_kwargs: object) -> None:
    """Score one image. Not available until a model is marked ready."""
    raise NotImplementedError("Inference runtime is implemented in Wave 6.")
