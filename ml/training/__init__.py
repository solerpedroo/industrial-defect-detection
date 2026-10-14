"""Offline training entry points. HTTP routes must not call these functions."""


def run_training(*_args: object, **_kwargs: object) -> None:
    """Dispatch a training job. Not available in Wave 0."""
    raise NotImplementedError("Training jobs are implemented in Waves 3 and 4.")
