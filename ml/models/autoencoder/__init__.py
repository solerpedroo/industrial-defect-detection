"""Convolutional autoencoder.

Wave 3 replaces this module with a PyTorch encoder-decoder trained on normal
images with Adam and MSE. Early stopping uses the normal validation split.
"""


def train(*_args: object, **_kwargs: object) -> None:
    """Train the autoencoder. Not available in Wave 0."""
    raise NotImplementedError("Autoencoder training is implemented in Wave 3.")


def reconstruction_error(*_args: object, **_kwargs: object) -> None:
    """Build a reconstruction error map. Not available in Wave 0."""
    raise NotImplementedError("Reconstruction maps are implemented in Wave 3.")
