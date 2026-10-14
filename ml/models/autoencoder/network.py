"""Trainable convolutional autoencoder for normal RGB images.

Input/output: BCHW float32 RGB [0,1], positive spatial dimensions divisible by
eight. Three stride-2 encoder blocks and mirrored decoder blocks restore size.
No pretrained/frozen backbone. Training uses normal fit images only; checkpoint
files belong outside Git. Error maps are channel-mean squared RGB differences,
shape BHW, float32 [0,1]. Image scores aggregate their spatial mean, not severity.
"""

import torch
from torch import nn


def _validate_rgb_batch(batch: torch.Tensor) -> None:
    if (
        not isinstance(batch, torch.Tensor)
        or batch.dtype != torch.float32
        or batch.ndim != 4
        or batch.shape[1] != 3
        or min(batch.shape) <= 0
    ):
        raise ValueError("Expected nonempty BCHW RGB float32 tensors.")
    if not torch.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
        raise ValueError("RGB tensor values must be finite and in [0,1].")


class ConvolutionalAutoencoder(nn.Module):
    """Restore BCHW RGB float32 [0,1] at the input's size, divisible by eight."""

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        if (
            not isinstance(channels, (list, tuple))
            or len(channels) != 3
            or any(type(value) is not int or not 1 <= value <= 256 for value in channels)
        ):
            raise ValueError("channels must contain three integer widths in [1,256].")
        self.channels = tuple(channels)
        encoder: list[nn.Module] = []
        previous = 3
        for width in channels:
            encoder.extend(
                [
                    nn.Conv2d(
                        previous, width, 3, stride=2, padding=1, device="cpu", dtype=torch.float32
                    ),
                    nn.ReLU(),
                ]
            )
            previous = width
        decoder: list[nn.Module] = []
        for index, width in enumerate((*reversed(channels[:-1]), 3)):
            decoder.append(
                nn.ConvTranspose2d(
                    previous, width, 4, stride=2, padding=1, device="cpu", dtype=torch.float32
                )
            )
            decoder.append(nn.Sigmoid() if index == 2 else nn.ReLU())
            previous = width
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _validate_rgb_batch(batch)
        if batch.shape[2] % 8 or batch.shape[3] % 8:
            raise ValueError("Spatial dimensions must be divisible by eight.")
        return self.decoder(self.encoder(batch))


def reconstruction_error(source: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Return BHW float32 channel-mean squared error [0,1] from BCHW RGB [0,1]."""
    _validate_rgb_batch(source)
    _validate_rgb_batch(reconstruction)
    if source.shape != reconstruction.shape or source.device != reconstruction.device:
        raise ValueError("Source and reconstruction must share shape and device.")
    return (source - reconstruction).square().mean(dim=1)
