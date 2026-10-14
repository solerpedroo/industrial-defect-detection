"""Normal-only convolutional autoencoder, trusted reload and reconstruction maps.

Offline fitting is ml.training.autoencoder.train_autoencoder. API serving remains
planned. RGB input uses [0,1]; scores are mean squared reconstruction error.
"""

from .checkpoint import load_autoencoder
from .network import ConvolutionalAutoencoder, reconstruction_error
from .prediction import AutoencoderPredictor, ReconstructionPrediction

__all__ = [
    "AutoencoderPredictor",
    "ConvolutionalAutoencoder",
    "ReconstructionPrediction",
    "load_autoencoder",
    "reconstruction_error",
]
