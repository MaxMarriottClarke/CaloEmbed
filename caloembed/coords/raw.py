"""Raw coordinate passthrough — no transformation applied.

Passes (x, y, z) coordinates and energy weights directly to CLUEstering.
This is the baseline against which all embedding methods are compared.
"""

import numpy as np
from caloembed.data.loader import EventData


def transform(event: EventData) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates and weights unchanged.

    Returns:
        coords:  (N, n_features) float32
        weights: (N,) float32
    """
    return event.coords.copy(), event.weights.copy()
