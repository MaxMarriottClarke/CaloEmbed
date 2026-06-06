"""Raw coordinate passthrough, no transformation applied.

Baseline pipeline: passes x,y,z and energy weights directly to CLUEstering.
"""

import numpy as np
from caloembed.data.loader import EventData


def transform(event: EventData) -> tuple[np.ndarray, np.ndarray]:
    return event.coords, event.weights
