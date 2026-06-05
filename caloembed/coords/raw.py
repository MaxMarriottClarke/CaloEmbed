"""Raw coordinate passthrough — no transformation applied.

Baseline pipeline: passes x,y,z and energy weights directly to CLUEstering.
CLUEstering makes its own contiguous copies internally, so no copy needed here.
"""

import numpy as np
from caloembed.data.loader import EventData


def transform(event: EventData) -> tuple[np.ndarray, np.ndarray]:
    return event.coords, event.weights
