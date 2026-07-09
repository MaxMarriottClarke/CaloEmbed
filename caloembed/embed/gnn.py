"""Run a trained GNN embedding model over events for clustering.

The model architecture and input-feature normalization are imported from the
`training/` package (added to sys.path) rather than re-implemented here, so an
embedding produced at inference is guaranteed to match how the model was
trained. Only the run directory (config.yaml + best.pt) is needed.

Output embeddings are L2-normalized, matching the InfoNCE training objective
which normalizes before computing cosine similarity.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from caloembed.data.loader import EventData

# Reuse the exact model + feature normalization from the training package.
_TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))
from src.models import build_model  # noqa: E402
from src.data import FEATURE_TRANSFORMS  # noqa: E402


class GNNEmbedder:
    """Load a trained checkpoint and embed events into the latent space."""

    def __init__(self, run_dir, checkpoint: str = "best.pt", device: str | None = None):
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

        self.features = tuple(cfg["data"]["features"])
        self.out_dim = int(cfg["model"]["out_dim"])
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = build_model({**cfg["model"], "in_dim": len(self.features)})
        state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _features(self, event: EventData) -> np.ndarray:
        """Build the normalized (N, F) input array in the training feature order."""
        raw = {
            "x":      event.coords[:, 0],
            "y":      event.coords[:, 1],
            "z":      event.coords[:, 2],
            "energy": event.weights,
            "eta":    event.lc_eta,
            "phi":    event.lc_phi,
        }
        cols = [FEATURE_TRANSFORMS[name](raw[name]).astype(np.float32) for name in self.features]
        return np.stack(cols, axis=1)

    @torch.no_grad()
    def embed(self, event: EventData) -> np.ndarray:
        """Return L2-normalized (N, out_dim) embeddings for one event."""
        x = torch.from_numpy(self._features(event)).to(self.device)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        data = type("G", (), {"x": x, "batch": batch})()
        z = self.model(data)
        z = F.normalize(z.float(), p=2, dim=1)
        return z.cpu().numpy()
