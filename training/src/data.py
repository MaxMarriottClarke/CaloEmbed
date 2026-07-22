"""HDF5 → PyTorch Geometric dataset for embedding training.

Loads the flat CSR-style layout produced by caloembed.data.preprocess
(lc/* feature arrays + lc/offsets per file) fully into memory as
normalized torch tensors, so Dataset.get() is a zero-copy slice and the
DataLoader never touches disk during training.

Per-event Data object:
  x       (n_lc, F) float32   normalized node features
  y       (n_lc,)   int64     truth shower id (argmax CaloParticle index, event-local)
  frac    (n_lc,)   float32   energy fraction of the argmax CP (1.0 = unambiguous)
  energy  (n_lc,)   float32   raw (un-normalized) LC energy, for energy-weighted losses
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

# Fixed normalization applied at load time. Chosen from d5/100k_mix stats:
# x,y ~ ±250 cm, |z| ~ 320-515 cm, energy steeply falling (median 0.09, p99 14).
FEATURE_TRANSFORMS = {
    "x":       lambda v: v / 100.0,
    "y":       lambda v: v / 100.0,
    "z":       lambda v: v / 400.0,
    "energy":  lambda v: np.log1p(v),
    "eta":     lambda v: v,
    "phi":     lambda v: v,
    "sin_phi": np.sin,
    "cos_phi": np.cos,
}

# Features read from a differently-named HDF5 column; unlisted names read lc/<name>.
FEATURE_SOURCE = {"sin_phi": "phi", "cos_phi": "phi"}

DEFAULT_FEATURES = ("x", "y", "z", "energy", "eta", "phi")


class HDF5Events(Dataset):
    """In-memory dataset over a list of HDF5 files, one graph per event."""

    def __init__(self, files, features=DEFAULT_FEATURES, max_events: int = -1):
        super().__init__()
        self.features = tuple(features)
        unknown = set(self.features) - set(FEATURE_TRANSFORMS)
        if unknown:
            raise ValueError(f"Unknown features {unknown}. Available: {list(FEATURE_TRANSFORMS)}")

        feat_parts, label_parts, frac_parts, energy_parts, offset_parts = [], [], [], [], []
        n_events = 0
        node_base = 0
        for path in tqdm(files, desc="Loading HDF5", unit="file"):
            with h5py.File(path, "r") as f:
                offsets = f["lc/offsets"][:]
                cols = [
                    FEATURE_TRANSFORMS[name](f[f"lc/{FEATURE_SOURCE.get(name, name)}"][:])
                    .astype(np.float32)
                    for name in self.features
                ]
                labels = f["truth/argmax_cp_idx"][:].astype(np.int64)
                fracs = f["truth/argmax_fraction"][:].astype(np.float32)
                energies = f["lc/energy"][:].astype(np.float32)

            if max_events > 0 and n_events + len(offsets) - 1 > max_events:
                keep = max_events - n_events
                offsets = offsets[: keep + 1]
                n_keep = int(offsets[-1])
                cols = [c[:n_keep] for c in cols]
                labels, fracs, energies = labels[:n_keep], fracs[:n_keep], energies[:n_keep]

            feat_parts.append(np.stack(cols, axis=1))
            label_parts.append(labels)
            frac_parts.append(fracs)
            energy_parts.append(energies)
            offset_parts.append(offsets[1:] + node_base)  # drop leading 0, shift to global
            node_base += int(offsets[-1])
            n_events += len(offsets) - 1
            if max_events > 0 and n_events >= max_events:
                break

        if n_events == 0:
            raise RuntimeError(f"No events loaded from {len(files)} files.")

        self._x = torch.from_numpy(np.concatenate(feat_parts))
        self._y = torch.from_numpy(np.concatenate(label_parts))
        self._frac = torch.from_numpy(np.concatenate(frac_parts))
        self._energy = torch.from_numpy(np.concatenate(energy_parts))
        self._offsets = np.concatenate([[0]] + offset_parts)

    def len(self) -> int:
        return len(self._offsets) - 1

    def get(self, idx: int) -> Data:
        s, e = self._offsets[idx], self._offsets[idx + 1]
        return Data(x=self._x[s:e], y=self._y[s:e], frac=self._frac[s:e],
                    energy=self._energy[s:e])

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def event_sizes(self) -> np.ndarray:
        """Node count per event, for size-bucketed batching (see src.sampler)."""
        return np.diff(self._offsets)


def discover_files(data_dir, max_files: int = -1) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No *.h5 files in {data_dir}")
    if max_files > 0:
        files = files[:max_files]
    return files


def _file_n_cp(path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["event/n_cp"][0])


def split_files(files, val_fraction: float, test_fraction: float = 0.0, seed: int = 0):
    """Deterministic file-level train/val/test split, stratified by particle
    count (each file holds events with a single n_cp, so a plain random split
    can skew the val/test particle-count mix).

    Returns (train, val, test); test is empty when test_fraction == 0.
    """
    groups: dict[int, list] = {}
    for fp in files:
        groups.setdefault(_file_n_cp(fp), []).append(fp)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for n_cp in sorted(groups):
        g = sorted(groups[n_cp])
        rng.shuffle(g)
        n_val = round(len(g) * val_fraction)
        n_test = round(len(g) * test_fraction)
        val += g[:n_val]
        test += g[n_val:n_val + n_test]
        train += g[n_val + n_test:]

    # With few files, per-group rounding can leave a requested split empty —
    # fall back to at least one file (taken from train, deterministically).
    if val_fraction > 0 and not val and train:
        val.append(train.pop())
    if test_fraction > 0 and not test and train:
        test.append(train.pop())
    if not train:
        raise ValueError("Split left no training files.")
    return train, val, test
