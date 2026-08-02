"""HDF5 event loader for the CaloEmbed pipeline.

Provides EventData (per-event container) and iterators that yield one event
at a time so the full dataset never has to sit in memory simultaneously.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py


@dataclass
class EventData:
    coords: np.ndarray
    weights: np.ndarray
    lc_cp_idx: np.ndarray
    n_truth_cp: int
    cp_energies: np.ndarray
    cp_pdg_ids: np.ndarray
    file_name: str
    event_idx: int
    lc_eta: np.ndarray = None
    lc_phi: np.ndarray = None


def iter_hdf5_events(path, max_events: int = -1):
    """Yield EventData for each event in one HDF5 file."""
    file_name = Path(path).stem
    with h5py.File(path, 'r') as f:
        lc_offsets = f['lc/offsets'][:]
        cp_offsets = f['event/cp_offsets'][:]
        n_events = len(lc_offsets) - 1
        if max_events > 0:
            n_events = min(n_events, max_events)
        lc_x = f['lc/x'][:]
        lc_y = f['lc/y'][:]
        lc_z = f['lc/z'][:]
        lc_e = f['lc/energy'][:]
        lc_eta = f['lc/eta'][:]
        lc_phi = f['lc/phi'][:]
        lc_cp_idx = f['truth/argmax_cp_idx'][:]
        cp_energy = f['event/cp_raw_energy'][:]
        cp_pdg_id = f['event/cp_pdg_id'][:]
        n_cp_arr = f['event/n_cp'][:]

    for ev in range(n_events):
        ls, le = int(lc_offsets[ev]), int(lc_offsets[ev + 1])
        cs, ce = int(cp_offsets[ev]), int(cp_offsets[ev + 1])
        yield EventData(
            coords=np.stack([lc_x[ls:le], lc_y[ls:le], lc_z[ls:le]], axis=1),
            weights=lc_e[ls:le].copy(),
            lc_cp_idx=lc_cp_idx[ls:le].copy(),
            n_truth_cp=int(n_cp_arr[ev]),
            cp_energies=cp_energy[cs:ce].copy(),
            cp_pdg_ids=cp_pdg_id[cs:ce].copy(),
            file_name=file_name,
            event_idx=ev,
            lc_eta=lc_eta[ls:le].copy(),
            lc_phi=lc_phi[ls:le].copy(),
        )


def iter_hdf5_dir(dir_path, max_events: int = -1, max_files: int = -1):
    """Yield EventData across all HDF5 files in a directory."""
    files = sorted(Path(dir_path).glob("*.h5"))
    if max_files > 0:
        files = files[:max_files]
    total = 0
    for fp in files:
        if max_events >= 0 and total >= max_events:
            return
        remaining = -1 if max_events < 0 else max_events - total
        for event in iter_hdf5_events(fp, max_events=remaining):
            yield event
            total += 1
            if max_events >= 0 and total >= max_events:
                return


def iter_hdf5_files(file_paths, max_events: int = -1):
    """Yield EventData across a given list of HDF5 files, in the given order."""
    total = 0
    for fp in file_paths:
        if max_events >= 0 and total >= max_events:
            return
        remaining = -1 if max_events < 0 else max_events - total
        for event in iter_hdf5_events(fp, max_events=remaining):
            yield event
            total += 1
            if max_events >= 0 and total >= max_events:
                return


def select_files_by_n_cp(dir_path, n_events_per_ncp: dict, seed: int = 0, exclude=None):
    """Randomly pick whole HDF5 files to hit target event counts per n_cp bucket.

    Relies on each file in the d5 100k_mix production having a single,
    uniform n_cp value across its ~100 events (true as of this dataset —
    verified by scanning all 1000 files). Only one event per file is read to
    determine its bucket, so this is cheap even over the full directory.
    Requested counts are rounded up to whole files.

    Args:
        exclude: iterable of paths to drop from the pool before sampling. Use
                 this to build a validation set disjoint from a tuning set —
                 a different `seed` alone does not guarantee disjointness.
    """
    files = sorted(Path(dir_path).glob("*.h5"))
    excluded = {Path(p).resolve() for p in (exclude or ())}
    by_ncp: dict[int, list[Path]] = {}
    for fp in files:
        if fp.resolve() in excluded:
            continue
        with h5py.File(fp, "r") as f:
            n_cp = int(f["event/n_cp"][0])
        by_ncp.setdefault(n_cp, []).append(fp)

    rng = np.random.default_rng(seed)
    selected: list[Path] = []
    for n_cp, n_events in n_events_per_ncp.items():
        pool = by_ncp.get(n_cp, [])
        events_per_file = 100
        n_files = -(-n_events // events_per_file)  # ceil
        if n_files > len(pool):
            raise ValueError(
                f"Requested {n_events} events at n_cp={n_cp} needs {n_files} files "
                f"but only {len(pool)} are available in {dir_path}"
            )
        idx = rng.choice(len(pool), size=n_files, replace=False)
        selected.extend(pool[i] for i in idx)

    rng.shuffle(selected)
    return selected
