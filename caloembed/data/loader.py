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
