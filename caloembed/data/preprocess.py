"""Convert HGCal ROOT files (ticlDumper/simtrackstersCP) to HDF5.

HDF5 layout per file:

  /lc/
    x, y, z, energy, eta, phi   float32 [N_lc_total]
    offsets                      int64   [N_events + 1]

  /truth/
    argmax_cp_idx    int32   [N_lc_total]    local CP index (within event) with highest fraction
    argmax_fraction  float32 [N_lc_total]    that CP's fraction for this LC

    # Full COO table of all (LC, CP) associations in each event
    coo_lc_idx   int32   [N_edges_total]    local LC index within event
    coo_cp_idx   int32   [N_edges_total]    local CP index within event
    coo_fraction float32 [N_edges_total]    energy fraction = 1 / multiplicity
    coo_offsets  int64   [N_events + 1]     edges for event i: coo_off[i]:coo_off[i+1]

  /event/
    n_lc, n_cp                 int32   [N_events]
    cp_raw_energy              float32 [N_cp_total]   
    cp_pdg_id                  int32   [N_cp_total]
    cp_offsets                 int64   [N_events + 1]
"""

from __future__ import annotations
import subprocess
import time
from pathlib import Path

import numpy as np

def _eta_phi(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Compute (eta, phi) from Cartesian coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    eta = (-0.5 * np.log((r - z) / (r + z))).astype(np.float32)
    phi = np.arctan2(y, x).astype(np.float32)
    return eta, phi

def process_event(tree, event_idx: int) -> dict:
    """Convert one TTree entry into numpy arrays.
    """
    tree.GetEntry(event_idx)

    n_cp: int = tree.NTracksters #only true for cp branch, otherwise number of reco objects
    n_lc: int = tree.NClusters

    lc_x      = np.empty(n_lc, dtype=np.float32)
    lc_y      = np.empty(n_lc, dtype=np.float32)
    lc_z      = np.empty(n_lc, dtype=np.float32)
    lc_energy = np.empty(n_lc, dtype=np.float32)
    lc_filled = np.zeros(n_lc, dtype=bool)   # track which indices we've written

    argmax_cp_idx   = np.full(n_lc, -1, dtype=np.int32)
    argmax_fraction = np.zeros(n_lc, dtype=np.float32)

    coo_lc_parts:  list[np.ndarray] = []
    coo_cp_parts:  list[np.ndarray] = []
    coo_frac_parts: list[np.ndarray] = []

    for cp_i in range(n_cp):
        # Convert ROOT std::vector<T> to numpy, one allocation per CP per feature
        idxs  = np.array(tree.vertices_indexes[cp_i],    dtype=np.int32)
        mults = np.array(tree.vertices_multiplicity[cp_i], dtype=np.float64)  
        xs    = np.array(tree.vertices_x[cp_i],    dtype=np.float32)
        ys    = np.array(tree.vertices_y[cp_i],    dtype=np.float32)
        zs    = np.array(tree.vertices_z[cp_i],    dtype=np.float32)
        es    = np.array(tree.vertices_energy[cp_i], dtype=np.float32)

        fracs = (1.0 / mults).astype(np.float32)

        # Fill LC coordinates 
        new_mask = ~lc_filled[idxs]
        new_idxs = idxs[new_mask]
        lc_x[new_idxs]      = xs[new_mask]
        lc_y[new_idxs]      = ys[new_mask]
        lc_z[new_idxs]      = zs[new_mask]
        lc_energy[new_idxs] = es[new_mask]
        lc_filled[new_idxs] = True

        # Update argmax: keep CP with strictly higher fraction (lower CP index wins ties)
        update_mask = fracs > argmax_fraction[idxs]
        update_idxs = idxs[update_mask]
        argmax_cp_idx[update_idxs]   = cp_i
        argmax_fraction[update_idxs] = fracs[update_mask]

        # Accumulate COO edges
        coo_lc_parts.append(idxs)
        coo_cp_parts.append(np.full(len(idxs), cp_i, dtype=np.int32))
        coo_frac_parts.append(fracs)

    # Drop noise LCs (not covered by any CP): ~0.003% of LCs, coordinates
    covered = np.where(lc_filled)[0] 
    n_covered = len(covered)
    n_unfilled = int(n_lc - n_covered)

    # Remap global LC index to compact index 
    global_to_compact = np.full(n_lc, -1, dtype=np.int32)
    global_to_compact[covered] = np.arange(n_covered, dtype=np.int32)

    # Compact LC feature arrays
    lc_eta, lc_phi = _eta_phi(lc_x[covered], lc_y[covered], lc_z[covered])

    # Build sorted COO with remapped lc indices
    coo_lc   = global_to_compact[np.concatenate(coo_lc_parts)]
    coo_cp   = np.concatenate(coo_cp_parts)
    coo_frac = np.concatenate(coo_frac_parts)
    order = np.lexsort((coo_cp, coo_lc))
    coo_lc   = coo_lc[order]
    coo_cp   = coo_cp[order]
    coo_frac = coo_frac[order]

    return {
        'lc': {
            'x':      lc_x[covered],
            'y':      lc_y[covered],
            'z':      lc_z[covered],
            'energy': lc_energy[covered],
            'eta':    lc_eta,
            'phi':    lc_phi,
        },
        'truth': {
            'argmax_cp_idx':   argmax_cp_idx[covered],
            'argmax_fraction': argmax_fraction[covered],
            'coo_lc_idx':      coo_lc,
            'coo_cp_idx':      coo_cp,
            'coo_fraction':    coo_frac,
        },
        'event': {
            'n_lc':          n_covered,
            'n_cp':          n_cp,
            'n_noise_lc':    n_unfilled,
            'cp_pdg_id':     np.array(tree.pdgID,     dtype=np.int32),
            'cp_raw_energy': np.array(tree.raw_energy, dtype=np.float32),
        },
    }



def convert_file(
    root_path: str | Path,
    hdf5_path: str | Path,
    compression: str = "gzip",
    compression_opts: int = 1,
) -> dict:
    """Convert one ROOT file to one HDF5 file.

    Args:
        root_path:        path to input ROOT file
        hdf5_path:        path to write HDF5 output (parent dir must exist)
        compression:      h5py compression filter ('gzip' or 'lzf')
        compression_opts: compression level (only used for gzip; 1 = fastest)
        verbose:          print progress

    Returns:
        summary dict with event/lc/edge counts and timing
    """
    import ROOT
    import h5py

    root_path = Path(root_path)
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    f = ROOT.TFile.Open(str(root_path))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")
    tree = f.Get("ticlDumper/simtrackstersCP")
    if not tree:
        raise RuntimeError(f"TTree 'ticlDumper/simtrackstersCP' not found in {root_path}")

    n_events = tree.GetEntries()
    events = [process_event(tree, i) for i in range(n_events)]
    f.Close()

    # Build flat arrays + CSR offsets
    n_lc_per_event   = np.array([ev['event']['n_lc']       for ev in events], dtype=np.int32)
    n_cp_per_event   = np.array([ev['event']['n_cp']       for ev in events], dtype=np.int32)
    n_noise_per_event = np.array([ev['event']['n_noise_lc'] for ev in events], dtype=np.int32)
    n_edge_per_event  = np.array([len(ev['truth']['coo_lc_idx']) for ev in events], dtype=np.int32)

    lc_offsets   = np.zeros(n_events + 1, dtype=np.int64)
    cp_offsets   = np.zeros(n_events + 1, dtype=np.int64)
    coo_offsets  = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(n_lc_per_event,   out=lc_offsets[1:])
    np.cumsum(n_cp_per_event,   out=cp_offsets[1:])
    np.cumsum(n_edge_per_event, out=coo_offsets[1:])

    def _cat(key1, key2):
        return np.concatenate([ev[key1][key2] for ev in events])

    lc_x      = _cat('lc', 'x')
    lc_y      = _cat('lc', 'y')
    lc_z      = _cat('lc', 'z')
    lc_energy = _cat('lc', 'energy')
    lc_eta    = _cat('lc', 'eta')
    lc_phi    = _cat('lc', 'phi')

    argmax_cp_idx   = _cat('truth', 'argmax_cp_idx')
    argmax_fraction = _cat('truth', 'argmax_fraction')
    coo_lc_idx      = _cat('truth', 'coo_lc_idx')
    coo_cp_idx      = _cat('truth', 'coo_cp_idx')
    coo_fraction    = _cat('truth', 'coo_fraction')

    cp_raw_energy = _cat('event', 'cp_raw_energy')
    cp_pdg_id     = _cat('event', 'cp_pdg_id')

    gz = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        gz["compression_opts"] = compression_opts

    with h5py.File(hdf5_path, "w") as hf:
        # Metadata attributes
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = "unavailable"

        hf.attrs["source_file"] = str(root_path)
        hf.attrs["n_events"]    = n_events
        hf.attrs["created_at"]  = time.strftime("%Y-%m-%dT%H:%M:%S")
        hf.attrs["git_hash"]    = git_hash

        lc_grp = hf.create_group("lc")
        for name, arr in [("x", lc_x), ("y", lc_y), ("z", lc_z),
                           ("energy", lc_energy), ("eta", lc_eta), ("phi", lc_phi)]:
            lc_grp.create_dataset(name, data=arr, **gz)
        lc_grp.create_dataset("offsets", data=lc_offsets)   # small, no compression

        tr_grp = hf.create_group("truth")
        tr_grp.create_dataset("argmax_cp_idx",   data=argmax_cp_idx,   **gz)
        tr_grp.create_dataset("argmax_fraction",  data=argmax_fraction,  **gz)
        tr_grp.create_dataset("coo_lc_idx",      data=coo_lc_idx,      **gz)
        tr_grp.create_dataset("coo_cp_idx",      data=coo_cp_idx,      **gz)
        tr_grp.create_dataset("coo_fraction",    data=coo_fraction,    **gz)
        tr_grp.create_dataset("coo_offsets",     data=coo_offsets)

        ev_grp = hf.create_group("event")
        ev_grp.create_dataset("n_lc",          data=n_lc_per_event)
        ev_grp.create_dataset("n_cp",          data=n_cp_per_event)
        ev_grp.create_dataset("n_noise_lc",    data=n_noise_per_event)
        ev_grp.create_dataset("cp_raw_energy", data=cp_raw_energy, **gz)
        ev_grp.create_dataset("cp_pdg_id",     data=cp_pdg_id,     **gz)
        ev_grp.create_dataset("cp_offsets",    data=cp_offsets)

    elapsed = time.perf_counter() - t0
    return {
        "root_file":     str(root_path),
        "hdf5_file":     str(hdf5_path),
        "n_events":      n_events,
        "n_lc_total":    int(lc_offsets[-1]),
        "n_cp_total":    int(cp_offsets[-1]),
        "n_edges_total": int(coo_offsets[-1]),
        "n_noise_lc":    int(np.sum(n_noise_per_event)),
        "elapsed_s":     round(elapsed, 2),
    }


