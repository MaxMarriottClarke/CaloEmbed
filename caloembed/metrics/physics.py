"""Per-event physics metrics for CLUE clustering output."""

import numpy as np


def compute_event_metrics(
    cluster_ids: np.ndarray,
    weights: np.ndarray,
    n_truth_cp: int,
    min_lc: int = 3,
) -> dict:
    """Compute basic physics metrics for one event.

    Args:
        cluster_ids: (N,) int32 from CLUE; outliers == -1
        weights:     (N,) float32 LC energies
        n_truth_cp:  number of truth CaloParticles in this event
        min_lc:      minimum LCs to count a trackster as reconstructed

    Returns dict with:
        n_reco:                   tracksters with >= min_lc LCs
        efficiency:               n_reco / n_truth_cp (can exceed 1 if CLUE over-clusters)
        total_lc_energy:          sum of all LC energies
        discarded_energy_fraction: energy not captured in valid tracksters
    """
    total_energy = float(np.sum(weights))

    non_outlier_mask = cluster_ids >= 0
    if not np.any(non_outlier_mask):
        return {
            "n_reco": 0,
            "efficiency": 0.0,
            "total_lc_energy": total_energy,
            "discarded_energy_fraction": 1.0,
        }

    counts = np.bincount(cluster_ids[non_outlier_mask])
    valid_ids = np.where(counts >= min_lc)[0]
    n_reco = len(valid_ids)

    in_valid = np.isin(cluster_ids, valid_ids)
    reco_energy = float(np.sum(weights[in_valid]))
    discarded_fraction = max(0.0, 1.0 - reco_energy / total_energy) if total_energy > 0 else 0.0

    efficiency = n_reco / n_truth_cp if n_truth_cp > 0 else 0.0

    return {
        "n_reco": n_reco,
        "efficiency": efficiency,
        "total_lc_energy": total_energy,
        "discarded_energy_fraction": discarded_fraction,
    }
