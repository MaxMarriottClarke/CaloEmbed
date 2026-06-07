"""Per-event physics metrics for CLUE clustering output."""

import numpy as np


def compute_purity(
    cluster_ids: np.ndarray,
    weights: np.ndarray,
    lc_cp_idx: np.ndarray,
    cp_energies: np.ndarray,
    min_lc: int = 3,
    score_threshold: float = 0.2,
) -> list[dict]:
    """Compute the reco-to-sim purity score for each reconstructed object in one event.

    A reconstructed object is a non-outlier CLUE cluster with more than `min_lc - 1`
    layer clusters (LCs). Each LC is taken to belong fully to the simulated object
    (CaloParticle) that contributed the largest fraction of its energy, i.e.
    `lc_cp_idx`.

    For a reco object R and simulated object S, the reco-to-sim score is

        score(R, S) = sum(E_lc^2 for lc in R, lc not in S) / sum(E_lc^2 for lc in R)

    Since every LC belongs to exactly one S, this is equivalent to

        score(R, S) = 1 - sum(E_lc^2 for lc in R and in S) / sum(E_lc^2 for lc in R)

    so the S that minimises score(R, S) is simply the simulated object that owns the
    largest share of R's energy^2 — found here via a single grouped reduction rather
    than looping over every (R, S) pair. A reco object is "pure" (associated to its
    best-matching simulated object) if that minimal score is below `score_threshold`.

    Args:
        cluster_ids:    (N,) int32 from CLUE; outliers == -1
        weights:        (N,) float32 LC energies
        lc_cp_idx:      (N,) int32 local CaloParticle index each LC is assigned to
        cp_energies:    (n_truth_cp,) float32 raw energy of each CaloParticle
        min_lc:         minimum LCs for a CLUE cluster to count as a reconstructed object
        score_threshold: reco-to-sim score below which a match is considered "pure"

    Returns:
        One dict per reconstructed object with keys:
        reco_id, n_lc, reco_energy, matched_cp_idx, matched_cp_energy, score, is_pure
    """
    energy_sq = weights.astype(np.float64) ** 2
    n_cp = len(cp_energies)

    objects = []
    reco_ids, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    for reco_id, n_lc in zip(reco_ids, counts):
        if n_lc < min_lc:
            continue

        mask = (cluster_ids == reco_id)
        e_sq = energy_sq[mask]
        total_e_sq = e_sq.sum()

        overlap_per_cp = np.bincount(lc_cp_idx[mask], weights=e_sq, minlength=n_cp)
        matched_cp_idx = int(np.argmax(overlap_per_cp))
        score = 1.0 - overlap_per_cp[matched_cp_idx] / total_e_sq

        objects.append({
            "reco_id":          int(reco_id),
            "n_lc":             int(n_lc),
            "reco_energy":      float(weights[mask].sum()),
            "matched_cp_idx":   matched_cp_idx,
            "matched_cp_energy": float(cp_energies[matched_cp_idx]),
            "score":            float(score),
            "is_pure":          bool(score < score_threshold),
        })

    return objects
