"""Per-event physics metrics for CLUE clustering output."""

import numpy as np


def compute_purity(
    cluster_ids: np.ndarray,
    weights: np.ndarray,
    lc_cp_idx: np.ndarray,
    cp_energies: np.ndarray,
    min_lc: int = 3,
    score_threshold: float = 0.2,
) -> dict[str, np.ndarray]:
    """Compute the reco-to-sim purity score for each reconstructed object in one event.

    Outlier LCs (cluster_id == -1) and clusters with fewer than min_lc layer
    clusters are excluded from all metrics.

    For a reco object R matched to simulated object S, the reco-to-sim score is:

        score(R, S) = 1 - sum(E_lc^2 for lc in R ∩ S) / sum(E_lc^2 for lc in R)

    The best-matching S minimises this score, equivalent to maximising energy^2
    overlap — found via a single grouped reduction rather than looping over all
    (R, S) pairs.

    Args:
        cluster_ids:     (N,) int32 from CLUE; outliers == -1
        weights:         (N,) float32 LC energies
        lc_cp_idx:       (N,) int32 CaloParticle index each LC is assigned to
        cp_energies:     (n_cp,) float32 raw energy of each CaloParticle
        min_lc:          minimum LCs for a CLUE cluster to be a reconstructed object
        score_threshold: score below which a reco object is considered pure

    Returns:
        Columnar dict, one entry per valid reconstructed object.
        Keys: reco_id, n_lc, reco_energy, matched_cp_idx, matched_cp_energy,
              score, is_pure
    """
    energy_sq = weights.astype(np.float64) ** 2
    n_cp = len(cp_energies)

    out_reco_id           = []
    out_n_lc              = []
    out_reco_energy       = []
    out_matched_cp_idx    = []
    out_matched_cp_energy = []
    out_score             = []
    out_is_pure           = []

    uniq_ids, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    for reco_id, n_lc in zip(uniq_ids, counts):
        if n_lc < min_lc:
            continue

        mask = cluster_ids == reco_id
        e_sq = energy_sq[mask]
        total_e_sq = e_sq.sum()

        overlap_per_cp = np.bincount(lc_cp_idx[mask], weights=e_sq, minlength=n_cp)
        matched_cp_idx = int(np.argmax(overlap_per_cp))
        score = 1.0 - overlap_per_cp[matched_cp_idx] / total_e_sq

        out_reco_id.append(int(reco_id))
        out_n_lc.append(int(n_lc))
        out_reco_energy.append(float(weights[mask].sum()))
        out_matched_cp_idx.append(matched_cp_idx)
        out_matched_cp_energy.append(float(cp_energies[matched_cp_idx]))
        out_score.append(score)
        out_is_pure.append(score < score_threshold)

    return {
        "reco_id":           np.array(out_reco_id,           dtype=np.int32),
        "n_lc":              np.array(out_n_lc,              dtype=np.int32),
        "reco_energy":       np.array(out_reco_energy,       dtype=np.float64),
        "matched_cp_idx":    np.array(out_matched_cp_idx,    dtype=np.int32),
        "matched_cp_energy": np.array(out_matched_cp_energy, dtype=np.float64),
        "score":             np.array(out_score,             dtype=np.float64),
        "is_pure":           np.array(out_is_pure,           dtype=bool),
    }


def compute_efficiency(
    cluster_ids: np.ndarray,
    weights: np.ndarray,
    lc_cp_idx: np.ndarray,
    cp_energies: np.ndarray,
    purity_result: dict[str, np.ndarray],
    efficiency_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute the sim-to-reco efficiency for each simulated object in one event.

    For each CaloParticle (CP) j, efficiency is the fraction of its total LC
    energy that is captured by reconstructed objects pointing to it:

        shared_energy(j) = sum(E_lc for lc in CP_j AND lc in any reco pointing to j)
        efficiency(j)    = shared_energy(j) / total_lc_energy(j)

    A reconstructed object "points to" CP j when j is its best-matching simulated
    object (i.e. j == purity_result["matched_cp_idx"] for that object). Only
    valid reconstructed objects — non-outlier clusters with >= min_lc LCs, as
    already filtered in purity_result — are considered.

    Args:
        cluster_ids:          (N,) int32 from CLUE; outliers == -1
        weights:              (N,) float32 LC energies
        lc_cp_idx:            (N,) int32 CaloParticle index each LC is assigned to
        cp_energies:          (n_cp,) float32 raw energy of each CaloParticle
        purity_result:        output of compute_purity for the same event
        efficiency_threshold: shared/total LC energy at or above which a CP is
                              considered efficiently reconstructed

    Returns:
        Columnar dict, one entry per CaloParticle (all CPs included; efficiency
        is 0 for CPs with no matching reconstructed object).
        Keys: cp_idx, cp_energy, shared_energy, total_lc_energy, efficiency,
              is_efficient
    """
    n_cp = len(cp_energies)
    matched_reco_ids    = purity_result["reco_id"]
    matched_cp_of_reco  = purity_result["matched_cp_idx"]

    out_cp_idx          = []
    out_cp_energy       = []
    out_shared_energy   = []
    out_total_lc_energy = []
    out_efficiency      = []
    out_is_efficient    = []

    for j in range(n_cp):
        in_cp_j = lc_cp_idx == j
        total_lc_energy = float(weights[in_cp_j].sum())

        reco_ids_for_j   = matched_reco_ids[matched_cp_of_reco == j]
        in_matching_reco = np.isin(cluster_ids, reco_ids_for_j)
        shared_energy    = float(weights[in_cp_j & in_matching_reco].sum())

        efficiency = shared_energy / total_lc_energy if total_lc_energy > 0 else 0.0

        out_cp_idx.append(j)
        out_cp_energy.append(float(cp_energies[j]))
        out_shared_energy.append(shared_energy)
        out_total_lc_energy.append(total_lc_energy)
        out_efficiency.append(efficiency)
        out_is_efficient.append(efficiency >= efficiency_threshold)

    return {
        "cp_idx":          np.array(out_cp_idx,          dtype=np.int32),
        "cp_energy":       np.array(out_cp_energy,       dtype=np.float64),
        "shared_energy":   np.array(out_shared_energy,   dtype=np.float64),
        "total_lc_energy": np.array(out_total_lc_energy, dtype=np.float64),
        "efficiency":      np.array(out_efficiency,      dtype=np.float64),
        "is_efficient":    np.array(out_is_efficient,    dtype=bool),
    }


def compute_number_ratio(purity_result: dict[str, np.ndarray], n_cp: int) -> dict:
    """Compute the reco-to-sim number ratio for one event.

    Only reconstructed objects that survived the min_lc and outlier filter
    (i.e. those present in purity_result) are counted.

    Args:
        purity_result: output of compute_purity for the same event
        n_cp:          number of simulated objects (CaloParticles) in the event

    Returns:
        Plain dict with keys: n_reco, n_sim, ratio
    """
    n_reco = int(len(purity_result["reco_id"]))
    return {
        "n_reco": n_reco,
        "n_sim":  n_cp,
        "ratio":  float(n_reco / n_cp) if n_cp > 0 else 0.0,
    }
