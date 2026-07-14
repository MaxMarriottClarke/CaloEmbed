"""Tests for compute_purity in caloembed/metrics/physics.py.

Each test targets one invariant. The reference-formula test is the most
important: it implements the standard CMS reco-to-sim formula directly
(the min form, no algebraic shortcuts) and checks the implementation
produces the same answer.
"""

import numpy as np
import pytest

from caloembed.metrics.physics import (
    compute_purity,
    compute_efficiency,
    compute_number_ratio,
)


def _reference_score(cluster_ids, weights, lc_cp_idx, n_cp, reco_id):
    """Standard CMS reco-to-sim formula, implemented verbatim.

    score(R, S_j) = sum_k[ min((fr_reco_k - fr_cp_k)^2, fr_reco_k^2) * e_k^2 ]
                  / sum_k[ fr_reco_k^2 * e_k^2 ]

    With fr_reco = 1 (each LC in exactly one reco object) and
    fr_cp = 1 if LC belongs to CP j else 0, the min term collapses to:
      0   for LCs inside CP j
      1   for LCs outside CP j

    Returns the minimum score over all CPs (the best match).
    """
    mask = cluster_ids == reco_id
    e_sq = weights[mask].astype(np.float64) ** 2
    cp_for_lc = lc_cp_idx[mask]
    total = e_sq.sum()

    best = np.inf
    for j in range(n_cp):
        in_j = cp_for_lc == j
        numerator = e_sq[~in_j].sum()
        best = min(best, numerator / total)
    return best


# basic correctness

def test_perfect_purity():
    # All LCs in cluster 0 are owned by CP 0.
    # overlap = total_e_sq, so score = 1 - 1 = 0.
    cluster_ids = np.array([0, 0, 0], dtype=np.int32)
    weights     = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    assert len(result["score"]) == 1
    assert result["score"][0] == pytest.approx(0.0)
    assert result["is_pure"][0]
    assert result["matched_cp_idx"][0] == 0


def test_mixed_cluster_score():
    # 4 LCs split between two CPs by energy: [2, 2] to CP0, [1, 1] to CP1.
    # total_e_sq = 4+4+1+1 = 10; CP0 overlap = 8; score = 1 - 8/10 = 0.2.
    cluster_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    weights     = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 5.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    assert len(result["score"]) == 1
    assert result["score"][0] == pytest.approx(0.2)
    assert result["matched_cp_idx"][0] == 0
    assert result["n_lc"][0] == 4
    assert result["reco_energy"][0] == pytest.approx(6.0)


def test_score_threshold_boundary():
    # Verify is_pure uses strict < (not <=).
    # Read back the actual computed score, then set the threshold to that exact
    # float value. score < score is False, so is_pure must be False.
    # This avoids hardcoding 0.2, which is not exactly representable in float64.
    cluster_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    weights     = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 5.0], dtype=np.float32)

    actual_score = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)["score"][0]

    at_threshold = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies,
                                  score_threshold=actual_score)
    assert not at_threshold["is_pure"][0]

    above_threshold = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies,
                                     score_threshold=actual_score + 1e-9)
    assert above_threshold["is_pure"][0]


# ── filtering ─────────────────────────────────────────────────────────────────

def test_outliers_not_treated_as_cluster():
    # cluster_id == -1 marks CLUE outliers. They must not appear as a reco object.
    # Two high-energy outliers sit alongside a valid 3-LC cluster.
    cluster_ids = np.array([-1, -1, 0, 0, 0], dtype=np.int32)
    weights     = np.array([10.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([30.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    assert len(result["score"]) == 1
    assert result["reco_id"][0] == 0
    assert result["n_lc"][0] == 3


def test_small_cluster_below_min_lc_skipped():
    # cluster 0 has 2 LCs (< default min_lc=3) → must be dropped.
    # cluster 1 has 3 LCs → must appear.
    cluster_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    assert len(result["score"]) == 1
    assert result["reco_id"][0] == 1


def test_multiple_clusters_independent():
    # Two valid clusters, each fully owned by its own CP.
    # Their scores must not bleed into each other.
    cluster_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 20.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    assert len(result["score"]) == 2
    # reco_ids are sorted by np.unique, so cluster 0 is index 0, cluster 1 is index 1
    assert result["score"][0] == pytest.approx(0.0)
    assert result["score"][1] == pytest.approx(0.0)


# ── equivalence with the standard formula ────────────────────────────────────

def test_matches_reference_formula():
    # 5 LCs in one cluster spread across 3 CPs — non-trivial split.
    # The reference function implements the full min() form from the paper.
    # If both agree, the 1-minus rearrangement is valid.
    cluster_ids = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    weights     = np.array([1.0, 2.0, 3.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    result = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies, min_lc=1)
    ref = _reference_score(cluster_ids, weights, lc_cp_idx, n_cp=3, reco_id=0)

    assert result["score"][0] == pytest.approx(ref, abs=1e-9)


# ── efficiency ────────────────────────────────────────────────────────────────

def test_efficiency_perfect():
    # All 3 LCs belong to CP 0 and are in cluster 0.
    # Every LC in CP 0 is captured by the matching reco object → efficiency = 1.
    cluster_ids = np.array([0, 0, 0], dtype=np.int32)
    weights     = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    eff    = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)

    assert len(eff["cp_idx"]) == 1
    assert eff["efficiency"][0] == pytest.approx(1.0)
    assert eff["shared_energy"][0] == pytest.approx(eff["total_lc_energy"][0])


def test_efficiency_partial_due_to_outlier():
    # 4 LCs all owned by CP 0. LC 3 is an outlier (not in any cluster).
    # Cluster 0 captures LCs 0-2 (energy 3.0); LC 3 (energy 1.0) is missed.
    # efficiency = 3 / 4 = 0.75.
    cluster_ids = np.array([0, 0, 0, -1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    eff    = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)

    assert eff["efficiency"][0] == pytest.approx(0.75)
    assert eff["total_lc_energy"][0] == pytest.approx(4.0)
    assert eff["shared_energy"][0]   == pytest.approx(3.0)


def test_efficiency_zero_for_unmatched_cp():
    # CP 0's LCs are in cluster 0 (valid). CP 1's LCs are all outliers.
    # Cluster 0 correctly matches CP 0, so CP 1 has no reco pointing to it
    # → CP 1 efficiency = 0.
    cluster_ids = np.array([-1, -1, -1, 0, 0, 0], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([1,   1,   1,   0,   0,   0  ], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    eff    = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)

    assert len(eff["cp_idx"]) == 2
    eff_by_cp = {int(eff["cp_idx"][i]): eff["efficiency"][i] for i in range(2)}
    assert eff_by_cp[0] == pytest.approx(1.0)
    assert eff_by_cp[1] == pytest.approx(0.0)


def test_is_efficient_flag():
    # 4 LCs for CP 0: 3 in cluster 0, 1 outlier → efficiency = 0.75.
    # With default threshold 0.5: is_efficient = True.
    # With threshold 0.8: is_efficient = False.
    cluster_ids = np.array([0, 0, 0, -1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)

    eff_default = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)
    assert eff_default["is_efficient"][0]   # 0.75 >= 0.5

    eff_tight = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity,
                                   efficiency_threshold=0.8)
    assert not eff_tight["is_efficient"][0]  # 0.75 < 0.8


def test_is_efficient_threshold_boundary():
    # Efficiency exactly at the threshold counts as efficient (>= is inclusive).
    # 2 of 4 equal-weight LCs in the matching reco → efficiency = 0.5 exactly.
    cluster_ids = np.array([0, 0, 0, -1, -1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    # efficiency = 3/5 = 0.6; use threshold = 0.6 to test the boundary exactly.
    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    actual_efficiency = compute_efficiency(
        cluster_ids, weights, lc_cp_idx, cp_energies, purity
    )["efficiency"][0]

    at = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity,
                            efficiency_threshold=actual_efficiency)
    assert at["is_efficient"][0]   # at threshold → efficient (>=)

    just_above = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity,
                                    efficiency_threshold=actual_efficiency + 1e-9)
    assert not just_above["is_efficient"][0]   # just above → not efficient


def test_efficiency_zero_when_all_clusters_filtered():
    # All clusters have fewer than min_lc=3 LCs, so purity_result is empty.
    # No reco objects survive → efficiency = 0 for all CPs.
    cluster_ids = np.array([0, 0, 1, 1], dtype=np.int32)   # 2 LCs each, all filtered
    weights     = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    eff    = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)

    assert len(purity["reco_id"]) == 0
    assert np.all(eff["efficiency"] == 0.0)


# ── number ratio ──────────────────────────────────────────────────────────────

def test_number_ratio_basic():
    # 2 valid reco objects out of 5 simulated → ratio = 0.4.
    purity_result = {"reco_id": np.array([0, 3], dtype=np.int32)}
    nr = compute_number_ratio(purity_result, n_cp=5)

    assert nr["n_reco"] == 2
    assert nr["n_sim"]  == 5
    assert nr["ratio"]  == pytest.approx(0.4)


def test_number_ratio_no_valid_reco():
    # All clusters filtered — n_reco = 0.
    cluster_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    nr     = compute_number_ratio(purity, n_cp=2)

    assert nr["n_reco"] == 0
    assert nr["ratio"]  == pytest.approx(0.0)


def test_number_ratio_consistent_with_purity():
    # n_reco from number_ratio must equal the length of purity_result["reco_id"].
    cluster_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    weights     = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 10.0], dtype=np.float32)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    nr     = compute_number_ratio(purity, n_cp=len(cp_energies))

    assert nr["n_reco"] == len(purity["reco_id"])
    assert nr["n_sim"]  == 2
