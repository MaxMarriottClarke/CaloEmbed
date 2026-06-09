"""Tests for caloembed/scripts/run_clue3d_baseline.py.

All tests use synthetic numpy arrays — no ROOT dependency.
The ROOT-reading functions (_extract_sim_event, _extract_clue3d_vertices)
are thin wrappers around the same TTree access pattern as preprocess.py
and are covered by the integration test in test_preprocessing.py.
"""

import numpy as np
import pytest

from caloembed.scripts.run_clue3d_baseline import build_cluster_ids
from caloembed.metrics.physics import compute_purity, compute_efficiency



def _identity_mapping(n):
    """global_to_compact where every LC is valid (no noise)."""
    return np.arange(n, dtype=np.int32)


def test_build_cluster_ids_basic():
    # 5 LCs, no noise, 2 tracksters:  trackster 0 → LCs 0,1,2   trackster 1 → LCs 3,4
    g2c = _identity_mapping(5)
    vertex_lists = [np.array([0, 1, 2]), np.array([3, 4])]
    ids = build_cluster_ids(vertex_lists, 5, g2c)
    assert ids.tolist() == [0, 0, 0, 1, 1]


def test_build_cluster_ids_outlier_lcs():
    # LCs 2 and 4 are not in any trackster → should be -1
    g2c = _identity_mapping(5)
    vertex_lists = [np.array([0, 1]), np.array([3])]
    ids = build_cluster_ids(vertex_lists, 5, g2c)
    assert ids[0] == 0
    assert ids[1] == 0
    assert ids[2] == -1
    assert ids[3] == 1
    assert ids[4] == -1


def test_build_cluster_ids_noise_lc_skipped():
    # 6 global LCs but only 5 compact (global index 2 is noise → g2c[2] == -1)
    g2c = np.array([0, 1, -1, 2, 3, 4], dtype=np.int32)
    # trackster 0 contains global LCs 0, 1, 2 — LC 2 is noise and should be dropped
    vertex_lists = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    ids = build_cluster_ids(vertex_lists, 5, g2c)
    assert ids[0] == 0   # global 0 → compact 0
    assert ids[1] == 0   # global 1 → compact 1
    # compact 2 (global 3) belongs to trackster 1
    assert ids[2] == 1
    assert ids[3] == 1   # global 4 → compact 3
    assert ids[4] == 1   # global 5 → compact 4


def test_build_cluster_ids_empty_tracksters():
    # No tracksters → everything is an outlier
    g2c = _identity_mapping(4)
    ids = build_cluster_ids([], 4, g2c)
    assert (ids == -1).all()


def test_build_cluster_ids_single_trackster():
    # One trackster containing all LCs
    g2c = _identity_mapping(3)
    ids = build_cluster_ids([np.array([0, 1, 2])], 3, g2c)
    assert (ids == 0).all()


def test_build_cluster_ids_returns_int32():
    g2c = _identity_mapping(3)
    ids = build_cluster_ids([np.array([0, 1, 2])], 3, g2c)
    assert ids.dtype == np.int32



def test_e2e_perfect_reconstruction():
    # 6 LCs: CP 0 owns LCs 0-2, CP 1 owns LCs 3-5.
    # CLUE3D trackster 0 = {LC 0,1,2}, trackster 1 = {LC 3,4,5} — perfect match.
    weights     = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 20.0], dtype=np.float32)

    g2c = _identity_mapping(6)
    vertex_lists = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    cluster_ids  = build_cluster_ids(vertex_lists, 6, g2c)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    assert len(purity["reco_id"]) == 2
    assert np.allclose(purity["score"], 0.0)
    assert purity["is_pure"].all()

    efficiency = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)
    assert np.allclose(efficiency["efficiency"], 1.0)
    assert efficiency["is_efficient"].all()


def test_e2e_one_trackster_mixed():
    # CLUE3D produces one trackster containing LCs from both CPs — impure.
    weights     = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 1, 1], dtype=np.int32)
    cp_energies = np.array([10.0, 5.0], dtype=np.float32)

    g2c = _identity_mapping(4)
    vertex_lists = [np.array([0, 1, 2, 3])]
    cluster_ids  = build_cluster_ids(vertex_lists, 4, g2c)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    assert len(purity["reco_id"]) == 1
    # total_e_sq = 4+4+1+1=10; CP0 overlap = 8; score = 1 - 8/10 = 0.2
    assert purity["score"][0] == pytest.approx(0.2)
    assert purity["matched_cp_idx"][0] == 0


def test_e2e_trackster_below_min_lc_filtered():
    # CLUE3D trackster has only 2 LCs — below min_lc=3 → not a valid reco object.
    weights     = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    g2c = _identity_mapping(5)
    vertex_lists = [np.array([0, 1])]   # only 2 LCs → filtered
    cluster_ids  = build_cluster_ids(vertex_lists, 5, g2c)

    purity = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies, min_lc=3)
    assert len(purity["reco_id"]) == 0


def test_e2e_noise_lcs_excluded_from_efficiency():
    # 5 global LCs; LC 4 (global) is noise and won't appear in compact space.
    # CLUE3D trackster covers global LCs 0-3; CP 0 owns compact LCs 0-3.
    # The noise LC is not counted in total_lc_energy, so efficiency can still be 1.
    g2c = np.array([0, 1, 2, 3, -1], dtype=np.int32)   # global LC 4 is noise
    weights     = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lc_cp_idx   = np.array([0, 0, 0, 0], dtype=np.int32)
    cp_energies = np.array([10.0], dtype=np.float32)

    vertex_lists = [np.array([0, 1, 2, 3, 4])]  # trackster includes the noise LC
    cluster_ids  = build_cluster_ids(vertex_lists, 4, g2c)

    # Noise LC is silently dropped; the 4 valid LCs are all in trackster 0
    assert (cluster_ids == 0).all()

    purity     = compute_purity(cluster_ids, weights, lc_cp_idx, cp_energies)
    efficiency = compute_efficiency(cluster_ids, weights, lc_cp_idx, cp_energies, purity)
    assert efficiency["efficiency"][0] == pytest.approx(1.0)
