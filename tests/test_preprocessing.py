"""Tests for the ROOT → HDF5 preprocessing logic.

Synthetic-data tests run without any ROOT file.
Integration tests (marked with @pytest.mark.integration) require the real ROOT file
and are skipped automatically if it is absent.

Key invariants tested:
  1. fraction = 1/multiplicity
  2. argmax_cp_idx is the CP with the highest fraction for each LC
  3. argmax_fraction equals that CP's actual fraction
  4. COO contains every (LC, CP) association, no more, no less
  5. Fraction sums per LC ≈ 1.0 (by TICL construction)
  6. LC coordinates are consistent across CPs that share an LC
  7. CSR offset arrays are monotone and correctly sized
  8. CP pdg_id and raw_energy are preserved in correct order
"""

from __future__ import annotations
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Synthetic data builder ────────────────────────────────────────────────────

class _FakeTree:
    """Mimics the interface of a ROOT TTree entry for simtrackstersCP."""

    def __init__(self, events: list[dict]):
        self._events = events
        self._current: dict = {}

    def GetEntry(self, i: int) -> None:
        self._current = self._events[i]

    def GetEntries(self) -> int:
        return len(self._events)

    @property
    def NTracksters(self):
        return len(self._current["cp"])

    @property
    def NClusters(self):
        return self._current["n_lc"]

    @property
    def pdgID(self):
        return [cp["pdg_id"] for cp in self._current["cp"]]

    @property
    def raw_energy(self):
        return [cp["raw_energy"] for cp in self._current["cp"]]

    @property
    def vertices_indexes(self):
        return [cp["lc_indices"] for cp in self._current["cp"]]

    @property
    def vertices_multiplicity(self):
        return [cp["lc_mults"] for cp in self._current["cp"]]

    @property
    def vertices_x(self):
        return [cp["lc_x"] for cp in self._current["cp"]]

    @property
    def vertices_y(self):
        return [cp["lc_y"] for cp in self._current["cp"]]

    @property
    def vertices_z(self):
        return [cp["lc_z"] for cp in self._current["cp"]]

    @property
    def vertices_energy(self):
        return [cp["lc_energy"] for cp in self._current["cp"]]


def _make_simple_event():
    """
    4 LCs, 2 CPs:
      CP 0: LCs [0, 1]   mults [1.0, 1.5]   → fracs [1.0,  0.667]
      CP 1: LCs [1, 2, 3] mults [3.0, 1.0, 1.0] → fracs [0.333, 1.0, 1.0]

    LC 0: only CP 0 → argmax_cp=0, frac=1.0
    LC 1: CP 0 (0.667) > CP 1 (0.333) → argmax_cp=0, frac=0.667
    LC 2: only CP 1 → argmax_cp=1, frac=1.0
    LC 3: only CP 1 → argmax_cp=1, frac=1.0

    COO sorted by (lc, cp):
      (0, 0, 1.0), (1, 0, 0.667), (1, 1, 0.333), (2, 1, 1.0), (3, 1, 1.0)
    """
    return {
        "n_lc": 4,
        "cp": [
            {
                "pdg_id": 211, "raw_energy": 100.0,
                "lc_indices": [0, 1],
                "lc_mults":   [1.0, 1.5],
                "lc_x":       [1.0, 2.0],
                "lc_y":       [3.0, 4.0],
                "lc_z":       [10.0, 20.0],
                "lc_energy":  [0.5, 0.3],
            },
            {
                "pdg_id": 11, "raw_energy": 50.0,
                "lc_indices": [1, 2, 3],
                "lc_mults":   [3.0, 1.0, 1.0],
                "lc_x":       [2.0, 5.0, 6.0],   # LC 1 must match CP 0's coords
                "lc_y":       [4.0, 7.0, 8.0],
                "lc_z":       [20.0, 30.0, 40.0],
                "lc_energy":  [0.3, 0.2, 0.1],
            },
        ],
    }


def _make_exclusive_event():
    """All LCs exclusively owned by one CP — no sharing."""
    return {
        "n_lc": 3,
        "cp": [
            {
                "pdg_id": 22, "raw_energy": 75.0,
                "lc_indices": [0, 1],
                "lc_mults":   [1.0, 1.0],
                "lc_x": [1.0, 2.0], "lc_y": [1.0, 2.0], "lc_z": [5.0, 10.0],
                "lc_energy": [0.8, 0.6],
            },
            {
                "pdg_id": 22, "raw_energy": 60.0,
                "lc_indices": [2],
                "lc_mults":   [1.0],
                "lc_x": [3.0], "lc_y": [3.0], "lc_z": [15.0],
                "lc_energy": [0.4],
            },
        ],
    }


# ── Tests: fraction and argmax logic ─────────────────────────────────────────

from caloembed.data.preprocess import process_event


def test_fraction_equals_inverse_multiplicity():
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc   = ev["truth"]["coo_lc_idx"]
    coo_cp   = ev["truth"]["coo_cp_idx"]
    coo_frac = ev["truth"]["coo_fraction"]

    # LC 0, CP 0: mult=1.0 → frac=1.0
    mask = (coo_lc == 0) & (coo_cp == 0)
    assert np.allclose(coo_frac[mask], 1.0 / 1.0, atol=1e-5)

    # LC 1, CP 0: mult=1.5 → frac≈0.6667
    mask = (coo_lc == 1) & (coo_cp == 0)
    assert np.allclose(coo_frac[mask], 1.0 / 1.5, atol=1e-5)

    # LC 1, CP 1: mult=3.0 → frac≈0.3333
    mask = (coo_lc == 1) & (coo_cp == 1)
    assert np.allclose(coo_frac[mask], 1.0 / 3.0, atol=1e-5)


def test_argmax_cp_correct():
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    argmax_cp   = ev["truth"]["argmax_cp_idx"]
    argmax_frac = ev["truth"]["argmax_fraction"]

    assert argmax_cp[0] == 0    # LC 0 → CP 0 (only CP)
    assert argmax_cp[1] == 0    # LC 1 → CP 0 (frac 0.667 > 0.333)
    assert argmax_cp[2] == 1    # LC 2 → CP 1 (only CP)
    assert argmax_cp[3] == 1    # LC 3 → CP 1 (only CP)

    assert np.allclose(argmax_frac[0], 1.0,       atol=1e-5)
    assert np.allclose(argmax_frac[1], 1.0 / 1.5, atol=1e-5)
    assert np.allclose(argmax_frac[2], 1.0,        atol=1e-5)


def test_argmax_fraction_is_max_fraction_for_lc():
    """argmax_fraction[lc] must equal max(coo_fraction) over all CPs for that LC."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc   = ev["truth"]["coo_lc_idx"]
    coo_frac = ev["truth"]["coo_fraction"]
    argmax_frac = ev["truth"]["argmax_fraction"]
    n_lc = ev["event"]["n_lc"]

    for lc_i in range(n_lc):
        mask = coo_lc == lc_i
        if not np.any(mask):
            continue
        expected_max = float(np.max(coo_frac[mask]))
        assert np.allclose(argmax_frac[lc_i], expected_max, atol=1e-5), (
            f"LC {lc_i}: argmax_fraction={argmax_frac[lc_i]:.4f}, "
            f"actual max={expected_max:.4f}"
        )


def test_argmax_cp_matches_argmax_fraction():
    """The CP in argmax_cp_idx must indeed hold argmax_fraction for that LC."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc   = ev["truth"]["coo_lc_idx"]
    coo_cp   = ev["truth"]["coo_cp_idx"]
    coo_frac = ev["truth"]["coo_fraction"]
    argmax_cp   = ev["truth"]["argmax_cp_idx"]
    argmax_frac = ev["truth"]["argmax_fraction"]
    n_lc = ev["event"]["n_lc"]

    for lc_i in range(n_lc):
        mask = (coo_lc == lc_i) & (coo_cp == argmax_cp[lc_i])
        assert np.any(mask), f"LC {lc_i}: argmax_cp={argmax_cp[lc_i]} not in COO"
        assert np.allclose(coo_frac[mask][0], argmax_frac[lc_i], atol=1e-5)


def test_fraction_sum_per_lc_near_one():
    """For synthetic data (no noise), fraction sums must be ≈ 1.0.

    In real data, sums can be < 1.0 because some energy is attributed to
    secondary particles / pileup not in the CP list — that is expected and
    not tested here.
    """
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc   = ev["truth"]["coo_lc_idx"]
    coo_frac = ev["truth"]["coo_fraction"]
    n_lc = ev["event"]["n_lc"]

    for lc_i in range(n_lc):
        s = float(np.sum(coo_frac[coo_lc == lc_i]))
        assert abs(s - 1.0) < 0.01, f"LC {lc_i}: fraction sum = {s:.4f}"


def test_fraction_sum_bounded():
    """Fraction sums must always be in (0, 1] — never negative or > 1."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc   = ev["truth"]["coo_lc_idx"]
    coo_frac = ev["truth"]["coo_fraction"]
    n_lc = ev["event"]["n_lc"]

    for lc_i in range(n_lc):
        s = float(np.sum(coo_frac[coo_lc == lc_i]))
        assert 0.0 < s <= 1.0 + 1e-5, f"LC {lc_i}: fraction sum = {s:.4f} out of (0,1]"


def test_coo_completeness():
    """COO must contain exactly one entry per (LC, CP) pair that owns that LC."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc = ev["truth"]["coo_lc_idx"]
    coo_cp = ev["truth"]["coo_cp_idx"]

    pairs = set(zip(coo_lc.tolist(), coo_cp.tolist()))

    # Expected pairs from _make_simple_event
    expected = {(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)}
    assert pairs == expected, f"COO pairs mismatch:\n  got={pairs}\n  expected={expected}"


def test_coo_no_duplicates():
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc = ev["truth"]["coo_lc_idx"]
    coo_cp = ev["truth"]["coo_cp_idx"]
    pairs = list(zip(coo_lc.tolist(), coo_cp.tolist()))
    assert len(pairs) == len(set(pairs)), "Duplicate (lc, cp) pairs in COO"


def test_coo_sorted():
    """COO must be sorted by (lc_idx, cp_idx) for deterministic downstream use."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    coo_lc = ev["truth"]["coo_lc_idx"]
    coo_cp = ev["truth"]["coo_cp_idx"]
    for i in range(len(coo_lc) - 1):
        assert (coo_lc[i], coo_cp[i]) <= (coo_lc[i + 1], coo_cp[i + 1]), (
            f"COO not sorted at position {i}"
        )


def test_exclusive_event_all_argmax_one():
    """When every LC belongs to exactly one CP, argmax_fraction must all be 1.0."""
    tree = _FakeTree([_make_exclusive_event()])
    ev = process_event(tree, 0)
    np.testing.assert_allclose(ev["truth"]["argmax_fraction"], 1.0, atol=1e-5)
    # argmax_cp assignments
    assert ev["truth"]["argmax_cp_idx"][0] == 0
    assert ev["truth"]["argmax_cp_idx"][1] == 0
    assert ev["truth"]["argmax_cp_idx"][2] == 1


def test_lc_coordinates_correct():
    """LC x/y/z/energy arrays must be filled from the correct global LC index."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    # LC 0: only in CP 0 → x=1.0, y=3.0, z=10.0, energy=0.5
    assert ev["lc"]["x"][0]      == pytest.approx(1.0)
    assert ev["lc"]["y"][0]      == pytest.approx(3.0)
    assert ev["lc"]["z"][0]      == pytest.approx(10.0)
    assert ev["lc"]["energy"][0] == pytest.approx(0.5)
    # LC 1: in both CPs with same coords → x=2.0, y=4.0, z=20.0, energy=0.3
    assert ev["lc"]["x"][1]      == pytest.approx(2.0)
    assert ev["lc"]["y"][1]      == pytest.approx(4.0)
    assert ev["lc"]["z"][1]      == pytest.approx(20.0)


def test_eta_phi_computed():
    """eta and phi arrays must be finite and in expected ranges."""
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    eta = ev["lc"]["eta"]
    phi = ev["lc"]["phi"]
    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(phi))
    assert np.all(np.abs(phi) <= math.pi + 1e-5)


def test_cp_metadata_preserved():
    tree = _FakeTree([_make_simple_event()])
    ev = process_event(tree, 0)
    assert ev["event"]["n_cp"] == 2
    assert ev["event"]["n_lc"] == 4
    assert list(ev["event"]["cp_pdg_id"])     == [211, 11]
    assert list(ev["event"]["cp_raw_energy"]) == pytest.approx([100.0, 50.0])


# ── Tests: HDF5 round-trip ────────────────────────────────────────────────────

def test_hdf5_roundtrip_offsets():
    """CSR offsets must be monotone and consistent with n_lc / n_cp / n_edges."""
    pytest.importorskip("h5py")
    from caloembed.data.preprocess import convert_file, read_event

    events = [_make_simple_event(), _make_exclusive_event()]
    tree = _FakeTree(events)

    # Monkey-patch convert_file to use our fake tree
    import caloembed.data.preprocess as pp

    original_fn = pp.convert_file

    def _fake_convert(root_path, hdf5_path, **kwargs):
        import h5py, time, subprocess
        import numpy as np

        n_events = len(events)
        all_ev = [pp.process_event(tree, i) for i in range(n_events)]

        n_lc_per = np.array([e["event"]["n_lc"] for e in all_ev], dtype=np.int32)
        n_cp_per = np.array([e["event"]["n_cp"] for e in all_ev], dtype=np.int32)
        n_edge_per = np.array([len(e["truth"]["coo_lc_idx"]) for e in all_ev], dtype=np.int32)

        lc_off  = np.zeros(n_events + 1, dtype=np.int64)
        cp_off  = np.zeros(n_events + 1, dtype=np.int64)
        coo_off = np.zeros(n_events + 1, dtype=np.int64)
        np.cumsum(n_lc_per,   out=lc_off[1:])
        np.cumsum(n_cp_per,   out=cp_off[1:])
        np.cumsum(n_edge_per, out=coo_off[1:])

        def _cat(k1, k2): return np.concatenate([e[k1][k2] for e in all_ev])

        with h5py.File(hdf5_path, "w") as hf:
            hf.attrs["source_file"] = "synthetic"
            hf.attrs["n_events"] = n_events
            hf.attrs["created_at"] = ""
            hf.attrs["git_hash"] = ""
            lc = hf.create_group("lc")
            for name in ["x", "y", "z", "energy", "eta", "phi"]:
                lc.create_dataset(name, data=_cat("lc", name))
            lc.create_dataset("offsets", data=lc_off)
            tr = hf.create_group("truth")
            for name in ["argmax_cp_idx", "argmax_fraction",
                         "coo_lc_idx", "coo_cp_idx", "coo_fraction"]:
                tr.create_dataset(name, data=_cat("truth", name))
            tr.create_dataset("coo_offsets", data=coo_off)
            ev = hf.create_group("event")
            ev.create_dataset("n_lc", data=n_lc_per)
            ev.create_dataset("n_cp", data=n_cp_per)
            ev.create_dataset("cp_raw_energy", data=_cat("event", "cp_raw_energy"))
            ev.create_dataset("cp_pdg_id",     data=_cat("event", "cp_pdg_id"))
            ev.create_dataset("cp_offsets",    data=cp_off)
        return {"status": "ok"}

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        h5_path = tf.name

    try:
        _fake_convert("dummy.root", h5_path)
        import h5py
        with h5py.File(h5_path, "r") as hf:
            lc_off  = hf["lc/offsets"][:]
            cp_off  = hf["event/cp_offsets"][:]
            coo_off = hf["truth/coo_offsets"][:]

        # Monotone
        assert np.all(np.diff(lc_off)  >= 0)
        assert np.all(np.diff(cp_off)  >= 0)
        assert np.all(np.diff(coo_off) >= 0)

        # Correct sizes: event 0 = simple (4 LC, 2 CP), event 1 = exclusive (3 LC, 2 CP)
        assert lc_off[1] - lc_off[0] == 4
        assert lc_off[2] - lc_off[1] == 3
        assert cp_off[1] - cp_off[0] == 2
        assert cp_off[2] - cp_off[1] == 2

        # COO edge count: simple event has 5 edges (0,1,1,2,3 × CPs)
        assert coo_off[1] - coo_off[0] == 5
        # Exclusive event: 3 edges (one per LC)
        assert coo_off[2] - coo_off[1] == 3

        # Read back event 0 and check argmax
        ev0 = read_event(h5_path, 0)
        assert ev0["event"]["n_lc"] == 4
        assert ev0["event"]["n_cp"] == 2
        assert ev0["truth"]["argmax_cp_idx"][1] == 0   # LC 1 → CP 0
    finally:
        Path(h5_path).unlink(missing_ok=True)


# ── Integration test: real ROOT file ─────────────────────────────────────────

REAL_ROOT = Path("/vols/cms/mm1221/cms/Data/100k/root/histo_921.root")

@pytest.mark.integration
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="Real ROOT file not present")
def test_real_file_fraction_sums():
    """On real data: fraction sums per LC must be in (0, 1].

    Sums < 1.0 are physically expected: some energy is attributed to secondary
    particles / pileup not in the simtrackstersCP list. Sums > 1.0 would
    indicate a bug in the multiplicity inversion.
    """
    import ROOT
    f = ROOT.TFile.Open(str(REAL_ROOT))
    tree = f.Get("ticlDumper/simtrackstersCP")

    for event_idx in [0, 10, 50, 99]:
        ev = process_event(tree, event_idx)
        coo_lc   = ev["truth"]["coo_lc_idx"]
        coo_frac = ev["truth"]["coo_fraction"]
        n_lc = ev["event"]["n_lc"]

        covered_lcs = set(coo_lc.tolist())
        for lc_i in covered_lcs:
            s = float(np.sum(coo_frac[coo_lc == lc_i]))
            assert 0.0 < s <= 1.0 + 1e-4, (
                f"Event {event_idx}, LC {lc_i}: fraction sum = {s:.4f} not in (0,1]"
            )

    f.Close()


@pytest.mark.integration
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="Real ROOT file not present")
def test_real_file_argmax_is_max():
    """On real data: argmax_fraction must equal the max COO fraction per LC."""
    import ROOT
    f = ROOT.TFile.Open(str(REAL_ROOT))
    tree = f.Get("ticlDumper/simtrackstersCP")

    for event_idx in [0, 50]:
        ev = process_event(tree, event_idx)
        coo_lc      = ev["truth"]["coo_lc_idx"]
        coo_frac    = ev["truth"]["coo_fraction"]
        argmax_frac = ev["truth"]["argmax_fraction"]
        n_lc = ev["event"]["n_lc"]

        for lc_i in range(n_lc):
            mask = coo_lc == lc_i
            expected = float(np.max(coo_frac[mask]))
            assert np.allclose(argmax_frac[lc_i], expected, atol=1e-5), (
                f"Event {event_idx}, LC {lc_i}"
            )

    f.Close()


@pytest.mark.integration
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="Real ROOT file not present")
def test_real_file_full_conversion(tmp_path):
    """Convert the real ROOT file and verify basic HDF5 structure."""
    from caloembed.data.preprocess import convert_file, read_event
    h5_path = tmp_path / "histo_921.h5"
    result = convert_file(REAL_ROOT, h5_path, verbose=False)

    assert result["n_events"] == 100
    assert result["n_lc_total"] > 0
    # n_edges >= n_lc - n_noise (noise LCs have no edges)
    assert result["n_edges_total"] >= result["n_lc_total"] - result["n_noise_lc"]

    import h5py
    with h5py.File(h5_path, "r") as hf:
        assert hf.attrs["n_events"] == 100

    for i in [0, 50, 99]:
        ev = read_event(h5_path, i)
        n_lc = ev["event"]["n_lc"]
        n_cp = ev["event"]["n_cp"]
        assert len(ev["lc"]["x"]) == n_lc
        assert len(ev["lc"]["eta"]) == n_lc
        assert len(ev["event"]["cp_pdg_id"]) == n_cp
        assert len(ev["event"]["cp_raw_energy"]) == n_cp
        assert len(ev["truth"]["argmax_cp_idx"]) == n_lc

        # Noise LCs are removed — no LC should have argmax_cp_idx == -1
        assert np.all(ev["truth"]["argmax_cp_idx"] >= 0), "Noise LC found in output"
        assert np.all(ev["truth"]["argmax_cp_idx"] < n_cp)
        assert np.all(ev["truth"]["coo_cp_idx"] < n_cp)
        assert np.all(ev["truth"]["coo_lc_idx"] < n_lc)
        # Fractions must be positive
        assert np.all(ev["truth"]["coo_fraction"] > 0)
        assert np.all(ev["truth"]["coo_fraction"] <= 1.0 + 1e-4)
