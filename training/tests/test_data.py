"""Data pipeline: synthetic-HDF5 loading, normalization, event boundaries,
splitting, and the full dataset → DataLoader → model → loss → backward chain."""

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data import DEFAULT_FEATURES, HDF5Events, split_files
from src.losses import build_loss
from src.models import build_model

REAL_DATA_DIR = "/vols/cms/mm1221/cms/Data/d5/100k_mix/hdf5"


def make_h5(path, event_sizes, n_cp=3, seed=0):
    """Write a file matching the caloembed preprocess layout. Each event's
    node values are offset by 1000*event_index so leakage across event
    boundaries is detectable. Returns the raw per-feature arrays."""
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(event_sizes)]).astype(np.int64)
    n = int(offsets[-1])
    ev_of_node = np.repeat(np.arange(len(event_sizes)), event_sizes)
    raw = {}
    with h5py.File(path, "w") as f:
        for name in DEFAULT_FEATURES:
            v = (rng.random(n) + 1000.0 * ev_of_node).astype(np.float32)
            raw[name] = v
            f[f"lc/{name}"] = v
        f["lc/offsets"] = offsets
        f["truth/argmax_cp_idx"] = rng.integers(0, n_cp, n).astype(np.int32)
        f["truth/argmax_fraction"] = rng.random(n).astype(np.float32)
        f["event/n_cp"] = np.full(len(event_sizes), n_cp, dtype=np.int32)
        raw["argmax_cp_idx"] = f["truth/argmax_cp_idx"][:]
        raw["argmax_fraction"] = f["truth/argmax_fraction"][:]
    return raw, offsets


def test_load_normalization_and_boundaries(tmp_path):
    raw, offsets = make_h5(tmp_path / "a.h5", event_sizes=[5, 8, 3])
    ds = HDF5Events([tmp_path / "a.h5"])
    assert len(ds) == 3
    for ev in range(3):
        s, e = offsets[ev], offsets[ev + 1]
        d = ds.get(ev)
        assert d.x.shape == (e - s, len(DEFAULT_FEATURES))
        # normalization exactly as specified, on exactly this event's nodes
        np.testing.assert_allclose(d.x[:, 0], raw["x"][s:e] / 100.0, rtol=1e-6)
        np.testing.assert_allclose(d.x[:, 2], raw["z"][s:e] / 400.0, rtol=1e-6)
        np.testing.assert_allclose(d.x[:, 3], np.log1p(raw["energy"][s:e]), rtol=1e-6)
        np.testing.assert_allclose(d.x[:, 4], raw["eta"][s:e], rtol=1e-6)
        np.testing.assert_array_equal(d.y, raw["argmax_cp_idx"][s:e])
        np.testing.assert_allclose(d.frac, raw["argmax_fraction"][s:e], rtol=1e-6)
        # raw energy is carried alongside the log1p-normalized feature column
        np.testing.assert_allclose(d.energy, raw["energy"][s:e], rtol=1e-6)


def test_derived_phi_features(tmp_path):
    """sin_phi/cos_phi are derived from the lc/phi column, not a column of
    their own, and leave the default feature set untouched."""
    raw, offsets = make_h5(tmp_path / "a.h5", event_sizes=[6, 4])
    features = ("x", "y", "z", "energy", "eta", "sin_phi", "cos_phi")
    ds = HDF5Events([tmp_path / "a.h5"], features=features)
    assert ds.num_features == 7
    for ev in range(2):
        s, e = offsets[ev], offsets[ev + 1]
        d = ds.get(ev)
        np.testing.assert_allclose(d.x[:, 5], np.sin(raw["phi"][s:e]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d.x[:, 6], np.cos(raw["phi"][s:e]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d.x[:, 5] ** 2 + d.x[:, 6] ** 2, 1.0, atol=1e-5)


def test_multi_file_concatenation(tmp_path):
    make_h5(tmp_path / "a.h5", [4, 6], seed=1)
    make_h5(tmp_path / "b.h5", [7], seed=2)
    ds = HDF5Events([tmp_path / "a.h5", tmp_path / "b.h5"])
    assert len(ds) == 3
    assert [ds.get(i).num_nodes for i in range(3)] == [4, 6, 7]


def test_max_events(tmp_path):
    make_h5(tmp_path / "a.h5", [4, 6, 5])
    ds = HDF5Events([tmp_path / "a.h5"], max_events=2)
    assert len(ds) == 2
    assert [ds.get(i).num_nodes for i in range(2)] == [4, 6]


def test_unknown_feature_rejected(tmp_path):
    make_h5(tmp_path / "a.h5", [4])
    with pytest.raises(ValueError, match="Unknown features"):
        HDF5Events([tmp_path / "a.h5"], features=("x", "nonsense"))


def test_split_stratified_deterministic_disjoint(tmp_path):
    files = []
    for n_cp in (2, 3, 4):
        for i in range(10):
            fp = tmp_path / f"h_{n_cp}_{i}.h5"
            make_h5(fp, [3], n_cp=n_cp, seed=n_cp * 10 + i)
            files.append(fp)
    train, val, test = split_files(files, val_fraction=0.2, test_fraction=0.1, seed=7)
    assert len(train) + len(val) + len(test) == 30
    assert not (set(train) & set(val)) and not (set(train) & set(test)) \
        and not (set(val) & set(test))
    # exact stratification: each n_cp group of 10 → 2 val, 1 test
    def ncp_counts(sub):
        out = {}
        for fp in sub:
            with h5py.File(fp) as f:
                out[int(f["event/n_cp"][0])] = out.get(int(f["event/n_cp"][0]), 0) + 1
        return out
    assert ncp_counts(val) == {2: 2, 3: 2, 4: 2}
    assert ncp_counts(test) == {2: 1, 3: 1, 4: 1}
    # deterministic
    again = split_files(files, val_fraction=0.2, test_fraction=0.1, seed=7)
    assert (train, val, test) == again


def test_end_to_end_batch_model_loss(tmp_path):
    make_h5(tmp_path / "a.h5", [30, 45, 25, 40], seed=3)
    ds = HDF5Events([tmp_path / "a.h5"])
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = build_model({"name": "edgeconv", "in_dim": ds.num_features,
                         "hidden_dim": 16, "num_layers": 2, "out_dim": 4, "k": 4})
    loss_fn = build_loss({"name": "infonce"})

    n_batches = 0
    for data in loader:
        emb = model(data)
        assert emb.shape == (data.num_nodes, 4)
        loss = loss_fn(emb, data)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        n_batches += 1
    assert n_batches == 2
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_end_to_end_transformer_margin(tmp_path):
    """Same chain for the geometric transformer + margin loss: batching (which
    pads to a dense tensor), forward, loss, backward."""
    make_h5(tmp_path / "a.h5", [30, 45, 25, 40], seed=3)
    features = ("x", "y", "z", "energy", "eta", "sin_phi", "cos_phi")
    ds = HDF5Events([tmp_path / "a.h5"], features=features)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = build_model({"name": "geo_transformer", "in_dim": ds.num_features,
                         "hidden_dim": 32, "num_layers": 2, "n_heads": 4,
                         "out_dim": 3, "geom_idx": [0, 1, 2, 4, 5, 6],
                         "bias_hidden": 8, "bias_chunk": 16})
    loss_fn = build_loss({"name": "discriminative"})

    for data in loader:
        emb = model(data)
        assert emb.shape == (data.num_nodes, 3)
        loss = loss_fn(emb, data)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.integration
def test_real_file_loads_and_runs():
    import pathlib
    fp = pathlib.Path(REAL_DATA_DIR) / "histo_0.h5"
    if not fp.exists():
        pytest.skip("real data not available")
    ds = HDF5Events([fp])
    assert len(ds) == 100
    data = next(iter(DataLoader(ds, batch_size=4)))
    model = build_model({"name": "edgeconv", "in_dim": ds.num_features}).eval()
    with torch.no_grad():
        emb = model(data)
    assert emb.shape == (data.num_nodes, 8)
    assert torch.isfinite(emb).all()
    # labels are event-local CP indices: non-negative, small
    assert int(data.y.min()) >= 0 and int(data.y.max()) < 10
