"""Model contract: shapes, per-event batching, gradients, architecture."""

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models import build_model

CFG = {"name": "edgeconv", "in_dim": 6, "hidden_dim": 32, "num_layers": 2,
       "out_dim": 8, "k": 4, "dropout": 0.3}


def _event(n, seed):
    g = torch.Generator().manual_seed(seed)
    return Data(x=torch.randn(n, CFG["in_dim"], generator=g), num_nodes=n)


def test_output_shape():
    model = build_model(CFG)
    data = Batch.from_data_list([_event(50, 0), _event(30, 1)])
    out = model(data)
    assert out.shape == (80, CFG["out_dim"])
    assert torch.isfinite(out).all()


def test_batching_does_not_mix_events():
    # In eval mode the model is deterministic; an event's embeddings must be
    # identical whether it is alone or batched with another event — this
    # fails if the kNN graph or message passing crosses event boundaries.
    model = build_model(CFG).eval()
    a, b = _event(40, 2), _event(25, 3)
    with torch.no_grad():
        alone = model(Batch.from_data_list([a]))
        together = model(Batch.from_data_list([a, b]))[:40]
    assert torch.allclose(alone, together, atol=1e-5)


def test_gradients_reach_encoder():
    model = build_model(CFG)
    data = Batch.from_data_list([_event(30, 4)])
    model(data).pow(2).sum().backward()
    first_linear = model.lc_encode[0]
    assert first_linear.weight.grad is not None
    assert torch.isfinite(first_linear.weight.grad).all()
    assert first_linear.weight.grad.abs().sum() > 0


def test_matches_reference_architecture():
    """Layer structure of the geant4sim reference Net at its defaults."""
    model = build_model({"name": "edgeconv", "in_dim": 5})
    # encoder: Linear(5,64) ELU Linear(64,64) ELU
    enc = list(model.lc_encode)
    assert [type(m) for m in enc] == [nn.Linear, nn.ELU, nn.Linear, nn.ELU]
    assert enc[0].in_features == 5 and enc[2].out_features == 64
    # 4 EdgeConv layers, each MLP Linear(128,64) ELU Linear(64,64)
    assert len(model.convs) == 4
    conv_mlp = list(model.convs[0].nn)
    assert conv_mlp[0].in_features == 128 and conv_mlp[2].out_features == 64
    # head: 64 → 64 → 32 → 8 with dropout 0.3
    head = list(model.output)
    assert [m.out_features for m in head if isinstance(m, nn.Linear)] == [64, 32, 8]
    assert all(m.p == 0.3 for m in head if isinstance(m, nn.Dropout))
    assert model.k == 20


def test_dropout_active_in_train_inactive_in_eval():
    model = build_model(CFG)
    data = Batch.from_data_list([_event(30, 5)])
    model.eval()
    with torch.no_grad():
        e1, e2 = model(data), model(data)
    assert torch.equal(e1, e2)
    model.train()
    with torch.no_grad():
        t1, t2 = model(data), model(data)
    assert not torch.equal(t1, t2)
