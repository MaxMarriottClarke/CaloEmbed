"""InfoNCE correctness: analytic cases + an independent loop-based reference."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.losses import build_loss
from src.losses.infonce import _event_infonce


def naive_infonce(emb, labels, temperature):
    """Straightforward per-anchor loop implementation of supervised InfoNCE
    (node-equal reduction), written independently of the vectorized version."""
    z = F.normalize(emb.double(), dim=1)
    n = len(labels)
    anchor_losses = []
    for i in range(n):
        positives = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not positives:
            continue
        sims = torch.tensor([
            float(z[i] @ z[j]) / temperature for j in range(n) if j != i
        ])
        denom = torch.logsumexp(sims, dim=0)
        others = [j for j in range(n) if j != i]
        pos_terms = [sims[others.index(j)] - denom for j in positives]
        anchor_losses.append(-sum(pos_terms) / len(pos_terms))
    if not anchor_losses:
        return 0.0
    return float(sum(anchor_losses) / len(anchor_losses))


def _batch(emb_list, label_list):
    return Batch.from_data_list([
        Data(x=e, y=l, num_nodes=len(l)) for e, l in zip(emb_list, label_list)
    ])


def test_matches_naive_reference():
    torch.manual_seed(0)
    emb = torch.randn(30, 8)
    labels = torch.randint(0, 4, (30,))
    loss_fn = build_loss({"name": "infonce", "temperature": 0.1})
    data = _batch([emb], [labels])
    got = float(loss_fn(emb, data))
    want = naive_infonce(emb, labels, 0.1)
    assert got == pytest.approx(want, rel=1e-5)


def test_two_nodes_same_label_is_zero():
    # Only one other node and it is the positive → log softmax prob = 0.
    emb = torch.randn(2, 4)
    data = _batch([emb], [torch.tensor([7, 7])])
    loss_fn = build_loss({"name": "infonce"})
    assert float(loss_fn(emb, data)) == pytest.approx(0.0, abs=1e-6)


def test_no_positives_gives_zero():
    emb = torch.randn(5, 4)
    data = _batch([emb], [torch.arange(5)])  # all labels distinct
    loss_fn = build_loss({"name": "infonce"})
    assert float(loss_fn(emb, data)) == 0.0


def test_analytic_three_nodes():
    # Nodes 0,1 share a label, node 2 differs. Orthogonal unit embeddings
    # except 0 and 1 identical → s(0,1)=1, s(0,2)=s(1,2)=0.
    emb = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    labels = torch.tensor([0, 0, 1])
    t = 0.5
    # anchors 0 and 1: -log(e^{1/t} / (e^{1/t} + e^0)); anchor 2: no positives.
    expected = -math.log(math.exp(1 / t) / (math.exp(1 / t) + 1))
    data = _batch([emb], [labels])
    loss_fn = build_loss({"name": "infonce", "temperature": t})
    assert float(loss_fn(emb, data)) == pytest.approx(expected, rel=1e-5)


def test_batch_is_mean_of_events_and_no_cross_event_pairs():
    torch.manual_seed(1)
    emb_a, lab_a = torch.randn(12, 8), torch.randint(0, 3, (12,))
    emb_b, lab_b = torch.randn(20, 8), torch.randint(0, 3, (20,))
    loss_fn = build_loss({"name": "infonce", "temperature": 0.1})

    la = float(loss_fn(emb_a, _batch([emb_a], [lab_a])))
    lb = float(loss_fn(emb_b, _batch([emb_b], [lab_b])))
    lab = float(loss_fn(torch.cat([emb_a, emb_b]), _batch([emb_a, emb_b], [lab_a, lab_b])))
    # Events must be independent: same labels appear in both events, so any
    # cross-event leakage would change the batched value.
    assert lab == pytest.approx((la + lb) / 2, rel=1e-5)


def test_separated_clusters_beat_shuffled_labels():
    torch.manual_seed(2)
    # Two tight, well-separated clusters.
    emb = torch.cat([
        torch.tensor([10., 0.]) + 0.01 * torch.randn(20, 2),
        torch.tensor([0., 10.]) + 0.01 * torch.randn(20, 2),
    ])
    labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    shuffled = labels[torch.randperm(40)]
    loss_fn = build_loss({"name": "infonce", "temperature": 0.1})
    good = float(loss_fn(emb, _batch([emb], [labels])))
    bad = float(loss_fn(emb, _batch([emb], [shuffled])))
    assert good < bad
    # Supervised InfoNCE floor: with k positives all at similarity 1 and all
    # negatives at 0 (t=0.1, so negatives are negligible), loss → log(k).
    assert good == pytest.approx(math.log(19), rel=1e-3)


def test_gradient_flows():
    emb = torch.randn(10, 4, requires_grad=True)
    labels = torch.randint(0, 2, (10,))
    loss = _event_infonce(F.normalize(emb, dim=1), labels, 0.1)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
