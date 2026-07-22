"""Discriminative margin loss: hinge behaviour, energy/purity weighting,
per-event independence, and gradient safety at zero distance."""

import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.losses import build_loss
from src.losses.margin import Discriminative

CFG = {"name": "discriminative", "delta_v": 0.5, "delta_d": 1.5,
       "reg_weight": 0.0, "purity_p0": 0.5, "purity_floor": 0.1}


def _event(z, y, frac=None, energy=None):
    n = z.shape[0]
    return Data(x=z, y=torch.tensor(y, dtype=torch.long),
                frac=torch.ones(n) if frac is None else torch.tensor(frac),
                energy=torch.ones(n) if energy is None else torch.tensor(energy),
                num_nodes=n)


def _loss(events, cfg=None):
    batch = Batch.from_data_list(events)
    return build_loss({**CFG, **(cfg or {})})(batch.x, batch)


def test_collation_keeps_per_node_truth_aligned():
    a = _event(torch.zeros(3, 3), [0, 0, 1], frac=[1.0, 0.5, 1.0], energy=[2.0, 3.0, 4.0])
    b = _event(torch.zeros(2, 3), [0, 1], frac=[1.0, 1.0], energy=[5.0, 6.0])
    batch = Batch.from_data_list([a, b])
    assert torch.equal(batch.energy, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(batch.frac, torch.tensor([1.0, 0.5, 1.0, 1.0, 1.0]))
    assert torch.equal(batch.y, torch.tensor([0, 0, 1, 0, 1]))
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1]))


def test_satisfied_margins_give_zero_loss():
    """Two blobs of radius < delta_v, centers 4 apart (> 2*delta_d = 3)."""
    z = torch.tensor([[-2.0, 0.3, 0.0], [-2.0, -0.3, 0.0],
                      [2.0, 0.3, 0.0], [2.0, -0.3, 0.0]])
    assert _loss([_event(z, [0, 0, 1, 1])]).item() == pytest.approx(0.0, abs=1e-6)


def test_spread_beyond_delta_v_is_penalized():
    """One cluster, points at +-1.0 from the centroid: relu(1.0 - 0.5)^2 = 0.25."""
    z = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _loss([_event(z, [0, 0])]).item() == pytest.approx(0.25, abs=1e-5)


def test_centers_closer_than_2_delta_d_are_penalized():
    """Two point-clusters 1.0 apart: relu(3.0 - 1.0)^2 = 4, both ordered pairs."""
    z = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _loss([_event(z, [0, 1])]).item() == pytest.approx(4.0, abs=1e-5)


def test_single_cluster_has_no_push_term():
    z = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert _loss([_event(z, [0, 0])]).item() == pytest.approx(0.0, abs=1e-6)


def test_regularizer_pulls_centers_to_origin():
    """Isolated, tight cluster at |mu| = 3: only the reg term survives."""
    z = torch.tensor([[3.0, 0.0, 0.0]])
    loss = _loss([_event(z, [0])], cfg={"reg_weight": 1.0})
    assert loss.item() == pytest.approx(3.0, abs=1e-5)


def test_purity_weight_ramp():
    loss_fn = Discriminative(purity_p0=0.5, purity_floor=0.1)
    w = loss_fn.purity_weight(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    # floored below p0 (a 50/50 shared hit keeps a small pull, not zero)
    assert w[0].item() == pytest.approx(0.1)
    assert w[2].item() == pytest.approx(0.1)
    assert w[3].item() == pytest.approx(0.55)
    assert w[4].item() == pytest.approx(1.0)


def test_weighting_moves_the_centroid_to_the_energetic_hits():
    """Unweighted, the centroid of [0, 0, 6] is 2; with the third hit carrying
    little weight it sits near 0, so the loss changes."""
    z = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    equal = _loss([_event(z, [0, 0, 0])])
    weighted = _loss([_event(z, [0, 0, 0], energy=[100.0, 100.0, 1.0])])
    assert weighted.item() < equal.item()
    # centroid ~ 6*1/201 = 0.03; the far hit dominates the (weight-normalized) sum
    expected = 1.0 * (6.0 - 6.0 / 201.0 - 0.5) ** 2 / 201.0
    assert weighted.item() == pytest.approx(expected, rel=1e-3)


def test_shared_hit_is_downweighted_but_not_ignored():
    """A p = 0.5 hit must still contribute: zeroing it would let it float into
    the density valley between two showers."""
    z = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pure = _loss([_event(z, [0, 0], frac=[1.0, 1.0])])
    shared = _loss([_event(z, [0, 0], frac=[1.0, 0.5])])
    zeroed = _loss([_event(z, [0, 0], frac=[1.0, 0.5])], cfg={"purity_floor": 0.0})
    assert 0.0 < shared.item() < pure.item()
    assert zeroed.item() == pytest.approx(0.0, abs=1e-6)


def test_labels_are_event_local():
    """Both events reuse labels 0/1 at different places in embedding space. If
    the loss pooled them, cluster 0 would span both and blow up L_var."""
    near = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    far = near + 100.0
    per_event = torch.stack([_loss([_event(near, [0, 1])]), _loss([_event(far, [0, 1])])])
    assert _loss([_event(near, [0, 1]), _event(far, [0, 1])]).item() == \
        pytest.approx(per_event.mean().item(), abs=1e-5)


def test_batch_loss_is_the_mean_over_events():
    a = _event(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), [0, 1])
    b = _event(torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]), [0, 1])
    expected = 0.5 * (_loss([a]).item() + _loss([b]).item())
    assert _loss([a, b]).item() == pytest.approx(expected, abs=1e-5)


def test_non_contiguous_labels_are_handled():
    """Truth CP indices are event-local but need not be 0..C-1 after argmax."""
    z = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _loss([_event(z, [3, 7])]).item() == pytest.approx(4.0, abs=1e-5)


def test_gradients_finite_for_singleton_clusters():
    """A one-node cluster sits exactly on its own centroid, where a naive norm
    has a NaN gradient."""
    z = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], requires_grad=True)
    batch = Batch.from_data_list([_event(z, [0, 1])])
    build_loss(CFG)(z, batch).backward()
    assert torch.isfinite(z.grad).all()


def test_var_gradient_contracts_a_loose_cluster():
    """One cluster spread beyond delta_v: both nodes move toward the centroid."""
    z = torch.tensor([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    batch = Batch.from_data_list([_event(z, [0, 0])])
    build_loss(CFG)(z, batch).backward()
    step = z - 0.01 * z.grad
    assert step[0, 0].item() > -2.0
    assert step[1, 0].item() < 0.0


def test_dist_gradient_separates_close_clusters():
    """Two point-clusters inside 2*delta_d: L_var is zero, so only the push acts."""
    z = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], requires_grad=True)
    batch = Batch.from_data_list([_event(z, [0, 1])])
    build_loss(CFG)(z, batch).backward()
    step = z - 0.01 * z.grad
    assert step[0, 0].item() < -0.5
    assert step[1, 0].item() > 0.5


def test_missing_energy_raises_clear_error():
    z = torch.zeros(2, 3)
    data = Batch.from_data_list([Data(x=z, y=torch.tensor([0, 1]),
                                      frac=torch.ones(2), num_nodes=2)])
    with pytest.raises(AttributeError, match="data.energy"):
        build_loss(CFG)(z, data)


def test_realistic_magnitudes_are_finite():
    torch.manual_seed(0)
    events = []
    for n_cp in (2, 5, 10):
        n = 400
        y = torch.randint(0, n_cp, (n,))
        events.append(_event(torch.randn(n, 3) * 2.0, y.tolist(),
                             frac=torch.where(torch.rand(n) < 0.17, 0.5, 1.0).tolist(),
                             energy=torch.rand(n).mul(10).tolist()))
    loss = _loss(events, cfg={"reg_weight": 1e-3})
    assert torch.isfinite(loss) and loss.item() > 0
    assert not math.isnan(loss.item())
