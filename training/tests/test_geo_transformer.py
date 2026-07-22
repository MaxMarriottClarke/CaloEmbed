"""Geometric transformer contract: shapes, dense-batch padding, geometry bias,
gradients. The padding checks matter most — the model works on a padded dense
tensor, so a bug there silently mixes events or corrupts short ones."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models import build_model
from src.models.geo_transformer import GeometryBias, mask_value

CFG = {"name": "geo_transformer", "in_dim": 7, "hidden_dim": 32, "num_layers": 2,
       "n_heads": 4, "out_dim": 3, "geom_idx": [0, 1, 2, 4, 5, 6], "bias_hidden": 8,
       "bias_chunk": 8, "drop_path": 0.0}


def _event(n, seed):
    g = torch.Generator().manual_seed(seed)
    return Data(x=torch.randn(n, CFG["in_dim"], generator=g), num_nodes=n)


def test_output_shape_and_finite():
    model = build_model(CFG).eval()
    data = Batch.from_data_list([_event(50, 0), _event(30, 1)])
    with torch.no_grad():
        out = model(data)
    assert out.shape == (80, CFG["out_dim"])
    assert torch.isfinite(out).all()


def test_output_is_not_normalized():
    """CLUE needs absolute Euclidean scale — the head must not L2-normalize."""
    model = build_model(CFG).eval()
    with torch.no_grad():
        out = model(Batch.from_data_list([_event(40, 7)]))
    assert not torch.allclose(out.norm(dim=1), torch.ones(40), atol=1e-3)


def test_padding_does_not_change_an_event():
    """An event's embedding must be identical alone and batched with a longer
    event (which pads it). Fails if padded keys leak into attention."""
    model = build_model(CFG).eval()
    short, long = _event(23, 2), _event(64, 3)
    with torch.no_grad():
        alone = model(Batch.from_data_list([short]))
        batched = model(Batch.from_data_list([short, long]))[:23]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_unpacked_rows_follow_batch_order():
    """Row i of the output must belong to node i of the PyG batch."""
    model = build_model(CFG).eval()
    events = [_event(n, 10 + i) for i, n in enumerate((17, 41, 9))]
    with torch.no_grad():
        batched = model(Batch.from_data_list(events))
        alone = torch.cat([model(Batch.from_data_list([e])) for e in events])
    assert torch.allclose(batched, alone, atol=1e-5)


def test_permutation_equivariance():
    model = build_model(CFG).eval()
    ev = _event(30, 4)
    perm = torch.randperm(30, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(Batch.from_data_list([ev]))
        out_perm = model(Batch.from_data_list([Data(x=ev.x[perm], num_nodes=30)]))
    assert torch.allclose(out[perm], out_perm, atol=1e-5)


def test_geometry_bias_shape_and_chunking():
    """Chunked evaluation must equal the unchunked one, and the bias must be
    antisymmetric-in-input: swapping i and j swaps the entry."""
    torch.manual_seed(0)
    geom = torch.randn(2, 21, 6)
    bias = GeometryBias(n_geom=6, n_heads=4, hidden=8, chunk=5).eval()
    with torch.no_grad():
        chunked = bias(geom)
        bias.chunk = 1000
        whole = bias(geom)
    assert chunked.shape == (2, 4, 21, 21)
    assert torch.allclose(chunked, whole, atol=1e-6)


def test_geometry_bias_reaches_attention():
    """Two nodes with identical non-geometric features but different positions
    must get different embeddings — otherwise the bias is doing nothing."""
    model = build_model(CFG).eval()
    x = torch.zeros(2, CFG["in_dim"])
    x[1, 0] = 5.0  # move node 1 in x only
    with torch.no_grad():
        out = model(Batch.from_data_list([Data(x=x, num_nodes=2)]))
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_gradients_flow_to_input_and_bias():
    model = build_model(CFG)
    data = Batch.from_data_list([_event(40, 5), _event(24, 6)])
    model(data).pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    assert model.geom_bias.mlp[0].weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0


def test_checkpointed_bias_matches_eval_path():
    """train() uses checkpointed chunks, eval() does not — same forward value."""
    model = build_model({**CFG, "drop_path": 0.0})
    data = Batch.from_data_list([_event(37, 8)])
    model.train()
    train_out = model(data)
    model.eval()
    with torch.no_grad():
        eval_out = model(data)
    assert torch.allclose(train_out.detach(), eval_out, atol=1e-5)


def test_drop_path_active_only_in_training():
    model = build_model({**CFG, "drop_path": 0.5})
    data = Batch.from_data_list([_event(30, 9), _event(30, 11)])
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(data), model(data))
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        a, b = model(data), model(data)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_forward_is_finite_and_close_to_fp32(dtype):
    """Both autocast dtypes must survive the padded-key mask — a fixed -1e9
    mask overflows fp16 — and stay close to the fp32 result."""
    model = build_model(CFG).eval()
    data = Batch.from_data_list([_event(50, 0), _event(31, 1)])
    with torch.no_grad():
        fp32 = model(data)
        with torch.amp.autocast("cpu", dtype=dtype):
            low = model(data).float()
    assert torch.isfinite(low).all()
    cos = torch.nn.functional.cosine_similarity(low, fp32, dim=1)
    assert cos.median() > 0.99


def test_mask_value_is_representable_in_every_dtype():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        value = torch.tensor(mask_value(dtype), dtype=torch.float64)
        assert torch.isfinite(value.to(dtype))
        # softmax must fully suppress a key at this bias, even against a big logit
        logits = torch.tensor([[100.0, mask_value(dtype)]], dtype=dtype)
        weights = torch.softmax(logits.float(), dim=1)
        assert weights[0, 1].item() == 0.0


def test_bad_geom_idx_rejected():
    with pytest.raises(ValueError, match="geom_idx"):
        build_model({**CFG, "geom_idx": [0, 99]})


def test_heads_must_divide_hidden_dim():
    with pytest.raises(ValueError, match="divisible"):
        build_model({**CFG, "hidden_dim": 30, "n_heads": 4})
