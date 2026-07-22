"""Size-bucketed batch sampler: coverage, batch shapes, padding reduction,
determinism, and that PyG's DataLoader accepts it."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.sampler import BucketedBatchSampler

rng = np.random.default_rng(0)
SIZES = rng.integers(950, 3500, size=1000)


def _padding_waste(batches, sizes):
    """Dense-attention cost (B * S_max^2) over useful work (sum of n_i^2)."""
    padded = sum(len(b) * max(sizes[i] for i in b) ** 2 for b in batches)
    useful = sum(sizes[i] ** 2 for b in batches for i in b)
    return padded / useful


def test_batches_have_the_requested_size():
    s = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, seed=1)
    batches = list(s)
    assert batches and all(len(b) == 4 for b in batches)


def test_every_event_used_at_most_once_per_epoch():
    s = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, seed=1)
    flat = [i for b in s for i in b]
    assert len(flat) == len(set(flat))
    assert set(flat) <= set(range(len(SIZES)))


def test_drop_last_false_covers_everything():
    s = BucketedBatchSampler(SIZES, batch_size=7, pool_batches=3, drop_last=False, seed=1)
    flat = sorted(i for b in s for i in b)
    assert flat == list(range(len(SIZES)))


def test_len_matches_actual_batch_count():
    for bs, pool, drop in [(4, 10, True), (7, 3, True), (7, 3, False), (16, 1, True)]:
        s = BucketedBatchSampler(SIZES, batch_size=bs, pool_batches=pool,
                                 drop_last=drop, seed=2)
        assert len(s) == len(list(s)), f"batch_size={bs} pool={pool} drop_last={drop}"


def test_bucketing_removes_most_padding_waste():
    """The whole point: random batching wastes ~1.9x at batch 4 on this size
    distribution; pooled sorting must bring that near 1.0."""
    order = rng.permutation(len(SIZES))
    random_batches = [order[i:i + 4].tolist() for i in range(0, len(order) - 3, 4)]
    random_waste = _padding_waste(random_batches, SIZES)

    bucketed = list(BucketedBatchSampler(SIZES, batch_size=4, pool_batches=50, seed=3))
    bucketed_waste = _padding_waste(bucketed, SIZES)

    assert random_waste > 1.5, f"test fixture too uniform (waste {random_waste:.2f})"
    assert bucketed_waste < 1.1, f"bucketing left {bucketed_waste:.2f}x waste"
    assert random_waste / bucketed_waste > 1.5


def test_bigger_pools_reduce_waste_monotonically():
    wastes = [_padding_waste(list(BucketedBatchSampler(SIZES, 4, pool, seed=4)), SIZES)
              for pool in (1, 5, 50)]
    assert wastes == sorted(wastes, reverse=True)


def test_epochs_differ_but_are_reproducible():
    s = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, seed=5)
    epoch1, epoch2 = list(s), list(s)
    assert epoch1 != epoch2, "sampler must reshuffle between epochs"

    again = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, seed=5)
    assert list(again) == epoch1, "same seed must reproduce the same epoch"

    different = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, seed=6)
    assert list(different) != epoch1


def test_no_shuffle_is_deterministic_and_size_ordered():
    s = BucketedBatchSampler(SIZES, batch_size=4, pool_batches=10, shuffle=False, seed=7)
    first, second = list(s), list(s)
    assert first == second                      # validation must be reproducible
    assert first[0] == sorted(first[0], key=lambda i: SIZES[i])


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="batch_size"):
        BucketedBatchSampler(SIZES, batch_size=0)
    with pytest.raises(ValueError, match="pool_batches"):
        BucketedBatchSampler(SIZES, batch_size=4, pool_batches=0)


def test_works_as_a_pyg_dataloader_batch_sampler():
    """PyG's DataLoader passes batch_size=1/shuffle=False to torch, which is
    compatible with batch_sampler — pin it, since a version bump could break it."""
    sizes = [5, 40, 12, 33, 7, 25]
    events = [Data(x=torch.randn(n, 3), num_nodes=n) for n in sizes]
    loader = DataLoader(events, batch_sampler=BucketedBatchSampler(
        np.array(sizes), batch_size=2, pool_batches=2, seed=8))

    seen, n_batches = 0, 0
    for batch in loader:
        assert int(batch.batch.max()) + 1 == 2
        seen += batch.num_nodes
        n_batches += 1
    assert n_batches == 3 and seen == sum(sizes)
