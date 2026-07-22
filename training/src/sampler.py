"""Size-bucketed batching, to stop dense padding dominating attention cost.

A transformer over a PyG batch runs on a dense (B, S, F) tensor where S is the
largest event in the batch, so cost scales with B * S^2 rather than with the
sum of the events' own sizes. With layer-cluster counts spanning ~950-3500,
uniformly random batches waste 1.9x (batch 4) to 2.3x (batch 8) of the compute
on padding — measured on this dataset.

Sorting globally by size would remove the waste but make every batch
homogeneous in event size, which here correlates with n_cp: the gradient would
then see all the 2-particle events together, all the 10-particle ones together.
So sort only *within a shuffled pool* of several batches and shuffle the batch
order afterwards. Padding waste drops to ~1.0x while batch composition stays
close to random.
"""

import numpy as np
from torch.utils.data import Sampler


class BucketedBatchSampler(Sampler):
    """Yield lists of dataset indices grouped by similar event size.

    sizes:        per-event node counts, indexed like the dataset
    pool_batches: batches per sorting pool; larger = less padding, more
                  size-correlation within a batch
    """

    def __init__(self, sizes, batch_size: int, pool_batches: int = 50,
                 shuffle: bool = True, drop_last: bool = True, seed: int = 0):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if pool_batches < 1:
            raise ValueError(f"pool_batches must be >= 1, got {pool_batches}")
        self.sizes = np.asarray(sizes)
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_batches
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def _batches(self):
        n = len(self.sizes)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)

        batches = []
        for start in range(0, n, self.pool_size):
            pool = order[start:start + self.pool_size]
            pool = pool[np.argsort(self.sizes[pool], kind="stable")]
            for i in range(0, len(pool), self.batch_size):
                batch = pool[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch.tolist())

        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        n = len(self.sizes)
        if self.drop_last:
            # whole batches per pool, summed over pools (the last pool is short)
            full, rest = divmod(n, self.pool_size)
            return full * (self.pool_size // self.batch_size) + rest // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
