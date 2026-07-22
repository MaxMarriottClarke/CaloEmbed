"""Optimizer/scheduler construction and the gradient-accumulation loop.

These sit in code shared with the GNN run, so each test also pins the
pre-existing default behaviour (adam + per-epoch schedulers, accum_steps=1).
"""

import torch
import torch.nn as nn
import pytest

from src.loop import run_epoch
from train import PER_ITERATION_SCHEDULERS, build_optimizer, build_scheduler


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.norm = nn.LayerNorm(2)

    def forward(self, data):
        return self.norm(self.lin(data.x))


class _Batch:
    """Minimal stand-in for a PyG Batch: run_epoch only calls .to() on it."""

    def __init__(self, n=8):
        self.x = torch.randn(n, 4)

    def to(self, device, **kwargs):
        self.x = self.x.to(device)
        return self


def test_default_optimizer_is_unchanged_adam():
    opt = build_optimizer(_Tiny(), {"lr": 3e-4, "weight_decay": 0.1})
    assert isinstance(opt, torch.optim.Adam) and not isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_adamw_excludes_biases_and_norms_from_decay():
    opt = build_optimizer(_Tiny(), {"optimizer": "adamw", "lr": 3e-4,
                                    "weight_decay": 0.05, "betas": [0.9, 0.95]})
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decay["params"])       # weight matrices only
    assert all(p.ndim <= 1 for p in no_decay["params"])   # biases + norm gains
    assert len(decay["params"]) == 1 and len(no_decay["params"]) == 3
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(_Tiny(), {"optimizer": "lion", "lr": 1e-3})


def test_warmup_cosine_shape():
    opt = build_optimizer(_Tiny(), {"lr": 1.0})
    sched = build_scheduler(opt, {"scheduler": "warmup_cosine", "warmup_fraction": 0.1},
                            epochs=10, steps_per_epoch=10)   # 100 steps, 10 warmup
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(0.1)          # first step, not zero
    assert lrs[9] == pytest.approx(1.0)          # peak at the end of warmup
    assert lrs[:10] == sorted(lrs[:10])          # monotone ramp up
    assert lrs[10:] == sorted(lrs[10:], reverse=True)   # monotone cosine decay
    assert lrs[-1] < 1e-3                        # decayed to ~zero


def test_warmup_cosine_is_per_iteration_and_others_are_not():
    assert "warmup_cosine" in PER_ITERATION_SCHEDULERS
    assert not PER_ITERATION_SCHEDULERS & {"none", "step", "cosine"}


def test_epoch_schedulers_still_build():
    opt = build_optimizer(_Tiny(), {"lr": 1e-3})
    assert build_scheduler(opt, {"scheduler": "none"}, 10) is None
    assert isinstance(build_scheduler(opt, {"scheduler": "cosine"}, 10),
                      torch.optim.lr_scheduler.CosineAnnealingLR)
    assert isinstance(build_scheduler(opt, {"scheduler": "step"}, 10),
                      torch.optim.lr_scheduler.StepLR)


def _count_steps(accum_steps, n_batches=12):
    """Run n_batches through run_epoch, counting optimizer and scheduler steps."""
    torch.manual_seed(0)
    model, device = _Tiny(), torch.device("cpu")
    opt = build_optimizer(model, {"lr": 1e-3})
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    steps = {"opt": 0, "sched": 0}
    real_step = opt.step

    def counting_step(*a, **k):
        steps["opt"] += 1
        return real_step(*a, **k)

    opt.step = counting_step
    sched = type("S", (), {"step": lambda self: steps.__setitem__("sched", steps["sched"] + 1)})()

    batches = [_Batch() for _ in range(n_batches)]
    run_epoch(model, batches, lambda z, d: z.pow(2).mean(), device,
              optimizer=opt, scaler=scaler, amp=False, accum_steps=accum_steps,
              scheduler=sched)
    return steps


def test_accumulation_reduces_optimizer_steps():
    assert _count_steps(accum_steps=1) == {"opt": 12, "sched": 12}
    assert _count_steps(accum_steps=4) == {"opt": 3, "sched": 3}


def test_accumulated_gradient_matches_one_big_batch():
    """Four batches at accum_steps=4 must give the same gradient as one batch
    of all the data — i.e. the 1/accum_steps scaling is right."""
    torch.manual_seed(0)
    batches = [_Batch(8) for _ in range(4)]
    loss_fn = lambda z, d: z.pow(2).mean()

    torch.manual_seed(1)
    model = _Tiny()
    opt = build_optimizer(model, {"lr": 0.0})   # lr 0: inspect grads, don't move
    # Snapshot at the optimizer step — run_epoch zeroes the grads right after.
    accumulated, real_step = [], opt.step
    opt.step = lambda *a, **k: (accumulated.extend(p.grad.clone() for p in model.parameters()),
                                real_step(*a, **k))[1]
    run_epoch(model, batches, loss_fn, torch.device("cpu"), optimizer=opt,
              scaler=torch.amp.GradScaler("cpu", enabled=False), amp=False,
              accum_steps=4)
    assert accumulated, "optimizer never stepped"

    torch.manual_seed(1)
    ref = _Tiny()
    combined = _Batch(0)
    combined.x = torch.cat([b.x for b in batches])
    loss_fn(ref(combined), combined).backward()

    # Tolerance is loose enough for fp32 summation order (four partial backward
    # passes vs one) but far tighter than any 1/accum_steps scaling mistake.
    for a, b in zip(accumulated, ref.parameters()):
        torch.testing.assert_close(a, b.grad, rtol=1e-4, atol=1e-5)


def test_eval_pass_does_not_touch_the_optimizer():
    torch.manual_seed(0)
    model = _Tiny()
    before = [p.clone() for p in model.parameters()]
    loss = run_epoch(model, [_Batch() for _ in range(3)], lambda z, d: z.pow(2).mean(),
                     torch.device("cpu"), amp=False)
    assert loss > 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
