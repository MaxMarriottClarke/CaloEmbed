"""Generic train/eval epoch. Knows only model(data) -> embeddings and
loss_fn(embeddings, data) -> scalar; everything else lives in configs."""

import torch
from tqdm import tqdm


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None,
              amp: bool = True, amp_dtype=torch.float16, grad_clip=None,
              accum_steps: int = 1, scheduler=None, desc: str = ""):
    """One pass over loader. Trains if optimizer is given, else evaluates.

    accum_steps > 1 accumulates gradients over that many batches per optimizer
    step (an effective batch size of accum_steps * loader.batch_size, for models
    whose per-event memory cost caps the real batch size). scheduler, if given,
    is stepped once per optimizer step rather than once per epoch.
    """
    training = optimizer is not None
    model.train(training)
    amp = amp and device.type == "cuda"

    if training:
        optimizer.zero_grad(set_to_none=True)

    total, n_batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, data in enumerate(tqdm(loader, desc=desc, leave=False)):
            data = data.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp):
                embeddings = model(data)
                loss = loss_fn(embeddings, data)

            if training:
                scaler.scale(loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            total += float(loss.detach())
            n_batches += 1

    return total / max(1, n_batches)
