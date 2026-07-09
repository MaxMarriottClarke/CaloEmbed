"""Generic train/eval epoch. Knows only model(data) -> embeddings and
loss_fn(embeddings, data) -> scalar; everything else lives in configs."""

import torch
from tqdm import tqdm


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None,
              amp: bool = True, grad_clip=None, desc: str = ""):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    amp = amp and device.type == "cuda"

    total, n_batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp):
                embeddings = model(data)
                loss = loss_fn(embeddings, data)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            total += float(loss.detach())
            n_batches += 1

    return total / max(1, n_batches)
