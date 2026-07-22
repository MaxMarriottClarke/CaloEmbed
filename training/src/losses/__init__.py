"""Loss registry.

Add a loss by decorating an nn.Module with @register_loss("name") in a new
module here, then importing that module at the bottom of this file. The loss
is selected in the config via loss.name; remaining loss.* keys are passed as
constructor kwargs.

Contract: forward(embeddings: (N, D), data: Batch) -> scalar. data carries
whatever truth the loss needs (.y shower ids, .frac argmax fractions, .batch).
"""

_LOSSES: dict[str, type] = {}


def register_loss(name: str):
    def decorator(cls):
        if name in _LOSSES:
            raise KeyError(f"Loss '{name}' already registered.")
        _LOSSES[name] = cls
        return cls
    return decorator


def build_loss(cfg: dict):
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(_LOSSES)}")
    return _LOSSES[name](**cfg)


from . import infonce  # noqa: E402,F401  (registers "infonce")
from . import margin  # noqa: E402,F401  (registers "discriminative")
