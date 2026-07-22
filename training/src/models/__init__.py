"""Model registry.

Add a model by decorating an nn.Module with @register_model("name") in a new
module here, then importing that module at the bottom of this file. The model
is selected in the config via model.name; remaining model.* keys are passed
as constructor kwargs.

Contract: forward(data: Batch) -> (N_nodes, out_dim) embeddings, where data
has .x (node features) and .batch (event assignment).
"""

_MODELS: dict[str, type] = {}


def register_model(name: str):
    def decorator(cls):
        if name in _MODELS:
            raise KeyError(f"Model '{name}' already registered.")
        _MODELS[name] = cls
        return cls
    return decorator


def build_model(cfg: dict):
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_MODELS)}")
    return _MODELS[name](**cfg)


from . import edgeconv  # noqa: E402,F401  (registers "edgeconv")
from . import geo_transformer  # noqa: E402,F401  (registers "geo_transformer")
