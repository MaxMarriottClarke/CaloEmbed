"""CLUEstering wrapper with automatic GPU/CPU backend selection."""

from dataclasses import dataclass
import numpy as np

try:
    import CLUEstering as clue
    _CLUE_AVAILABLE = True
except ImportError:
    _CLUE_AVAILABLE = False
    clue = None


@dataclass(frozen=True)
class ClueResult:
    cluster_ids: np.ndarray
    n_clusters: int
    n_outliers: int
    elapsed_ms: float
    backend: str


def available_backends() -> list[str]:
    if not _CLUE_AVAILABLE:
        return []
    return list(clue.backends)


def _resolve_backend(requested: str) -> str:
    if not _CLUE_AVAILABLE:
        raise RuntimeError(
            "CLUEstering is not installed. "
            "Run: pip install -e CLUEstering/"
        )
    if requested != "auto":
        if requested not in available_backends():
            raise ValueError(
                f"Backend '{requested}' not available. Available: {available_backends()}"
            )
        return requested
    for backend in ["gpu cuda", "gpu hip", "cpu openmp", "cpu serial"]:
        if backend in available_backends():
            return backend
    return "cpu serial"


def run_clue(
    coords: np.ndarray,
    weights: np.ndarray,
    dc: float,
    rhoc: float,
    do: float | None = None,
    dm: float | None = None,
    ppbin: int = 128,
    metric: str = "euclidean",
    metric_params: list[float] | None = None,
    backend: str = "auto",
    block_size: int = 1024,
    device_id: int = 0,
) -> ClueResult:
    """Run CLUE clustering on one event.

    Args:
        coords:        (N, n_dim) float32 
        weights:       (N,) float32
        dc:            critical distance for local density calculation
        rhoc:          density threshold separating seeds from outliers
        do:            nearest-higher search radius; defaults to dc
        dm:            follower connection radius; defaults to dc
        ppbin:         average points per spatial tile
        metric:        'euclidean', 'manhattan', 'chebyshev',
                       'weighted_euclidean', 'weighted_chebyshev', 'periodic_euclidean'
        metric_params: per-dimension weights/periods for parameterised metrics
        backend:       'auto', 'gpu cuda', 'gpu hip', 'cpu openmp', 'cpu serial'
        block_size:    CUDA thread block size
        device_id:     GPU device index
    """
    selected = _resolve_backend(backend)

    c = clue.clusterer(dc, rhoc, do, dm, ppbin)
    c.choose_kernel("flat", [0.5])
    if metric != "euclidean":
        c.choose_metric(metric, metric_params)

    n_dim = coords.shape[1]
    data = [np.ascontiguousarray(coords[:, i], dtype=np.float32) for i in range(n_dim)]
    data.append(np.ascontiguousarray(weights, dtype=np.float32))
    c.read_data(data)

    c.run_clue(backend=selected, block_size=block_size, device_id=device_id)

    ids = c.cluster_ids.copy()
    return ClueResult(
        cluster_ids=ids,
        n_clusters=c.n_clusters,
        n_outliers=int(np.sum(ids == -1)),
        elapsed_ms=c._elapsed_time,
        backend=selected,
    )
