"""CLUEstering wrapper with automatic GPU/CPU backend selection.

CLUEstering's C++/Alpaka core is already compiled into backend .so files.
This module selects the best available backend and provides a clean interface
that the rest of the pipeline talks to.

Backend priority (auto mode): gpu cuda > cpu openmp > cpu serial
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import CLUEstering as clue
    _CLUE_AVAILABLE = True
except ImportError:
    _CLUE_AVAILABLE = False
    clue = None


@dataclass
class ClueResult:
    cluster_ids: np.ndarray   # (N,) int32; outliers labelled -1
    n_clusters: int
    n_outliers: int
    elapsed_ms: float         # CLUEstering internal wall-clock timing
    backend: str


def available_backends() -> list[str]:
    if not _CLUE_AVAILABLE:
        return []
    found = ["cpu serial"]
    if clue.is_openmp_available():
        found.append("cpu openmp")
    if clue.is_cuda_available():
        found.append("gpu cuda")
    if clue.is_hip_available():
        found.append("gpu hip")
    return found


def _resolve_backend(requested: str) -> str:
    if not _CLUE_AVAILABLE:
        raise RuntimeError(
            "CLUEstering is not installed. "
            "Run: pip install -e CLUEstering/  (requires cmake + pybind11)"
        )
    if requested != "auto":
        if requested not in available_backends():
            raise ValueError(
                f"Backend '{requested}' not available. Available: {available_backends()}"
            )
        return requested
    # auto-select best
    for backend in ["gpu cuda", "gpu hip", "cpu openmp", "cpu serial"]:
        if backend in available_backends():
            return backend
    return "cpu serial"


def run_clue(
    coords: np.ndarray,
    weights: np.ndarray,
    density_radius: float,
    min_density: float,
    outlier_distance: float | None = None,
    seeding_distance: float | None = None,
    ppbin: int = 128,
    kernel: str = "flat",
    kernel_params: list[float] | None = None,
    backend: str = "auto",
    block_size: int = 1024,
    device_id: int = 0,
) -> ClueResult:
    """Run CLUE clustering.

    Args:
        coords:          (N, n_dim) float32 — point coordinates
        weights:         (N,) float32 — point weights (rechit energies)
        density_radius:  spatial scale for local density calculation
        min_density:     minimum density threshold to become a seed
        outlier_distance: follower-search radius (defaults to density_radius)
        seeding_distance: seed-search radius (defaults to density_radius)
        ppbin:           average points per spatial tile (tunes tile grid)
        kernel:          'flat', 'exp', or 'gaus'
        kernel_params:   kernel parameters (e.g. [0.5] for flat)
        backend:         'auto', 'gpu cuda', 'gpu hip', 'cpu openmp', 'cpu serial'
        block_size:      CUDA thread block size
        device_id:       GPU device index
    """
    if kernel_params is None:
        kernel_params = [0.5]

    selected = _resolve_backend(backend)

    c = clue.clusterer(
        density_radius,
        min_density,
        outlier_distance,   # None → CLUEstering defaults to density_radius
        seeding_distance,
        ppbin,
    )
    c.choose_kernel(kernel, kernel_params)

    # CLUEstering read_data expects: [coord_dim_0, coord_dim_1, ..., weights]
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
