"""Main pipeline entry point.

Loads HDF5 events, runs CLUEstering, computes physics metrics,
and saves per-event results to Parquet for later analysis.

Usage:
  caloembed-run --config configs/raw.yaml --output results/raw/
  caloembed-run --config configs/raw.yaml --data /path/to/hdf5/ --max-events 500
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from caloembed.data.loader import iter_hdf5_dir
from caloembed.clustering.clue import run_clue, available_backends
from caloembed.metrics.physics import compute_purity


_TRANSFORMS = {
    "raw": "caloembed.coords.raw",
    # "normalized": "caloembed.coords.normalized",
    # "umap":       "caloembed.coords.umap_embed",
    # "gnn":        "caloembed.coords.gnn_embed",
}


def _get_transform(pipeline: str):
    if pipeline not in _TRANSFORMS:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Available: {list(_TRANSFORMS)}")
    import importlib
    return importlib.import_module(_TRANSFORMS[pipeline]).transform


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description="CaloEmbed clustering pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Override data.dir from config")
    parser.add_argument("--output", help="Override output.dir from config")
    parser.add_argument("--backend", help="Override clustering.backend")
    parser.add_argument("--max-events", type=int, help="Override data.max_events")
    parser.add_argument("--max-files", type=int, help="Override data.max_files")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.data:
        config["data"]["dir"] = args.data
    if args.output:
        config["output"]["dir"] = args.output
    if args.backend:
        config["clustering"]["backend"] = args.backend
    if args.max_events is not None:
        config["data"]["max_events"] = args.max_events
    if args.max_files is not None:
        config["data"]["max_files"] = args.max_files

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    clue_cfg = config["clustering"]
    min_lc = config.get("metrics", {}).get("min_lc", 3)
    score_threshold = config.get("metrics", {}).get("purity_threshold", 0.2)

    transform = _get_transform(config.get("pipeline", "raw"))

    # Extract CLUE params once
    dc           = clue_cfg["dc"]
    rhoc         = clue_cfg["rhoc"]
    do           = clue_cfg.get("do")
    dm           = clue_cfg.get("dm")
    ppbin        = clue_cfg.get("ppbin", 128)
    metric       = clue_cfg.get("metric", "euclidean")
    metric_params = clue_cfg.get("metric_params")
    backend      = clue_cfg.get("backend", "auto")
    block_size   = clue_cfg.get("block_size", 1024)
    device_id    = clue_cfg.get("device_id", 0)

    print(f"Pipeline: {config.get('pipeline', 'raw')}  |  backends: {available_backends()}")

    events = iter_hdf5_dir(
        dir_path=data_cfg["dir"],
        max_events=data_cfg.get("max_events", -1),
        max_files=data_cfg.get("max_files", -1),
    )

    rows = []
    clue_times = []
    t_start = time.perf_counter()

    for i, event in enumerate(events):
        coords, weights = transform(event)

        result = run_clue(
            coords=coords, weights=weights,
            dc=dc, rhoc=rhoc, do=do, dm=dm, ppbin=ppbin,
            metric=metric, metric_params=metric_params,
            backend=backend, block_size=block_size, device_id=device_id,
        )

        objects = compute_purity(
            cluster_ids=result.cluster_ids,
            weights=weights,
            lc_cp_idx=event.lc_cp_idx,
            cp_energies=event.cp_energies,
            min_lc=min_lc,
            score_threshold=score_threshold,
        )

        clue_times.append(result.elapsed_ms)
        for obj in objects:
            rows.append({
                "file_name":   event.file_name,
                "event_idx":   event.event_idx,
                "n_truth_cp":  event.n_truth_cp,
                "clue_ms":     result.elapsed_ms,
                "backend":     result.backend,
                **obj,
            })

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i+1} events  ({elapsed:.1f}s)  last: {result.n_clusters} clusters [{result.backend}]")

    n_events = len(clue_times)
    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {n_events} events in {elapsed:.1f}s ({elapsed / max(n_events, 1) * 1000:.1f} ms/event)")

    df = pd.DataFrame(rows)
    out_path = output_dir / "objects.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Per-object metrics → {out_path}")

    summary = {
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash":        _git_hash(),
        "config":          config,
        "n_events":        n_events,
        "n_objects":       len(rows),
        "mean_purity":     float(df["is_pure"].mean()) if len(rows) else 0.0,
        "mean_clue_ms":    float(np.mean(clue_times)) if clue_times else 0.0,
        "backend":         rows[0]["backend"] if rows else "unknown",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"purity={summary['mean_purity']:.3f}  "
          f"objects={summary['n_objects']}  "
          f"clue={summary['mean_clue_ms']:.1f}ms")


if __name__ == "__main__":
    main()
