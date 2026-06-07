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
from caloembed.clustering.clue import run_clue, probe_backend, available_backends
from caloembed.metrics.physics import compute_purity, compute_efficiency, compute_number_ratio


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
    min_lc               = config.get("metrics", {}).get("min_lc", 3)
    score_threshold      = config.get("metrics", {}).get("purity_threshold", 0.2)
    efficiency_threshold = config.get("metrics", {}).get("efficiency_threshold", 0.5)

    transform = _get_transform(config.get("pipeline", "raw"))

    dc            = clue_cfg["dc"]
    rhoc          = clue_cfg["rhoc"]
    do            = clue_cfg.get("do")
    dm            = clue_cfg.get("dm")
    ppbin         = clue_cfg.get("ppbin", 128)
    metric        = clue_cfg.get("metric", "euclidean")
    metric_params = clue_cfg.get("metric_params")
    backend       = clue_cfg.get("backend", "auto")
    block_size    = clue_cfg.get("block_size", 1024)
    device_id     = clue_cfg.get("device_id", 0)

    selected_backend = probe_backend(backend, device_id)
    print(f"Pipeline: {config.get('pipeline', 'raw')}  |  backend: {selected_backend}")

    events = iter_hdf5_dir(
        dir_path=data_cfg["dir"],
        max_events=data_cfg.get("max_events", -1),
        max_files=data_cfg.get("max_files", -1),
    )

    obj_cols = {}
    eff_cols = {}
    event_rows = []
    clue_times = []
    t_start    = time.perf_counter()

    for i, event in enumerate(events):
        coords, weights = transform(event)

        try:
            result = run_clue(
                coords=coords, weights=weights,
                dc=dc, rhoc=rhoc, do=do, dm=dm, ppbin=ppbin,
                metric=metric, metric_params=metric_params,
                backend=selected_backend, block_size=block_size, device_id=device_id,
            )
        except RuntimeError as e:
            print(f"\n  Warning: [{selected_backend}] failed on event {i} "
                  f"(file={event.file_name}, idx={event.event_idx}, "
                  f"n_lc={len(weights)}): {e}")
            try:
                selected_backend = probe_backend("auto", device_id)
                print(f"  Switched to [{selected_backend}] for remainder of run.")
                result = run_clue(
                    coords=coords, weights=weights,
                    dc=dc, rhoc=rhoc, do=do, dm=dm, ppbin=ppbin,
                    metric=metric, metric_params=metric_params,
                    backend=selected_backend, block_size=block_size, device_id=device_id,
                )
            except RuntimeError as e2:
                print(f"  Skipping event {i}: all backends failed ({e2})")
                continue

        purity = compute_purity(
            cluster_ids=result.cluster_ids,
            weights=weights,
            lc_cp_idx=event.lc_cp_idx,
            cp_energies=event.cp_energies,
            min_lc=min_lc,
            score_threshold=score_threshold,
        )
        efficiency = compute_efficiency(
            cluster_ids=result.cluster_ids,
            weights=weights,
            lc_cp_idx=event.lc_cp_idx,
            cp_energies=event.cp_energies,
            purity_result=purity,
            efficiency_threshold=efficiency_threshold,
        )
        nr = compute_number_ratio(purity, event.n_truth_cp)

        n_reco = len(purity["reco_id"])
        n_cp   = len(efficiency["cp_idx"])

        for k, arr in purity.items():
            obj_cols.setdefault(k, []).extend(arr.tolist())
        obj_cols.setdefault("file_name",  []).extend([event.file_name]  * n_reco)
        obj_cols.setdefault("event_idx",  []).extend([event.event_idx]  * n_reco)
        obj_cols.setdefault("n_truth_cp", []).extend([event.n_truth_cp] * n_reco)
        obj_cols.setdefault("clue_ms",    []).extend([result.elapsed_ms] * n_reco)
        obj_cols.setdefault("backend",    []).extend([result.backend]   * n_reco)

        for k, arr in efficiency.items():
            eff_cols.setdefault(k, []).extend(arr.tolist())
        eff_cols.setdefault("file_name",  []).extend([event.file_name]  * n_cp)
        eff_cols.setdefault("event_idx",  []).extend([event.event_idx]  * n_cp)
        eff_cols.setdefault("n_truth_cp", []).extend([event.n_truth_cp] * n_cp)
        eff_cols.setdefault("clue_ms",    []).extend([result.elapsed_ms] * n_cp)
        eff_cols.setdefault("backend",    []).extend([result.backend]   * n_cp)

        event_rows.append(dict(
            file_name=event.file_name, event_idx=event.event_idx,
            n_truth_cp=event.n_truth_cp, clue_ms=result.elapsed_ms,
            backend=result.backend, total_lc_energy=float(weights.sum()), **nr,
        ))

        clue_times.append(result.elapsed_ms)

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i+1} events  ({elapsed:.1f}s)  last: {result.n_clusters} clusters [{result.backend}]")

    n_events = len(clue_times)
    elapsed  = time.perf_counter() - t_start
    print(f"\nDone: {n_events} events in {elapsed:.1f}s ({elapsed / max(n_events, 1) * 1000:.1f} ms/event)")

    df_objects    = pd.DataFrame(obj_cols)    if obj_cols    else pd.DataFrame()
    df_efficiency = pd.DataFrame(eff_cols)    if eff_cols    else pd.DataFrame()
    df_events     = pd.DataFrame(event_rows)

    df_objects.to_parquet(   output_dir / "objects.parquet",    index=False)
    df_efficiency.to_parquet(output_dir / "efficiency.parquet", index=False)
    df_events.to_parquet(    output_dir / "events.parquet",     index=False)

    print(f"Per-object metrics  → {output_dir / 'objects.parquet'}")
    print(f"Per-CP efficiency   → {output_dir / 'efficiency.parquet'}")
    print(f"Per-event ratios    → {output_dir / 'events.parquet'}")

    actual_backend = df_events["backend"].iloc[0] if len(df_events) else "unknown"

    summary = {
        "run": {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_hash":   _git_hash(),
            "output_dir": str(output_dir),
        },
        "setup": {
            "pipeline": config.get("pipeline", "raw"),
            "n_events": n_events,
            "n_files":  int(df_events["file_name"].nunique()) if len(df_events) else 0,
            "clue": {
                "dc":           dc,
                "rhoc":         rhoc,
                "do":           do,
                "dm":           dm,
                "ppbin":        ppbin,
                "metric":       metric,
                "metric_params": metric_params,
                "backend":      actual_backend,
            },
            "thresholds": {
                "min_lc":               min_lc,
                "purity_threshold":     score_threshold,
                "efficiency_threshold": efficiency_threshold,
            },
        },
        "results": {
            "n_reco_objects":    len(df_objects),
            "mean_purity":       float(df_objects["is_pure"].mean())       if len(df_objects)    else 0.0,
            "mean_efficiency":   float(df_efficiency["is_efficient"].mean()) if len(df_efficiency) else 0.0,
            "mean_number_ratio": float(df_events["ratio"].mean())           if len(df_events)     else 0.0,
            "mean_clue_ms":      float(np.mean(clue_times))                if clue_times         else 0.0,
        },
        "outputs": {
            "objects":    str(output_dir / "objects.parquet"),
            "efficiency": str(output_dir / "efficiency.parquet"),
            "events":     str(output_dir / "events.parquet"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    r = summary["results"]
    print(f"purity={r['mean_purity']:.3f}  "
          f"efficiency={r['mean_efficiency']:.3f}  "
          f"ratio={r['mean_number_ratio']:.3f}  "
          f"clue={r['mean_clue_ms']:.1f}ms")


if __name__ == "__main__":
    main()
