"""Main pipeline entry point.

Runs the full pipeline for a single coordinate-space variant:
    load data → transform coordinates → CLUEstering → metrics → plots

Example usage:

  # Basic run (auto-selects GPU if available):
  caloembed-run --config configs/raw.yaml --data /path/to/data.root --output results/raw/

  # Force CPU:
  caloembed-run --config configs/raw.yaml --data /path/to/data.root --output results/raw/ --backend "cpu serial"

  # Full Nsight Systems profile (captures NVTX ranges + CUDA kernels):
  nsys profile \\
      --trace=cuda,nvtx \\
      --output=results/raw/nsys_profile \\
      --force-overwrite=true \\
      caloembed-run --config configs/raw.yaml --data /path/to/data.root --output results/raw/
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_metadata(config: dict, output_dir: Path) -> None:
    """Save config + git hash + timestamp for reproducibility."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_hash"] = git_hash
    except Exception:
        meta["git_hash"] = "unavailable"

    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


def _get_coord_transform(pipeline: str):
    if pipeline == "raw":
        from caloembed.coords.raw import transform
        return transform
    # Future pipelines slot in here:
    # elif pipeline == "normalized":
    #     from caloembed.coords.normalized import transform; return transform
    # elif pipeline == "umap":
    #     from caloembed.coords.umap_embed import transform; return transform
    # elif pipeline == "gnn":
    #     from caloembed.coords.gnn_embed import transform; return transform
    raise ValueError(f"Unknown pipeline: '{pipeline}'. Valid: raw, normalized, umap, gnn")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CaloEmbed clustering pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", help="Override data path from config")
    parser.add_argument("--output", help="Override output directory from config")
    parser.add_argument("--backend", help="Override CLUEstering backend (e.g. 'gpu cuda')")
    parser.add_argument("--max-events", type=int, help="Override max_events from config")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.data:
        config["data"]["path"] = args.data
    if args.output:
        config["output"]["dir"] = args.output
    if args.backend:
        config["clustering"]["backend"] = args.backend
    if args.max_events is not None:
        config["data"]["max_events"] = args.max_events

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    _save_metadata(config, output_dir)

    pipeline = config.get("pipeline", "raw")
    transform = _get_coord_transform(pipeline)

    # ── Data loading ────────────────────────────────────────────────────────
    from caloembed.data.loader import load_root_events
    from caloembed.clustering.clue import run_clue, available_backends
    from caloembed.metrics import physics as phys_metrics
    from caloembed.metrics.compute import Timer, ComputeMetrics, reset_gpu_memory_stats
    import caloembed.plots as plotting

    data_cfg = config["data"]
    clue_cfg = config["clustering"]
    prof_cfg = config.get("profiling", {})

    use_cuda = "cuda" in clue_cfg.get("backend", "auto") or clue_cfg.get("backend") == "auto"
    use_nvtx = prof_cfg.get("nvtx", False)

    print(f"Available CLUEstering backends: {available_backends()}")

    print(f"Loading data from {data_cfg['path']} ...")
    events = load_root_events(
        path=data_cfg["path"],
        tree_name=data_cfg.get("tree", "events"),
        coord_branches=data_cfg.get("branches", ["rechit_x", "rechit_y", "rechit_z"]),
        weight_branch=data_cfg.get("weight_branch", "rechit_energy"),
        max_events=data_cfg.get("max_events", -1),
    )
    print(f"  Loaded {len(events)} events")

    # ── Per-event loop ───────────────────────────────────────────────────────
    all_phys: list[phys_metrics.PhysicsMetrics] = []
    all_compute: list[dict] = []
    all_cluster_ids: list[np.ndarray] = []
    sample_coords = None

    kernel_cfg = clue_cfg.get("kernel", {})

    for i, event in enumerate(events):
        timer = Timer(use_cuda=use_cuda, use_nvtx=use_nvtx)
        reset_gpu_memory_stats()

        with timer.measure("transform"):
            coords, weights = transform(event)

        with timer.measure("clue"):
            result = run_clue(
                coords=coords,
                weights=weights,
                density_radius=clue_cfg["density_radius"],
                min_density=clue_cfg["min_density"],
                outlier_distance=clue_cfg.get("outlier_distance"),
                seeding_distance=clue_cfg.get("seeding_distance"),
                ppbin=clue_cfg.get("ppbin", 128),
                kernel=kernel_cfg.get("type", "flat"),
                kernel_params=kernel_cfg.get("params", [0.5]),
                backend=clue_cfg.get("backend", "auto"),
                block_size=clue_cfg.get("block_size", 1024),
                device_id=clue_cfg.get("device_id", 0),
            )

        with timer.measure("metrics"):
            if event.truth is not None:
                phys = phys_metrics.compute_with_truth(
                    result.cluster_ids,
                    weights,
                    event.truth,
                    min_shared_fraction=config.get("metrics", {}).get("truth_matching_threshold", 0.5),
                )
            else:
                phys = phys_metrics.compute_cluster_stats(result.cluster_ids, weights)

        compute = ComputeMetrics(timings=timer.timings, backend=result.backend)
        compute.clue_internal_ms = result.elapsed_ms
        compute.capture_gpu_memory()

        all_phys.append(phys)
        all_compute.append(compute.summary())
        all_cluster_ids.append(result.cluster_ids)

        if i == 0:
            sample_coords = coords   # keep first event for scatter plot

        if (i + 1) % max(1, len(events) // 10) == 0 or i == len(events) - 1:
            print(f"  Event {i+1}/{len(events)} — "
                  f"{result.n_clusters} clusters, "
                  f"clue={result.elapsed_ms:.1f} ms [{result.backend}]")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    agg_phys = phys_metrics.aggregate(all_phys)
    mean_timings = {
        k: float(np.mean([c["timings_ms"].get(k, 0) for c in all_compute]))
        for k in all_compute[0]["timings_ms"]
    }

    summary = {
        "pipeline": pipeline,
        "n_events": len(events),
        "backend": all_compute[0]["backend"],
        "physics": agg_phys,
        "mean_timings_ms": mean_timings,
        "mean_total_ms": float(np.mean([c["total_ms"] for c in all_compute])),
        "mean_clue_internal_ms": float(np.mean([c["clue_internal_ms"] for c in all_compute if c["clue_internal_ms"] is not None])),
        "peak_gpu_memory_mb": all_compute[-1].get("peak_gpu_memory_mb"),
    }

    print("\n── Summary ─────────────────────────────────────────")
    print(json.dumps(summary, indent=2))

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_cfg = config.get("output", {})

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "compute_per_event.json").write_text(json.dumps(all_compute, indent=2))

    if out_cfg.get("save_arrays", True):
        np.save(output_dir / "cluster_ids.npy",
                np.array(all_cluster_ids, dtype=object))

    # ── Plots ────────────────────────────────────────────────────────────────
    if out_cfg.get("save_plots", True):
        fmt = out_cfg.get("plot_format", "pdf")

        all_sizes = np.concatenate([m.cluster_sizes for m in all_phys if m.cluster_sizes is not None and len(m.cluster_sizes) > 0])
        if len(all_sizes) > 0:
            plotting.plot_cluster_sizes(all_sizes, plots_dir / f"cluster_sizes.{fmt}", pipeline=pipeline)

        all_energies = np.concatenate([m.cluster_energies for m in all_phys if m.cluster_energies is not None and len(m.cluster_energies) > 0])
        if len(all_energies) > 0:
            plotting.plot_cluster_energies(all_energies, plots_dir / f"cluster_energies.{fmt}", pipeline=pipeline)

        all_responses = np.concatenate([m.energy_response for m in all_phys if m.energy_response is not None and len(m.energy_response) > 0]) if any(m.energy_response is not None for m in all_phys) else np.array([])
        if len(all_responses) > 0:
            plotting.plot_energy_response(all_responses, plots_dir / f"energy_response.{fmt}", pipeline=pipeline)

        timings_list = [c["timings_ms"] for c in all_compute]
        plotting.plot_timing_breakdown(timings_list, plots_dir / f"timing.{fmt}", pipeline=pipeline)

        if sample_coords is not None:
            plotting.plot_hits_2d(sample_coords, all_cluster_ids[0],
                                  output_path=plots_dir / f"hits_2d.{fmt}", pipeline=pipeline)

        print(f"\nPlots saved to {plots_dir}/")

    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
