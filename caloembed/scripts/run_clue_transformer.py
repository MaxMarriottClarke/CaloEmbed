"""Cluster geo_transformer embeddings with CLUE at a fixed parameter set.

Embeds each event with a trained model, runs CLUE on the raw Euclidean
embedding (no L2-normalization, no z_scale — see caloembed/embed/transformer.py),
and computes the same physics metrics as the CLUE pipeline. Outputs match the
schema of caloembed-run (objects/efficiency/events.parquet + summary.json) so
the result overlays as a labelled curve in caloembed-plot.

Usage:
  caloembed-run-clue-transformer --run-dir results/training/geo_transformer_a6000_500k \
      --data /path/to/hdf5 --output results/dataframes/d5/transformer_clue_500k_knee \
      --dc 0.2424 --rhoc 0.01 --do 1.1769 --dm 3.7764
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caloembed.clustering.clue import run_clue, probe_backend
from caloembed.data.loader import iter_hdf5_files
from caloembed.embed.transformer import TransformerEmbedder
from caloembed.metrics.physics import compute_efficiency, compute_purity


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description="geo_transformer + CLUE clustering pipeline")
    parser.add_argument("--run-dir", required=True, help="Training run dir (config.yaml + best.pt)")
    parser.add_argument("--data",    required=True, help="Directory of HDF5 files")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--dc",   type=float, required=True)
    parser.add_argument("--rhoc", type=float, required=True)
    parser.add_argument("--do",   type=float, required=True, dest="do_")
    parser.add_argument("--dm",   type=float, required=True)
    parser.add_argument("--ppbin",      type=int, default=128)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device-id",  type=int, default=0)
    parser.add_argument("--backend",    default="auto")
    parser.add_argument("--min-lc",               type=int,   default=3)
    parser.add_argument("--purity-threshold",     type=float, default=0.2)
    parser.add_argument("--efficiency-threshold", type=float, default=0.5)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--max-files",  type=int, default=-1)
    # File-range selection so one working point can be split across several
    # jobs and merged afterwards (caloembed-merge-dataframes) — the CPU pool
    # only has small free slots, so many short jobs schedule far better than
    # a few long ones.
    parser.add_argument("--file-start", type=int, default=0,
                        help="Index of the first HDF5 file to process (sorted order)")
    parser.add_argument("--device", help="cuda | cpu for the embedder (default: cuda if available)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = probe_backend(args.backend, args.device_id)
    embedder = TransformerEmbedder(args.run_dir, device=args.device)
    print(f"Model: {args.run_dir}  |  device: {embedder.device}  |  out_dim: {embedder.out_dim}")
    print(f"CLUE: dc={args.dc} rhoc={args.rhoc} do={args.do_} dm={args.dm}  |  backend: {backend}")

    obj_cols, eff_cols, event_rows = {}, {}, []
    times = []
    n_failed = 0
    t_start = time.perf_counter()

    files = sorted(Path(args.data).glob("*.h5"))[args.file_start:]
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No HDF5 files selected from {args.data} at --file-start {args.file_start}")
    print(f"Files: {len(files)} ({files[0].name} .. {files[-1].name})")

    events = iter_hdf5_files(files, max_events=args.max_events)
    for i, event in enumerate(events):
        t0 = time.perf_counter()
        emb = embedder.embed(event)
        try:
            result = run_clue(
                coords=emb, weights=event.weights,
                dc=args.dc, rhoc=args.rhoc, do=args.do_, dm=args.dm,
                ppbin=args.ppbin, metric="euclidean",
                backend=backend, block_size=args.block_size, device_id=args.device_id,
            )
        except RuntimeError:
            # Same failure mode the tuner tolerates: CLUE can reject a
            # degenerate event outright. Skip it rather than abort the run.
            n_failed += 1
            continue
        cluster_ids = result.cluster_ids
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        purity = compute_purity(
            cluster_ids=cluster_ids, weights=event.weights,
            lc_cp_idx=event.lc_cp_idx, cp_energies=event.cp_energies,
            min_lc=args.min_lc, score_threshold=args.purity_threshold,
        )
        efficiency = compute_efficiency(
            cluster_ids=cluster_ids, weights=event.weights,
            lc_cp_idx=event.lc_cp_idx, cp_energies=event.cp_energies,
            purity_result=purity, efficiency_threshold=args.efficiency_threshold,
        )
        n_reco = len(purity["reco_id"])
        n_cp   = len(efficiency["cp_idx"])
        matched_cp_pdgids = event.cp_pdg_ids[purity["matched_cp_idx"]]

        for k, arr in purity.items():
            obj_cols.setdefault(k, []).extend(arr.tolist())
        obj_cols.setdefault("matched_cp_pdgid", []).extend(matched_cp_pdgids.tolist())
        obj_cols.setdefault("file_name",  []).extend([event.file_name]  * n_reco)
        obj_cols.setdefault("event_idx",  []).extend([event.event_idx]  * n_reco)
        obj_cols.setdefault("n_truth_cp", []).extend([event.n_truth_cp] * n_reco)
        obj_cols.setdefault("clue_ms",    []).extend([elapsed_ms]       * n_reco)
        obj_cols.setdefault("backend",    []).extend([backend]          * n_reco)

        for k, arr in efficiency.items():
            eff_cols.setdefault(k, []).extend(arr.tolist())
        eff_cols.setdefault("cp_pdgid",   []).extend(
            event.cp_pdg_ids[efficiency["cp_idx"]].tolist())
        eff_cols.setdefault("file_name",  []).extend([event.file_name]  * n_cp)
        eff_cols.setdefault("event_idx",  []).extend([event.event_idx]  * n_cp)
        eff_cols.setdefault("n_truth_cp", []).extend([event.n_truth_cp] * n_cp)
        eff_cols.setdefault("clue_ms",    []).extend([elapsed_ms]       * n_cp)
        eff_cols.setdefault("backend",    []).extend([backend]          * n_cp)

        total_lc_energy = float(event.weights.sum())
        for pdgid in np.unique(event.cp_pdg_ids):
            n_sim_pdg  = int(np.sum(event.cp_pdg_ids == pdgid))
            n_reco_pdg = int(np.sum(matched_cp_pdgids == pdgid))
            event_rows.append(dict(
                file_name=event.file_name, event_idx=event.event_idx,
                pdgid=int(pdgid), n_truth_cp=event.n_truth_cp,
                clue_ms=elapsed_ms, backend=backend,
                total_lc_energy=total_lc_energy,
                n_reco=n_reco_pdg, n_sim=n_sim_pdg,
                ratio=float(n_reco_pdg / n_sim_pdg) if n_sim_pdg > 0 else 0.0,
            ))

        times.append(elapsed_ms)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1} events  ({time.perf_counter() - t_start:.0f}s)", flush=True)

    n_events = len(times)
    elapsed  = time.perf_counter() - t_start
    print(f"\nDone: {n_events} events in {elapsed:.1f}s ({elapsed / max(n_events, 1) * 1000:.1f} ms/event)")
    if n_failed:
        print(f"  {n_failed} events skipped (CLUE raised)")

    df_objects    = pd.DataFrame(obj_cols) if obj_cols else pd.DataFrame()
    df_efficiency = pd.DataFrame(eff_cols) if eff_cols else pd.DataFrame()
    df_events     = pd.DataFrame(event_rows)

    df_objects.to_parquet(   output_dir / "objects.parquet",    index=False)
    df_efficiency.to_parquet(output_dir / "efficiency.parquet", index=False)
    df_events.to_parquet(    output_dir / "events.parquet",     index=False)

    summary = {
        "run": {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_hash":   _git_hash(),
            "output_dir": str(output_dir),
        },
        "setup": {
            "pipeline": "transformer_clue",
            "n_events": n_events,
            "n_files":  int(df_events["file_name"].nunique()) if len(df_events) else 0,
            "n_failed": n_failed,
            "transformer": {
                "run_dir": str(args.run_dir),
                "data":    str(args.data),
            },
            "clue": {
                "dc":      args.dc,
                "rhoc":    args.rhoc,
                "do":      args.do_,
                "dm":      args.dm,
                "ppbin":   args.ppbin,
                "metric":  "euclidean",
                "backend": backend,
            },
            "thresholds": {
                "min_lc":               args.min_lc,
                "purity_threshold":     args.purity_threshold,
                "efficiency_threshold": args.efficiency_threshold,
            },
        },
        "results": {
            "n_reco_objects":    len(df_objects),
            "mean_purity":       float(df_objects["is_pure"].mean())         if len(df_objects)    else 0.0,
            "mean_efficiency":   float(df_efficiency["is_efficient"].mean()) if len(df_efficiency) else 0.0,
            "mean_number_ratio": float(df_events["ratio"].mean())            if len(df_events)     else 0.0,
            "mean_ms":           float(np.mean(times))                       if times              else 0.0,
        },
        "outputs": {
            "objects":    str(output_dir / "objects.parquet"),
            "efficiency": str(output_dir / "efficiency.parquet"),
            "events":     str(output_dir / "events.parquet"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    r = summary["results"]
    print(f"purity={r['mean_purity']:.3f}  efficiency={r['mean_efficiency']:.3f}  "
          f"ratio={r['mean_number_ratio']:.3f}  {r['mean_ms']:.1f}ms/ev")


if __name__ == "__main__":
    main()
