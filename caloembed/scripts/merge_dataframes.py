"""Merge dataframe directories produced by chunked pipeline runs into one.

A working point split across several jobs (see --file-start) leaves one
objects/efficiency/events.parquet set per chunk. This concatenates them and
rebuilds summary.json so the result is indistinguishable from a single run and
drops straight into caloembed-plot.

Usage:
  caloembed-merge-dataframes --output results/dataframes/d5/transformer_clue_500k_knee \
      results/dataframes/d5/chunks/transformer_clue_500k_knee_*
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_TABLES = ["objects", "efficiency", "events"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge chunked pipeline dataframes")
    parser.add_argument("chunks", nargs="+", help="Chunk directories to merge")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args(argv)

    chunk_dirs = [Path(c) for c in args.chunks]
    missing = [c for c in chunk_dirs if not (c / "summary.json").exists()]
    if missing:
        raise SystemExit("Missing summary.json in: " + ", ".join(str(m) for m in missing))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [json.loads((c / "summary.json").read_text()) for c in chunk_dirs]

    # The chunks must only differ in which files they covered — a mismatch here
    # means dataframes from different working points are about to be merged.
    # 'backend' is excluded: chunks legitimately land on nodes with different
    # CLUE backends (gpu cuda / cpu openmp / cpu serial), which changes nothing
    # about the physics.
    setups = [s["setup"] for s in summaries]

    def _comparable(setup):
        clue = {k: v for k, v in setup.get("clue", {}).items() if k != "backend"}
        return {"pipeline": setup.get("pipeline"), "clue": clue,
                "thresholds": setup.get("thresholds")}

    values = [json.dumps(_comparable(s), sort_keys=True) for s in setups]
    if len(set(values)) > 1:
        raise SystemExit("Chunks disagree on setup — refusing to merge:\n  " +
                         "\n  ".join(f"{c}: {v}" for c, v in zip(chunk_dirs, values)))

    merged = {}
    for name in _TABLES:
        frames = [pd.read_parquet(c / f"{name}.parquet") for c in chunk_dirs]
        frames = [f for f in frames if len(f)]
        merged[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        merged[name].to_parquet(output_dir / f"{name}.parquet", index=False)

    df_objects, df_efficiency, df_events = (merged[n] for n in _TABLES)

    # Guard against double-counting if a file range was covered twice.
    if len(df_events):
        dupes = df_events.duplicated(subset=["file_name", "event_idx", "pdgid"]).sum()
        if dupes:
            raise SystemExit(f"{dupes} duplicated (file_name, event_idx, pdgid) rows — "
                             f"chunk file ranges overlap.")

    n_events = int(df_events.groupby(["file_name", "event_idx"]).ngroups) if len(df_events) else 0

    summary = {
        "run": {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_hash":   summaries[0]["run"].get("git_hash", "unavailable"),
            "output_dir": str(output_dir),
            "merged_from": [str(c) for c in chunk_dirs],
        },
        "setup": {
            **setups[0],
            "n_events": n_events,
            "n_files":  int(df_events["file_name"].nunique()) if len(df_events) else 0,
            "n_failed": int(sum(s.get("n_failed", 0) for s in setups)),
        },
        "results": {
            "n_reco_objects":    len(df_objects),
            "mean_purity":       float(df_objects["is_pure"].mean())         if len(df_objects)    else 0.0,
            "mean_efficiency":   float(df_efficiency["is_efficient"].mean()) if len(df_efficiency) else 0.0,
            "mean_number_ratio": float(df_events["ratio"].mean())            if len(df_events)     else 0.0,
            "mean_ms":           float(np.mean([s["results"]["mean_ms"] for s in summaries])),
        },
        "outputs": {
            "objects":    str(output_dir / "objects.parquet"),
            "efficiency": str(output_dir / "efficiency.parquet"),
            "events":     str(output_dir / "events.parquet"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    r = summary["results"]
    print(f"Merged {len(chunk_dirs)} chunks → {output_dir}")
    print(f"  {n_events} events, {summary['setup']['n_files']} files, {r['n_reco_objects']} reco objects")
    print(f"  purity={r['mean_purity']:.3f}  efficiency={r['mean_efficiency']:.3f}  "
          f"ratio={r['mean_number_ratio']:.3f}")


if __name__ == "__main__":
    main()
