"""Tune the agglomerative-clustering distance threshold on GNN embeddings.

The threshold is a single scalar, so the whole Pareto front is just the image
of a 1D sweep — a deterministic grid gives the exact, complete tradeoff curve
without the stochasticity of a swarm optimiser. For each threshold we cluster
every event and aggregate the same three objectives used for CLUE tuning:

  max_purity_score (minimize)  mean of per-event max(reco-to-sim score)
  min_efficiency   (maximize)  mean of per-event min(per-CP efficiency)
  penalized_ratio  (minimize)  mean of per-event penalised n_reco/n_sim

The GNN hierarchy is built once per event; each threshold is a cheap tree cut.

Usage:
  caloembed-tune-agglo --config configs/tune_agglo.yaml
  caloembed-tune-agglo --config configs/tune_agglo.yaml --output results/run2
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from caloembed.clustering.agglomerative import build_linkage, cluster_at
from caloembed.data.loader import iter_hdf5_events
from caloembed.embed.gnn import GNNEmbedder
from caloembed.metrics.physics import compute_efficiency, compute_purity


@dataclass(frozen=True)
class _Event:
    link: object          # scipy linkage matrix (or None for < 2 LCs)
    n_lc: int
    weights: np.ndarray
    lc_cp_idx: np.ndarray
    n_truth_cp: int
    cp_energies: np.ndarray


def _penalised_ratio(ratio: float, penalty: float) -> float:
    if ratio >= 1.0:
        return ratio
    return 1.0 + penalty * (1.0 - ratio)


def _val_files(run_dir: Path, data_dir: Path) -> list[Path]:
    """Validation HDF5 files from the model's splits.json."""
    splits = json.loads((run_dir / "splits.json").read_text())
    files = [data_dir / name for name in splits["val"]]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} val files not found under {data_dir}, e.g. {missing[0]}")
    return files


def _objectives(events: list[_Event], threshold: float, min_lc: int, penalty: float):
    max_scores, min_effs, pen_ratios = [], [], []
    for ev in events:
        cluster_ids = cluster_at(ev.link, ev.n_lc, threshold)

        purity = compute_purity(
            cluster_ids=cluster_ids, weights=ev.weights,
            lc_cp_idx=ev.lc_cp_idx, cp_energies=ev.cp_energies, min_lc=min_lc,
        )
        eff = compute_efficiency(
            cluster_ids=cluster_ids, weights=ev.weights,
            lc_cp_idx=ev.lc_cp_idx, cp_energies=ev.cp_energies, purity_result=purity,
        )

        scores = purity["score"]
        max_scores.append(float(scores.max()) if len(scores) else 1.0)
        effs = eff["efficiency"]
        min_effs.append(float(effs.min()) if len(effs) else 0.0)
        ratio = len(purity["reco_id"]) / ev.n_truth_cp if ev.n_truth_cp > 0 else 0.0
        pen_ratios.append(_penalised_ratio(ratio, penalty))

    return (
        float(np.mean(max_scores)),
        float(np.mean(min_effs)),
        float(np.mean(pen_ratios)),
    )


def _select_best(df: pd.DataFrame) -> int:
    """Row index closest to the ideal point in normalized objective space.

    penalized_ratio spans orders of magnitude (the under-cluster penalty blows
    up ratio<1 points), so it is compared in log space — otherwise those
    outliers dominate the range and the ratio objective is effectively ignored.
    """
    def norm(values, lower_is_better):
        v = np.asarray(values, dtype=float)
        span = v.max() - v.min()
        if span == 0:
            return np.zeros_like(v)
        bad = (v - v.min()) if lower_is_better else (v.max() - v)
        return bad / span

    dist = np.sqrt(
        norm(df["max_purity_score"], True) ** 2
        + norm(df["min_efficiency"], False) ** 2
        + norm(np.log(df["penalized_ratio"]), True) ** 2
    )
    return int(np.argmin(dist))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune agglomerative distance threshold on GNN embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="e.g. configs/tune_agglo.yaml")
    parser.add_argument("--output", help="Override output.dir from config")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.output:
        cfg["output"]["dir"] = args.output

    run_dir     = Path(cfg["model"]["run_dir"])
    data_dir    = Path(cfg["data"]["dir"])
    n_events    = cfg["data"]["n_events"]
    linkage_m   = cfg["clustering"].get("linkage", "average")
    metric      = cfg["clustering"].get("metric", "cosine")
    sweep_cfg   = cfg["sweep"]
    min_lc      = cfg.get("metrics", {}).get("min_lc", 3)
    penalty     = cfg.get("objectives", {}).get("under_cluster_penalty", 1000.0)
    output_dir  = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    space = np.linspace if sweep_cfg.get("spacing", "linear") == "linear" else np.geomspace
    thresholds = space(sweep_cfg["min"], sweep_cfg["max"], sweep_cfg["n_points"])

    print(f"Loading GNN embedder from {run_dir} ...")
    embedder = GNNEmbedder(run_dir)
    print(f"  device: {embedder.device}  out_dim: {embedder.out_dim}")

    print(f"Embedding {n_events} validation events + building hierarchies ...")
    t0 = time.perf_counter()
    events: list[_Event] = []
    for path in _val_files(run_dir, data_dir):
        if len(events) >= n_events:
            break
        for ev in iter_hdf5_events(path, max_events=n_events - len(events)):
            emb = embedder.embed(ev)
            events.append(_Event(
                link=build_linkage(emb, method=linkage_m, metric=metric),
                n_lc=len(emb),
                weights=ev.weights,
                lc_cp_idx=ev.lc_cp_idx,
                n_truth_cp=ev.n_truth_cp,
                cp_energies=ev.cp_energies,
            ))
    if not events:
        raise RuntimeError("No events loaded — check data.dir / splits.json.")
    print(f"  {len(events)} events in {time.perf_counter() - t0:.1f}s")

    print(f"Sweeping {len(thresholds)} thresholds "
          f"[{thresholds[0]:.3g}, {thresholds[-1]:.3g}] ({metric} + {linkage_m}) ...")
    rows = []
    t0 = time.perf_counter()
    for t in thresholds:
        mp, me, pr = _objectives(events, float(t), min_lc, penalty)
        rows.append({"threshold": float(t),
                     "max_purity_score": mp, "min_efficiency": me, "penalized_ratio": pr})
    df = pd.DataFrame(rows)
    print(f"  swept in {time.perf_counter() - t0:.1f}s")

    best_i = _select_best(df)
    best = df.iloc[best_i]

    df.to_parquet(output_dir / "sweep.parquet", index=False)
    snapshot = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_file":    str(args.config),
        "run_dir":        str(run_dir),
        "data_dir":       str(data_dir),
        "n_events":       len(events),
        "linkage":        linkage_m,
        "metric":         metric,
        "sweep":          {"min": sweep_cfg["min"], "max": sweep_cfg["max"],
                           "n_points": sweep_cfg["n_points"],
                           "spacing": sweep_cfg.get("spacing", "linear")},
        "min_lc":         min_lc,
        "penalty":        penalty,
        "suggested_threshold": float(best["threshold"]),
        "suggested_objectives": {
            "max_purity_score": float(best["max_purity_score"]),
            "min_efficiency":   float(best["min_efficiency"]),
            "penalized_ratio":  float(best["penalized_ratio"]),
        },
    }
    (output_dir / "config.json").write_text(json.dumps(snapshot, indent=2))

    print(f"\n{'threshold':>10} {'max_purity':>11} {'min_eff':>9} {'pen_ratio':>10}")
    for _, r in df.iterrows():
        mark = "  <-- suggested" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:10.4f} {r['max_purity_score']:11.4f} "
              f"{r['min_efficiency']:9.4f} {r['penalized_ratio']:10.4f}{mark}")

    print(f"\nSuggested threshold (closest-to-ideal): {best['threshold']:.4f}")
    print(f"Sweep curve  → {output_dir}/sweep.parquet")
    print(f"Snapshot     → {output_dir}/config.json")


if __name__ == "__main__":
    main()
