"""Re-score a MOPSO Pareto front on a held-out event set.

The tuning objectives are computed on the same events the swarm optimised over,
so a front's own numbers are optimistic by construction. This re-runs every
front point on a disjoint validation set drawn from the same directory with the
same n_cp mix, and reports tuning-vs-validation side by side. The working point
should be chosen on the validation columns.

Also records mean purity / mean efficiency / mean number ratio as diagnostics.
Those are what the physics plots show, but they are NOT the tuning objectives
(which are per-event extrema), so they are reported and never optimised.

Usage:
  caloembed-eval-front results/param_tuning/raw_d5_strat
  caloembed-eval-front results/param_tuning/raw_d5_strat --seed 137
  caloembed-eval-front results/param_tuning/raw_d5_strat \\
      --extra-point dc=1.1784,rhoc=2.2580,do=3.9951,dm=2.5715,z_scale=0.1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caloembed.data.loader import iter_hdf5_files, select_files_by_n_cp
from caloembed.clustering.clue import run_clue, probe_backend
from caloembed.metrics.physics import compute_purity, compute_efficiency
from caloembed.scripts.tune_clue import _Event, _penalised_ratio, _fmt_duration


PARAM_ORDER = ["dc", "rhoc", "do", "dm", "z_scale"]


def _score_point(params: dict, events, backend, block_size, device_id,
                 min_lc, penalty, ratio_cap):
    """Objectives + diagnostics for one parameter set over all events."""
    max_scores, min_effs, pen_ratios = [], [], []
    mean_purities, mean_effs, ratios = [], [], []

    for ev in events:
        try:
            result = run_clue(
                coords=ev.coords, weights=ev.weights,
                dc=params["dc"], rhoc=params["rhoc"],
                do=params["do"], dm=params["dm"],
                ppbin=128,
                metric="weighted_chebyshev",
                metric_params=[1.0, 1.0, params["z_scale"]],
                backend=backend, block_size=block_size, device_id=device_id,
            )
        except RuntimeError:
            max_scores.append(1.0)
            min_effs.append(0.0)
            pen_ratios.append(_penalised_ratio(0.0, penalty))
            mean_purities.append(0.0)
            mean_effs.append(0.0)
            ratios.append(0.0)
            continue

        purity = compute_purity(
            cluster_ids=result.cluster_ids, weights=ev.weights,
            lc_cp_idx=ev.lc_cp_idx, cp_energies=ev.cp_energies, min_lc=min_lc,
        )
        eff = compute_efficiency(
            cluster_ids=result.cluster_ids, weights=ev.weights,
            lc_cp_idx=ev.lc_cp_idx, cp_energies=ev.cp_energies,
            purity_result=purity,
        )

        scores       = purity["score"]
        efficiencies = eff["efficiency"]

        max_scores.append(float(scores.max()) if len(scores) else 1.0)
        min_effs.append(float(efficiencies.min()) if len(efficiencies) else 0.0)

        ratio = len(purity["reco_id"]) / ev.n_truth_cp if ev.n_truth_cp > 0 else 0.0
        pen_ratios.append(_penalised_ratio(ratio, penalty))

        # Diagnostics: purity is reported as 1 - score to match the physics plots.
        mean_purities.append(float((1.0 - scores).mean()) if len(scores) else 0.0)
        mean_effs.append(float(efficiencies.mean()) if len(efficiencies) else 0.0)
        ratios.append(ratio)

    mean_ratio = float(np.mean(pen_ratios))
    feasible   = mean_ratio <= ratio_cap

    return {
        # Same three objectives as the tuner, in the same orientation.
        "max_purity_score": float(np.mean(max_scores)),
        "min_efficiency":   float(np.mean(min_effs)),
        "penalized_ratio":  mean_ratio,
        "feasible":         feasible,
        # Diagnostics only.
        "mean_purity":      float(np.mean(mean_purities)),
        "mean_efficiency":  float(np.mean(mean_effs)),
        "mean_ratio":       float(np.mean(ratios)),
    }


def _parse_extra_point(spec: str) -> dict:
    point = {}
    for item in spec.split(","):
        key, _, value = item.partition("=")
        point[key.strip()] = float(value)
    missing = set(PARAM_ORDER) - set(point)
    if missing:
        raise ValueError(f"--extra-point is missing {sorted(missing)}")
    return point


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-score a Pareto front on held-out events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_dir", help="Tune output dir (holds config.json + pareto_front.parquet)")
    parser.add_argument("--seed", type=int, default=137,
                        help="Seed for validation file selection (must differ from the tune's select_seed)")
    parser.add_argument("--data", help="Override data dir (default: the tune's data_dir)")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--extra-point", action="append", default=[],
                        help="Also score an arbitrary point, e.g. "
                             "'dc=1.1784,rhoc=2.258,do=3.9951,dm=2.5715,z_scale=0.1'. Repeatable.")
    parser.add_argument("--output", help="Output parquet (default: <results_dir>/pareto_front_val.parquet)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    cfg = json.loads((results_dir / "config.json").read_text())

    front = pd.read_parquet(results_dir / "pareto_front.parquet")
    print(f"Loaded {len(front)} front points from {results_dir}")

    data_dir      = args.data or cfg["data_dir"]
    select_by_ncp = cfg.get("select_by_ncp")
    if not select_by_ncp:
        raise SystemExit(
            "This front was tuned without select_by_ncp, so there is no defined "
            "n_cp mix to reproduce for a validation set. Re-tune with a "
            "stratified config (configs/tune_raw_d5_strat.yaml)."
        )
    if args.seed == cfg.get("select_seed"):
        raise SystemExit(
            f"--seed {args.seed} equals the tune's select_seed — the validation "
            f"set would not be disjoint. Pick a different seed."
        )

    counts = {int(k): int(v) for k, v in select_by_ncp.items()}

    # Rebuild the exact tuning file set, then sample the validation set from
    # what is left. Excluding by path is what makes the two sets disjoint — a
    # different seed alone does not.
    tune_files = select_files_by_n_cp(data_dir, counts, seed=cfg["select_seed"])
    val_files  = select_files_by_n_cp(data_dir, counts, seed=args.seed, exclude=tune_files)
    overlap    = {p.resolve() for p in tune_files} & {p.resolve() for p in val_files}
    assert not overlap, f"validation set overlaps the tuning set: {sorted(overlap)[:3]}"
    print(f"Validation: {len(val_files)} files (disjoint from {len(tune_files)} tuning files), "
          f"n_cp mix {counts}")

    backend = probe_backend(args.backend)
    print(f"Backend: {backend}")

    t0 = time.perf_counter()
    events = [
        _Event(coords=raw.coords, weights=raw.weights, lc_cp_idx=raw.lc_cp_idx,
               n_truth_cp=raw.n_truth_cp, cp_energies=raw.cp_energies)
        for raw in iter_hdf5_files(val_files)
    ]
    print(f"  {len(events)} validation events in {time.perf_counter() - t0:.1f}s")

    min_lc    = cfg.get("min_lc", 3)
    penalty   = cfg.get("penalty", 1000.0)
    ratio_cap = cfg.get("ratio_cap", float("inf"))

    rows = []
    for i, r in enumerate(front.to_dict("records")):
        rows.append({"source": "front", "front_index": i, **r})
    for spec in args.extra_point:
        rows.append({"source": "extra", "front_index": -1, **_parse_extra_point(spec)})

    print(f"Scoring {len(rows)} points on {len(events)} events ...", flush=True)
    t0 = time.perf_counter()
    out = []
    for i, row in enumerate(rows):
        params = {k: float(row[k]) for k in PARAM_ORDER}
        val    = _score_point(params, events, backend, 1024, 0, min_lc, penalty, ratio_cap)
        out.append({
            "source":      row["source"],
            "front_index": row["front_index"],
            **params,
            # Tuning-set values, carried through for the overfit comparison.
            "tune_max_purity_score": row.get("max_purity_score", np.nan),
            "tune_min_efficiency":   row.get("min_efficiency",   np.nan),
            "tune_penalized_ratio":  row.get("penalized_ratio",  np.nan),
            **{f"val_{k}": v for k, v in val.items()},
        })
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            done    = i + 1
            elapsed = time.perf_counter() - t0
            print(f"  {done}/{len(rows)}  elapsed {_fmt_duration(elapsed)}  "
                  f"remaining ~{_fmt_duration(elapsed / done * (len(rows) - done))}", flush=True)

    df = pd.DataFrame(out)
    out_path = Path(args.output) if args.output else results_dir / "pareto_front_val.parquet"
    df.to_parquet(out_path, index=False)

    # Overfit read: how far the objectives move from tuning to validation.
    f = df[df.source == "front"]
    print(f"\nTuning → validation shift over {len(f)} front points (median relative change):")
    for obj in ["max_purity_score", "min_efficiency", "penalized_ratio"]:
        rel = (f[f"val_{obj}"] - f[f"tune_{obj}"]) / f[f"tune_{obj}"].abs().replace(0, np.nan)
        print(f"  {obj:>18}  {rel.median() * 100:+6.1f}%")

    n_infeasible = int((~f["val_feasible"]).sum())
    if n_infeasible:
        print(f"\n{n_infeasible}/{len(f)} front points are INFEASIBLE on validation "
              f"(val penalized_ratio > {ratio_cap}) — they only passed on the tuning set.")

    print(f"\nBest validation points (feasible only):")
    ok = f[f["val_feasible"]]
    if ok.empty:
        print("  none — every front point failed the ratio cap on validation.")
    else:
        for label, idx in [
            ("Best purity",     ok["val_max_purity_score"].idxmin()),
            ("Best efficiency", ok["val_min_efficiency"].idxmax()),
            ("Best ratio",      ok["val_penalized_ratio"].idxmin()),
        ]:
            r = ok.loc[idx]
            print(f"  {label:16s} dc={r.dc:7.4f} rhoc={r.rhoc:7.4f} do={r.do:7.4f} "
                  f"dm={r.dm:7.4f} z={r.z_scale:6.4f}  "
                  f"purity={r.val_max_purity_score:.4f} eff={r.val_min_efficiency:.4f} "
                  f"ratio={r.val_penalized_ratio:.4f}")

    extras = df[df.source == "extra"]
    for _, r in extras.iterrows():
        print(f"  {'[extra point]':16s} dc={r.dc:7.4f} rhoc={r.rhoc:7.4f} do={r.do:7.4f} "
              f"dm={r.dm:7.4f} z={r.z_scale:6.4f}  "
              f"purity={r.val_max_purity_score:.4f} eff={r.val_min_efficiency:.4f} "
              f"ratio={r.val_penalized_ratio:.4f}"
              f"{'' if r.val_feasible else '  [INFEASIBLE]'}")

    print(f"\nValidation front → {out_path}")


if __name__ == "__main__":
    main()
