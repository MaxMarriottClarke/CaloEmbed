"""Tune CLUEstering parameters with MOPSO on a trained geo_transformer's embeddings.

This is the geo_transformer analogue of `tune_clue_gnn.py`: instead of a GNN's
L2-normalized (unit-sphere) embeddings, we cluster the geo_transformer's raw
Euclidean output — the discriminative margin loss it was trained with sets the
absolute distance scale directly (delta_v = intra-cluster target, delta_d =
inter-cluster target; see training/src/models/geo_transformer.py), so no
L2-normalization is applied and no z_scale axis weighting is needed.

Embeddings are computed once per event (they do not depend on the CLUE
parameters), then MOPSO searches over the CLUE thresholds only.

Parameters tuned (defined in config under 'parameters'):
  dc, rhoc, do, dm  — CLUE distance/density thresholds

Objectives (3), aggregated over the first N events in data.dir (same as tune_clue_gnn):
  max_purity_score (minimize)  mean of per-event max(reco-to-sim score)
  min_efficiency   (maximize)  mean of per-event min(per-CP efficiency)
  penalized_ratio  (minimize)  mean of per-event penalised n_reco/n_sim
                               ratio>=1: score=ratio; ratio<1: score=1+penalty*(1-ratio)

Usage:
  caloembed-tune-clue-transformer --config configs/tune_clue_transformer.yaml
  caloembed-tune-clue-transformer --config configs/tune_clue_transformer.yaml --output results/run2
  caloembed-tune-clue-transformer --config configs/tune_clue_transformer.yaml --resume
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import patatune
import yaml

from caloembed.clustering.clue import run_clue, probe_backend
from caloembed.data.loader import iter_hdf5_dir
from caloembed.embed.transformer import TransformerEmbedder
from caloembed.metrics.physics import compute_efficiency, compute_purity


@dataclass(frozen=True)
class _Event:
    coords: np.ndarray          # (N, 3) transformer embedding
    weights: np.ndarray
    lc_cp_idx: np.ndarray
    n_truth_cp: int
    cp_energies: np.ndarray


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _penalised_ratio(ratio: float, penalty: float) -> float:
    # if ratio = 1 all good
    if ratio >= 1.0:
        return ratio
    return 1.0 + penalty * (1.0 - ratio)


def _fmt_duration(seconds: float) -> str:
    # prints time in appropriate format
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _make_objective(
    events: list,
    backend: str,
    block_size: int,
    device_id: int,
    ppbin: int,
    min_lc: int,
    penalty: float,
):
    """Build the CLUE evaluation closure over pre-embedded events."""

    def evaluate(x: np.ndarray) -> list[float]:
        dc, rhoc, do, dm = (float(v) for v in x)

        max_scores: list[float] = []
        min_effs: list[float] = []
        pen_ratios: list[float] = []

        for ev in events:
            try:
                result = run_clue(
                    coords=ev.coords, weights=ev.weights,
                    dc=dc, rhoc=rhoc, do=do, dm=dm,
                    ppbin=ppbin,
                    metric="euclidean",
                    backend=backend, block_size=block_size, device_id=device_id,
                )
            except RuntimeError:
                max_scores.append(1.0)
                min_effs.append(0.0)
                pen_ratios.append(_penalised_ratio(0.0, penalty))
                continue

            purity = compute_purity(
                cluster_ids=result.cluster_ids,
                weights=ev.weights,
                lc_cp_idx=ev.lc_cp_idx,
                cp_energies=ev.cp_energies,
                min_lc=min_lc,
            )
            eff = compute_efficiency(
                cluster_ids=result.cluster_ids,
                weights=ev.weights,
                lc_cp_idx=ev.lc_cp_idx,
                cp_energies=ev.cp_energies,
                purity_result=purity,
            )

            scores = purity["score"]
            max_scores.append(float(scores.max()) if len(scores) else 1.0)

            efficiencies = eff["efficiency"]
            min_effs.append(float(efficiencies.min()) if len(efficiencies) else 0.0)

            ratio = len(purity["reco_id"]) / ev.n_truth_cp if ev.n_truth_cp > 0 else 0.0
            pen_ratios.append(_penalised_ratio(ratio, penalty))

        return [
            float(np.mean(max_scores)),
            float(np.mean(min_effs)),
            float(np.mean(pen_ratios)),
        ]

    return evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune CLUE parameters with MOPSO on geo_transformer embeddings — see configs/tune_clue_transformer.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",  required=True, help="Tune config YAML (e.g. configs/tune_clue_transformer.yaml)")
    parser.add_argument("--data",    help="Override data.dir from config")
    parser.add_argument("--output",  help="Override output.dir from config")
    parser.add_argument("--backend", help="Override clustering.backend from config")
    parser.add_argument("--resume",  action="store_true", help="Resume from checkpoint")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)

    # CLI overrides
    if args.data:
        cfg["data"]["dir"] = args.data
    if args.output:
        cfg["output"]["dir"] = args.output
    if args.backend:
        cfg["clustering"]["backend"] = args.backend

    data_cfg    = cfg["data"]
    clue_cfg    = cfg["clustering"]
    metrics_cfg = cfg.get("metrics", {})
    params_cfg  = cfg["parameters"]
    mopso_cfg   = cfg["mopso"]
    obj_cfg     = cfg.get("objectives", {})
    run_dir     = Path(cfg["model"]["run_dir"])
    output_dir  = Path(cfg["output"]["dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    min_lc  = metrics_cfg.get("min_lc", 3)
    penalty = obj_cfg.get("under_cluster_penalty", 1000.0)

    param_names   = list(params_cfg.keys())
    lb            = [params_cfg[p]["lower"]   for p in param_names]
    ub            = [params_cfg[p]["upper"]   for p in param_names]
    default_point = [params_cfg[p]["default"] for p in param_names]

    patatune.Randomizer.rng = np.random.default_rng(mopso_cfg["seed"])
    patatune.Logger.setLevel("WARNING")

    patatune.FileManager.saving_enabled  = True
    patatune.FileManager.headers_enabled = True
    patatune.FileManager.working_dir     = str(output_dir)
    patatune.FileManager.loading_enabled = args.resume

    backend    = probe_backend(clue_cfg.get("backend", "auto"), clue_cfg.get("device_id", 0))
    block_size = clue_cfg.get("block_size", 1024)
    device_id  = clue_cfg.get("device_id", 0)
    ppbin      = clue_cfg.get("ppbin", 128)
    print(f"Backend: {backend}")

    data_dir = Path(data_cfg["dir"])
    n_events = data_cfg["n_events"]

    print(f"Loading transformer embedder from {run_dir} ...")
    embedder = TransformerEmbedder(run_dir)
    print(f"  device: {embedder.device}  out_dim: {embedder.out_dim}")

    print(f"Embedding {n_events} events from {data_dir} ...")
    t0 = time.perf_counter()
    events: list[_Event] = []
    for ev in iter_hdf5_dir(dir_path=data_dir, max_events=n_events):
        events.append(_Event(
            coords=embedder.embed(ev),
            weights=ev.weights,
            lc_cp_idx=ev.lc_cp_idx,
            n_truth_cp=ev.n_truth_cp,
            cp_energies=ev.cp_energies,
        ))
    if not events:
        raise RuntimeError("No events loaded — check data.dir in config.")
    print(f"  {len(events)} events in {time.perf_counter() - t0:.1f}s")

    objective = patatune.ElementWiseObjective(
        [_make_objective(events, backend, block_size, device_id, ppbin, min_lc, penalty)],
        num_objectives=3,
        directions=["minimize", "maximize", "minimize"],
        objective_names=["max_purity_score", "min_efficiency", "penalized_ratio"],
    )

    n_particles  = mopso_cfg["n_particles"]
    n_iterations = mopso_cfg["n_iterations"]

    mopso = patatune.MOPSO(
        objective=objective,
        lower_bounds=lb,
        upper_bounds=ub,
        param_names=param_names,
        num_particles=n_particles,
        inertia_weight=mopso_cfg["inertia_weight"],
        cognitive_coefficient=mopso_cfg["cognitive_coefficient"],
        social_coefficient=mopso_cfg["social_coefficient"],
        initial_particles_position=mopso_cfg["initial_position"],
        default_point=default_point,
        topology=mopso_cfg["topology"],
    )

    run_cfg = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_file":   str(args.config),
        "run_dir":       str(run_dir),
        "data_dir":      str(data_dir),
        "n_events":      len(events),
        "backend":       backend,
        "metric":        "euclidean",
        "n_particles":   n_particles,
        "n_iterations":  n_iterations,
        "seed":          mopso_cfg["seed"],
        "param_names":   param_names,
        "lower_bounds":  lb,
        "upper_bounds":  ub,
        "default_point": default_point,
        "min_lc":        min_lc,
        "penalty":       penalty,
    }
    (output_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))
    print(f"Config snapshot: {output_dir}/config.json")

    n_todo = n_iterations - mopso.iteration
    if n_todo <= 0:
        print(f"Already at iteration {mopso.iteration} — nothing to do. "
              f"Increase mopso.n_iterations in config.")
        return

    resume_msg = f" (resuming from iter {mopso.iteration})" if mopso.iteration > 0 else ""
    print(f"Starting MOPSO: {n_particles} particles × {n_iterations} iterations{resume_msg}")

    print(f"Timing first iteration ({n_particles} particles × {len(events)} events)...",
          flush=True)
    t0_iter = time.perf_counter()
    mopso.step()
    mopso.save_state()
    mopso.export_state()
    first_iter_s = time.perf_counter() - t0_iter
    total_est_s  = first_iter_s * n_todo
    finish_at    = time.strftime("%H:%M", time.localtime(time.time() + total_est_s))
    print(f"First iteration:  {first_iter_s:.1f}s")
    print(f"Estimated total:  {n_todo} iters × {first_iter_s:.1f}s/iter"
          f" = {_fmt_duration(total_est_s)}  (finish ~{finish_at})\n")

    iter_times = [first_iter_s]
    log_every  = max(1, n_todo // 20)
    width      = len(str(n_iterations))

    for _ in range(mopso.iteration, n_iterations):
        t0_iter = time.perf_counter()
        mopso.step()
        mopso.save_state()
        mopso.export_state()
        iter_times.append(time.perf_counter() - t0_iter)

        done = mopso.iteration
        if done % log_every == 0 or done == n_iterations:
            avg_s     = float(np.mean(iter_times[-10:]))
            elapsed   = sum(iter_times)
            remaining = avg_s * (n_iterations - done)
            print(f"  iter {done:>{width}}/{n_iterations}"
                  f"  {avg_s:.1f}s/iter"
                  f"  elapsed {_fmt_duration(elapsed)}"
                  f"  remaining ~{_fmt_duration(remaining)}"
                  f"  pareto {len(mopso.pareto_front)}",
                  flush=True)

    patatune.FileManager.save_zarr(
        mopso.history, "checkpoint/mopso.zip",
        param_names=param_names,
        objective_names=objective.objective_names,
        lower_bounds=lb,
        upper_bounds=ub,
    )
    pareto  = mopso.pareto_front
    elapsed = sum(iter_times)

    directions = np.array(objective.directions)
    rows = []
    for p in pareto:
        orig = np.ravel(p.fitness) * directions
        row = {n: float(v) for n, v in zip(param_names, p.position)}
        row["max_purity_score"] = float(orig[0])
        row["min_efficiency"]   = float(orig[1])
        row["penalized_ratio"]  = float(orig[2])
        rows.append(row)
    rows.sort(key=lambda r: r["max_purity_score"])

    import pandas as pd
    pd.DataFrame(rows).to_parquet(output_dir / "pareto_front.parquet", index=False)

    avg_s = elapsed / len(iter_times)
    print(f"\nDone in {_fmt_duration(elapsed)}  ({avg_s:.1f}s/iter avg)  "
          f"Pareto size: {len(pareto)}")
    print(f"\n{'dc':>8} {'rhoc':>7} {'do':>8} {'dm':>8}  "
          f"{'max_purity':>10} {'min_eff':>9} {'pen_ratio':>10}")
    for r in rows:
        print(f"{r['dc']:8.4f} {r['rhoc']:7.4f} {r['do']:8.4f} {r['dm']:8.4f}  "
              f"{r['max_purity_score']:10.4f} {r['min_efficiency']:9.4f} "
              f"{r['penalized_ratio']:10.4f}")

    print(f"\nPareto front  → {output_dir}/pareto_front.parquet")
    print(f"CSV snapshot  → {output_dir}/checkpoint/pareto_front.csv")
    print(f"Checkpoint    → {output_dir}/checkpoint/mopso.pkl")


if __name__ == "__main__":
    main()
