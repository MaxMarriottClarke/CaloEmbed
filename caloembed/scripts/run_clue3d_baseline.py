"""Compute purity/efficiency metrics for the CLUE3D baseline reconstruction.

Reads ROOT files with uproot — simtrackstersCP for ground truth and
hltTiclTrackstersCLUE3DHigh for CLUE3D cluster assignments — builds the
same compact LC index used in the main pipeline, and runs the existing
metric functions.

Output parquet files match the schema from caloembed-run so they can be
overlaid in caloembed-plot as a separate labelled run entry.

Usage:
    caloembed-clue3d --data /path/to/root/dir --output results/clue3d/
    caloembed-clue3d --data /path/to/root/dir --output results/clue3d/ --max-events 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caloembed.metrics.physics import compute_purity, compute_efficiency

_SIM_TREE = "ticlDumper/simtrackstersCP"
_C3D_TREE = "ticlDumper/hltTiclTrackstersCLUE3DHigh"


def build_cluster_ids(
    trackster_vertex_lists: list,
    n_lc_compact: int,
    global_to_compact: np.ndarray,
) -> np.ndarray:
    """Map CLUE3D tracksters onto the compact LC index space.

    Args:
        trackster_vertex_lists: list of arrays, each holding the global
            within-event LC indices belonging to one CLUE3D trackster.
        n_lc_compact:     number of LCs in the compact (noise-dropped) index.
        global_to_compact: (n_lc_global,) int32 — global LC index → compact
            index; -1 for noise LCs dropped during preprocessing.

    Returns:
        (n_lc_compact,) int32 — cluster_ids compatible with compute_purity /
        compute_efficiency.  LCs not in any trackster are set to -1.
    """
    cluster_ids = np.full(n_lc_compact, -1, dtype=np.int32)
    for trackster_id, global_idxs in enumerate(trackster_vertex_lists):
        compact_idxs = global_to_compact[global_idxs]
        valid = compact_idxs >= 0
        cluster_ids[compact_idxs[valid]] = trackster_id
    return cluster_ids


def _extract_sim_event(
    vertices_indexes_ev,
    vertices_multiplicity_ev,
    vertices_energy_ev,
    n_lc_global: int,
    cp_raw_energy_ev,
    cp_pdg_id_ev,
) -> dict:
    """Build compact LC arrays for one event from uproot per-event slices.

    Returns a dict with:
        weights, lc_cp_idx         — compact LC arrays (noise LCs excluded)
        cp_energies, cp_pdg_ids    — per-CP arrays
        n_compact, global_to_compact
    """
    n_cp = len(vertices_indexes_ev)

    lc_energy       = np.empty(n_lc_global, dtype=np.float32)
    lc_filled       = np.zeros(n_lc_global, dtype=bool)
    argmax_cp_idx   = np.full(n_lc_global, -1,  dtype=np.int32)
    argmax_fraction = np.zeros(n_lc_global,      dtype=np.float32)

    for cp_i in range(n_cp):
        idxs  = np.asarray(vertices_indexes_ev[cp_i],      dtype=np.int32)
        mults = np.asarray(vertices_multiplicity_ev[cp_i], dtype=np.float64)
        es    = np.asarray(vertices_energy_ev[cp_i],       dtype=np.float32)
        fracs = (1.0 / mults).astype(np.float32)

        new_mask = ~lc_filled[idxs]
        lc_energy[idxs[new_mask]] = es[new_mask]
        lc_filled[idxs[new_mask]] = True

        update_mask = fracs > argmax_fraction[idxs]
        argmax_cp_idx[idxs[update_mask]]   = cp_i
        argmax_fraction[idxs[update_mask]] = fracs[update_mask]

    covered = np.where(lc_filled)[0]
    n_compact = len(covered)
    global_to_compact = np.full(n_lc_global, -1, dtype=np.int32)
    global_to_compact[covered] = np.arange(n_compact, dtype=np.int32)

    return {
        "weights":           lc_energy[covered],
        "lc_cp_idx":         argmax_cp_idx[covered],
        "cp_energies":       np.asarray(cp_raw_energy_ev, dtype=np.float32),
        "cp_pdg_ids":        np.asarray(cp_pdg_id_ev,     dtype=np.int32),
        "n_compact":         n_compact,
        "global_to_compact": global_to_compact,
    }


def _process_file(
    root_path, min_lc, score_threshold, efficiency_threshold, max_events, file_name
):
    import uproot

    with uproot.open(root_path) as f:
        n_available = min(f[_SIM_TREE].num_entries, f[_C3D_TREE].num_entries)
        entry_stop  = min(n_available, max_events) if max_events > 0 else n_available

        sim_ak = f[_SIM_TREE].arrays(
            ["NTracksters", "NClusters", "vertices_indexes",
             "vertices_multiplicity", "vertices_energy", "raw_energy", "pdgID"],
            library="ak", entry_stop=entry_stop,
        )
        c3d_ak = f[_C3D_TREE].arrays(
            ["NTracksters", "vertices_indexes"],
            library="ak", entry_stop=entry_stop,
        )

    obj_cols   = {}
    eff_cols   = {}
    event_rows = []

    for ev in range(entry_stop):
        ev_sim = _extract_sim_event(
            sim_ak["vertices_indexes"][ev],
            sim_ak["vertices_multiplicity"][ev],
            sim_ak["vertices_energy"][ev],
            int(sim_ak["NClusters"][ev]),
            sim_ak["raw_energy"][ev],
            sim_ak["pdgID"][ev],
        )
        vertex_lists = [
            np.asarray(c3d_ak["vertices_indexes"][ev][i], dtype=np.int32)
            for i in range(int(c3d_ak["NTracksters"][ev]))
        ]
        cluster_ids = build_cluster_ids(
            vertex_lists, ev_sim["n_compact"], ev_sim["global_to_compact"]
        )

        purity = compute_purity(
            cluster_ids=cluster_ids,
            weights=ev_sim["weights"],
            lc_cp_idx=ev_sim["lc_cp_idx"],
            cp_energies=ev_sim["cp_energies"],
            min_lc=min_lc,
            score_threshold=score_threshold,
        )
        efficiency = compute_efficiency(
            cluster_ids=cluster_ids,
            weights=ev_sim["weights"],
            lc_cp_idx=ev_sim["lc_cp_idx"],
            cp_energies=ev_sim["cp_energies"],
            purity_result=purity,
            efficiency_threshold=efficiency_threshold,
        )

        n_reco            = len(purity["reco_id"])
        n_cp              = len(efficiency["cp_idx"])
        matched_cp_pdgids = ev_sim["cp_pdg_ids"][purity["matched_cp_idx"]]
        n_truth_cp        = int(len(ev_sim["cp_energies"]))

        for k, arr in purity.items():
            obj_cols.setdefault(k, []).extend(arr.tolist())
        obj_cols.setdefault("matched_cp_pdgid", []).extend(matched_cp_pdgids.tolist())
        obj_cols.setdefault("file_name",  []).extend([file_name]  * n_reco)
        obj_cols.setdefault("event_idx",  []).extend([ev]         * n_reco)
        obj_cols.setdefault("n_truth_cp", []).extend([n_truth_cp] * n_reco)
        obj_cols.setdefault("clue_ms",    []).extend([0.0]        * n_reco)
        obj_cols.setdefault("backend",    []).extend(["clue3d"]   * n_reco)

        for k, arr in efficiency.items():
            eff_cols.setdefault(k, []).extend(arr.tolist())
        eff_cols.setdefault("cp_pdgid",   []).extend(
            ev_sim["cp_pdg_ids"][efficiency["cp_idx"]].tolist()
        )
        eff_cols.setdefault("file_name",  []).extend([file_name]  * n_cp)
        eff_cols.setdefault("event_idx",  []).extend([ev]         * n_cp)
        eff_cols.setdefault("n_truth_cp", []).extend([n_truth_cp] * n_cp)
        eff_cols.setdefault("clue_ms",    []).extend([0.0]        * n_cp)
        eff_cols.setdefault("backend",    []).extend(["clue3d"]   * n_cp)

        total_lc_energy = float(ev_sim["weights"].sum())
        for pdgid in np.unique(ev_sim["cp_pdg_ids"]):
            n_sim_pdg  = int(np.sum(ev_sim["cp_pdg_ids"] == pdgid))
            n_reco_pdg = int(np.sum(matched_cp_pdgids == pdgid))
            event_rows.append(dict(
                file_name=file_name,
                event_idx=ev,
                pdgid=int(pdgid),
                n_truth_cp=n_truth_cp,
                clue_ms=0.0,
                backend="clue3d",
                total_lc_energy=total_lc_energy,
                n_reco=n_reco_pdg,
                n_sim=n_sim_pdg,
                ratio=float(n_reco_pdg / n_sim_pdg) if n_sim_pdg > 0 else 0.0,
            ))

    return obj_cols, eff_cols, event_rows, entry_stop


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute CLUE3D baseline metrics")
    parser.add_argument("--data",                 required=True,        help="Directory of ROOT files")
    parser.add_argument("--output",               required=True,        help="Output directory")
    parser.add_argument("--max-events",           type=int, default=-1, help="Cap total events")
    parser.add_argument("--max-files",            type=int, default=-1, help="Cap number of files")
    parser.add_argument("--min-lc",               type=int,   default=3,   metavar="N")
    parser.add_argument("--purity-threshold",     type=float, default=0.2, metavar="F")
    parser.add_argument("--efficiency-threshold", type=float, default=0.5, metavar="F")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_files = sorted(Path(args.data).glob("*.root"))
    if args.max_files > 0:
        root_files = root_files[:args.max_files]
    if not root_files:
        raise RuntimeError(f"No ROOT files found in {args.data}")

    all_obj      = {}
    all_eff      = {}
    all_events   = []
    total_events = 0
    t_start      = time.perf_counter()
    remaining    = args.max_events

    for root_path in root_files:
        if args.max_events >= 0 and total_events >= args.max_events:
            break

        print(f"  {root_path.name} ...", end=" ", flush=True)
        obj_cols, eff_cols, event_rows, n_ev = _process_file(
            root_path,
            min_lc=args.min_lc,
            score_threshold=args.purity_threshold,
            efficiency_threshold=args.efficiency_threshold,
            max_events=remaining,
            file_name=root_path.stem,
        )
        for k, v in obj_cols.items():
            all_obj.setdefault(k, []).extend(v)
        for k, v in eff_cols.items():
            all_eff.setdefault(k, []).extend(v)
        all_events.extend(event_rows)
        total_events += n_ev
        remaining = -1 if args.max_events < 0 else args.max_events - total_events
        print(f"{n_ev} events")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {total_events} events in {elapsed:.1f}s "
          f"({elapsed / max(total_events, 1) * 1000:.1f} ms/event)")

    df_objects    = pd.DataFrame(all_obj)
    df_efficiency = pd.DataFrame(all_eff)
    df_events     = pd.DataFrame(all_events)

    df_objects.to_parquet(   output_dir / "objects.parquet",    index=False)
    df_efficiency.to_parquet(output_dir / "efficiency.parquet", index=False)
    df_events.to_parquet(    output_dir / "events.parquet",     index=False)

    summary = {
        "run": {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "output_dir": str(output_dir),
        },
        "setup": {
            "pipeline": "clue3d",
            "n_events": total_events,
            "n_files":  len(root_files),
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
          f"ratio={r['mean_number_ratio']:.3f}")


if __name__ == "__main__":
    main()
