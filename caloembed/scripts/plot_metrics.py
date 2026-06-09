"""Plot purity, efficiency and number ratio from one or more pipeline runs.

Usage:
    caloembed-plot configs/plots.yaml
    caloembed-plot configs/plots.yaml --output plots/my_run/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


_PARTICLES = [
    (11,  "Electron"),
    (211, "Pion"),
]



def _load_run(entry: dict) -> dict:
    summary_path = Path(entry["summary"])
    summary = json.loads(summary_path.read_text())
    label = entry.get("label") or summary["setup"]["pipeline"]
    outputs = summary["outputs"]
    return {
        "label":      label,
        "summary":    summary,
        "objects":    pd.read_parquet(outputs["objects"]),
        "efficiency": pd.read_parquet(outputs["efficiency"]),
        "events":     pd.read_parquet(outputs["events"]),
    }


def _filter_by_pdgid(run: dict, pdgid: int) -> dict:
    return {
        **run,
        "objects":    run["objects"][   run["objects"]["matched_cp_pdgid"].abs() == pdgid],
        "efficiency": run["efficiency"][ run["efficiency"]["cp_pdgid"].abs()      == pdgid],
        "events":     run["events"][     run["events"]["pdgid"].abs()              == pdgid],
    }



def _binned_fraction(x: np.ndarray, flag: np.ndarray, bins: np.ndarray):
    """Fraction of flag==True per bin with binomial uncertainty."""
    idx = np.digitize(x, bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    frac = np.full(len(centres), np.nan)
    err  = np.full(len(centres), np.nan)
    for b in range(1, len(bins)):
        sel = idx == b
        n = sel.sum()
        if n > 0:
            p = float(flag[sel].mean())
            frac[b - 1] = p
            err[b - 1]  = np.sqrt(p * (1.0 - p) / n)
    return centres, frac, err


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    """Mean of y per bin with standard error."""
    idx = np.digitize(x, bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(centres), np.nan)
    err  = np.full(len(centres), np.nan)
    for b in range(1, len(bins)):
        sel = idx == b
        n = sel.sum()
        if n > 0:
            vals = y[sel]
            mean[b - 1] = vals.mean()
            err[b - 1]  = vals.std() / np.sqrt(n)
    return centres, mean, err



def _plot_purity(ax, runs, bins, colors):
    for run, color in zip(runs, colors):
        df = run["objects"]
        cx, frac, err = _binned_fraction(
            df["matched_cp_energy"].to_numpy(),
            df["is_pure"].to_numpy(),
            bins,
        )
        good = ~np.isnan(frac)
        ax.errorbar(cx[good], frac[good], yerr=err[good],
                    label=run["label"], color=color,
                    marker="o", markersize=5, capsize=3, linewidth=1.5)

    ax.set_xlabel("Matched CP energy [GeV]")
    ax.set_ylabel("Purity fraction")
    ax.set_title("Reco-to-sim purity")
    ax.set_ylim(0, 1.05)


def _plot_efficiency(ax, runs, bins, colors):
    for run, color in zip(runs, colors):
        df = run["efficiency"]
        cx, frac, err = _binned_fraction(
            df["cp_energy"].to_numpy(),
            df["is_efficient"].to_numpy(),
            bins,
        )
        good = ~np.isnan(frac)
        ax.errorbar(cx[good], frac[good], yerr=err[good],
                    label=run["label"], color=color,
                    marker="s", markersize=5, capsize=3, linewidth=1.5)

    ax.set_xlabel("CP energy [GeV]")
    ax.set_ylabel("Efficiency fraction")
    ax.set_title("Sim-to-reco efficiency")
    ax.set_ylim(0, 1.05)


def _plot_number_ratio(ax, runs, bins, colors):
    for run, color in zip(runs, colors):
        df = run["events"]
        cx, mean, err = _binned_mean(
            df["total_lc_energy"].to_numpy(),
            df["ratio"].to_numpy(),
            bins,
        )
        good = ~np.isnan(mean)
        ax.errorbar(cx[good], mean[good], yerr=err[good],
                    label=run["label"], color=color,
                    marker="^", markersize=5, capsize=3, linewidth=1.5)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="N reco = N sim")
    ax.set_xlabel("Total LC energy [GeV]")
    ax.set_ylabel("N reco / N sim")
    ax.set_title("Number ratio")


def _finish_ax(ax):
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(direction="in", top=True, right=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot CaloEmbed metrics")
    parser.add_argument("config", help="YAML plot config file")
    parser.add_argument("--output", help="Override output directory from config")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    runs = [_load_run(r) for r in cfg["runs"]]

    out_dir = Path(args.output or cfg.get("output_dir", "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Energy bins for purity and efficiency (shared across particle types)
    energy_bins = np.linspace(
        cfg.get("energy_min", 20),
        cfg.get("energy_max", 300),
        cfg.get("n_energy_bins", 14) + 1,
    )

    colors = [plt.get_cmap("tab10")(i) for i in range(len(runs))]

    fig, axes = plt.subplots(len(_PARTICLES), 3, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.32, hspace=0.45)

    for row, (pdgid, label) in enumerate(_PARTICLES):
        filtered_runs = [_filter_by_pdgid(r, pdgid) for r in runs]

        # Auto-range ratio bins from this particle type's LC energies
        all_lc_e = np.concatenate([r["events"]["total_lc_energy"].to_numpy() for r in filtered_runs])
        ratio_bins = np.linspace(
            cfg.get("ratio_energy_min", float(np.percentile(all_lc_e, 2))),
            cfg.get("ratio_energy_max", float(np.percentile(all_lc_e, 98))),
            cfg.get("n_ratio_bins", 14) + 1,
        )

        _plot_purity(       axes[row, 0], filtered_runs, energy_bins, colors)
        _plot_efficiency(   axes[row, 1], filtered_runs, energy_bins, colors)
        _plot_number_ratio( axes[row, 2], filtered_runs, ratio_bins,  colors)

        for ax in axes[row]:
            ax.set_title(f"{label}: {ax.get_title()}")
            _finish_ax(ax)

    out_path = out_dir / "metrics.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
