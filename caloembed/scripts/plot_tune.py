"""Plot MOPSO tuning results: Pareto front projections, parameter distributions,
and convergence history.

Usage:
  caloembed-plot-tune results/tune_raw/
  caloembed-plot-tune results/tune_raw/ --output plots/tune/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd



def _finish_ax(ax, legend=True):
    if legend:
        ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(direction="in", top=True, right=True)


def _colorbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)



def _plot_pareto_objectives(df: pd.DataFrame, out_path: Path):
    """Three 2D projections of the Pareto front, each coloured by the third objective."""
    obj_cols   = ["max_purity_score", "min_efficiency", "penalized_ratio"]
    obj_labels = ["Max purity score (min)", "Min efficiency (max)", "Number ratio (min)"]

    # Pairs: (x_idx, y_idx, colour_idx)
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.subplots_adjust(wspace=0.38)

    for ax, (xi, yi, ci) in zip(axes, pairs):
        x = df[obj_cols[xi]].to_numpy()
        y = df[obj_cols[yi]].to_numpy()
        c = df[obj_cols[ci]].to_numpy()

        sc = ax.scatter(x, y, c=c, cmap="viridis_r", s=40, edgecolors="k",
                        linewidths=0.4, zorder=3)
        _colorbar(fig, ax, sc, obj_labels[ci])

        ax.set_xlabel(obj_labels[xi], fontsize=10)
        ax.set_ylabel(obj_labels[yi], fontsize=10)
        _finish_ax(ax, legend=False)

    fig.suptitle("Pareto front: pairwise objective projections", fontsize=12, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")



def _plot_pareto_parameters(df: pd.DataFrame, out_path: Path):
    """Each parameter's value across the Pareto front, coloured by efficiency."""
    param_cols  = ["dc", "rhoc", "do", "dm", "z_scale"]
    param_labels = ["dc", "rhoc", "do", "dm", "z-scale"]

    x   = df["max_purity_score"].to_numpy()
    eff = df["min_efficiency"].to_numpy()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    fig.subplots_adjust(wspace=0.38)

    for ax, col, label in zip(axes, param_cols, param_labels):
        sc = ax.scatter(x, df[col].to_numpy(), c=eff, cmap="RdYlGn",
                        vmin=0, vmax=1, s=40, edgecolors="k",
                        linewidths=0.4, zorder=3)
        ax.set_xlabel("Max purity score (min)", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        _finish_ax(ax, legend=False)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.012, 0.7])
    cb  = fig.colorbar(sc, cax=cax)
    cb.set_label("Min efficiency (max)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Parameter values across the Pareto front", fontsize=12, y=1.02)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")



def _plot_convergence(history_dir: Path, out_path: Path):
    """Per-iteration best objectives and Pareto front size from history CSVs."""
    files = sorted(history_dir.glob("iteration*.csv"),
                   key=lambda p: int(p.stem.replace("iteration", "")))
    if not files:
        print("No history files found — skipping convergence plot.")
        return

    iters, best_purity, best_eff, best_ratio, pareto_sizes = [], [], [], [], []
    pareto_purity, pareto_eff, pareto_ratio = [], [], []

    for f in files:
        it = int(f.stem.replace("iteration", ""))
        df = pd.read_csv(f)
        # Drop any repeated header rows that patatune may write
        df = df[pd.to_numeric(df["dc"], errors="coerce").notna()].astype(float)
        if df.empty:
            continue
        iters.append(it)
        best_purity.append(df["max_purity_score"].min())
        best_eff.append(df["min_efficiency"].max())
        best_ratio.append(df["penalized_ratio"].min())

    iters = np.array(iters)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, vals, label, color in zip(
        axes,
        [best_purity, best_eff, best_ratio],
        ["Best max purity score (min)", "Best min efficiency (max)", "Best penalised ratio (min)"],
        ["steelblue", "seagreen", "tomato"],
    ):
        ax.plot(iters, vals, marker="o", markersize=4, linewidth=1.5, color=color)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        _finish_ax(ax, legend=False)

    fig.suptitle("Best objective value per iteration", fontsize=12, y=1.02)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")



def _print_summary(df: pd.DataFrame):
    """Print a short selection guide: extreme points + balanced compromise."""
    print(f"  Total solutions: {len(df)}\n")

    # Normalise objectives to [0,1] for scoring (lower = better for all after flipping eff)
    p  = df["max_purity_score"].to_numpy()
    e  = df["min_efficiency"].to_numpy()
    r  = df["penalized_ratio"].to_numpy()

    p_n = (p - p.min()) / (p.max() - p.min() + 1e-12)
    e_n = (e.max() - e) / (e.max() - e.min() + 1e-12)   # flip: higher eff is better
    r_n = (r - r.min()) / (r.max() - r.min() + 1e-12)

    balanced = p_n + e_n + r_n

    labels = {
        "Best purity":     df.iloc[p.argmin()],
        "Best efficiency": df.iloc[e.argmax()],
        "Best ratio":      df.iloc[r.argmin()],
        "Balanced":        df.iloc[balanced.argmin()],
    }

    print(f"  {'':18s} {'dc':>7} {'rhoc':>7} {'do':>7} {'dm':>7} {'z_scale':>8}  "
          f"{'purity':>8} {'eff':>8} {'ratio':>8}")
    print(f"  {'-'*85}")
    for name, row in labels.items():
        print(f"  {name:18s} "
              f"{row['dc']:7.4f} {row['rhoc']:7.4f} {row['do']:7.4f} "
              f"{row['dm']:7.4f} {row['z_scale']:8.4f}  "
              f"{row['max_purity_score']:8.4f} {row['min_efficiency']:8.4f} "
              f"{row['penalized_ratio']:8.4f}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot MOPSO tuning results")
    parser.add_argument("results_dir", help="Path to tune results directory")
    parser.add_argument("--output", help="Override output directory (default: <results_dir>/plots)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output) if args.output else Path("plots") / results_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = results_dir / "pareto_front.parquet"
    csv_path     = results_dir / "checkpoint" / "pareto_front.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[pd.to_numeric(df["dc"], errors="coerce").notna()].astype(float)
    else:
        raise FileNotFoundError(f"No pareto_front.parquet or checkpoint/pareto_front.csv in {results_dir}")

    print(f"Loaded {len(df)} Pareto front solutions from {results_dir}")

    _plot_pareto_objectives(df, out_dir / "pareto_objectives.png")
    _plot_pareto_parameters(df, out_dir / "pareto_parameters.png")
    _plot_convergence(results_dir / "history", out_dir / "convergence.png")
    _print_summary(df)


if __name__ == "__main__":
    main()
