"""Plot training/validation loss curves from a metrics.csv file.

Usage:
    caloembed-plot-loss results/training/edgeconv_infonce/metrics.csv
    caloembed-plot-loss results/training/edgeconv_infonce/metrics.csv --output plots/loss.png
"""

import argparse
from pathlib import Path

import pandas as pd

from caloembed.plotting.loss import plot_loss


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot training/validation loss curves")
    parser.add_argument("csv_path", type=Path, help="Path to metrics.csv from training")
    parser.add_argument("--output", type=Path, help="Output image path (default: <csv_dir>/loss.png)")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv_path)
    fig = plot_loss(df)

    out_path = args.output or args.csv_path.parent / "loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
