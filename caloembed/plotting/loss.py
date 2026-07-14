"""Reusable plotting functions for training loss curves."""

import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_loss(df: pd.DataFrame, x_col: str = "epoch",
              y_train: str = "train_loss", y_val: str = "val_loss") -> Figure:
    required = {x_col, y_train, y_val}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV columns are missing: {missing}")

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_train], label=y_train)
    ax.plot(df[x_col], df[y_val], label=y_val)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    return fig
