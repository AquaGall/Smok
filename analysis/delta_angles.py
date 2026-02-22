import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_delta(df, angle_cols):
    """
    ΔK_i(n) = K_i(n) - K_i(n-1)
    """
    rows = []
    prev = None

    for _, row in df.iterrows():
        if prev is None:
            deltas = [np.nan] * len(angle_cols)
        else:
            deltas = []
            for c in angle_cols:
                a = row[c]
                b = prev[c]
                if np.isnan(a) or np.isnan(b):
                    deltas.append(np.nan)
                else:
                    deltas.append(a - b)
        rows.append([row["n"], row["prime"]] + deltas)
        prev = row

    delta_df = pd.DataFrame(
        rows,
        columns=["n", "prime"] + angle_cols,
    )
    return delta_df


def plot_delta(df, cols, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cols:
        if c in df.columns:
            ax.plot(df["n"], df[c], label=f"Δ{c}", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("ΔKąt")
    ax.legend()
    ax.grid(True)
    return fig
