import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(df, cols, title):
    data = df[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        data,
        cmap="viridis",
        cbar=True,
        ax=ax,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Segment (K_i)")
    ax.set_ylabel("n (kolejne liczby)")
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols)
    return fig
