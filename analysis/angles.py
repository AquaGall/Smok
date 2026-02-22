import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_angles_df(
    n_min,
    n_max,
    ulam_coordinates,
    fabrik,
    compute_angles,
    L,
    R,
    max_iters,
    is_prime,
):
    rows = []

    # maksymalna liczba segmentów – bezpiecznie zawyżamy
    max_k = int(np.ceil(n_max / L)) + 1

    for n in range(int(n_min), int(n_max) + 1):
        x, y = ulam_coordinates(n)
        target = np.array([x, y])

        dist = np.sqrt(x * x + y * y)
        k = int(np.ceil(dist / L))

        centers = [np.array([0.0, 0.0])]
        for _ in range(k):
            centers.append(centers[-1] + np.array([L, 0]))

        centers = fabrik(centers, target, L, R, max_iters=max_iters)
        angles = compute_angles(centers)

        padded = list(angles) + [np.nan] * (max_k - len(angles))
        rows.append([n, is_prime(n), x, y] + padded)

    df = pd.DataFrame(
        rows,
        columns=["n", "prime", "x", "y"] + [f"K{i+1}" for i in range(max_k)],
    )
    return df


def plot_angles(df, cols, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cols:
        if c in df.columns:
            ax.plot(df["n"], df[c], label=c, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("Kąt")
    ax.legend()
    ax.grid(True)
    return fig
