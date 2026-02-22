# fabrik_engine.py

import numpy as np

def fabrik(centers, target, L, R, max_iters=50, tol=1e-4):
    """
    Klasyczny algorytm FABRIK z końcówką o promieniu R.
    centers: lista punktów (numpy arrays)
    target: punkt docelowy (numpy array)
    L: długość segmentu
    R: promień końcówki
    """
    centers = centers.copy()
    base = centers[0].copy()

    total_len = (len(centers)-1) * L + R
    if np.linalg.norm(target - base) > total_len:
        direction = (target - base) / np.linalg.norm(target - base)
        for i in range(1, len(centers)):
            centers[i] = centers[i-1] + direction * L
        return centers

    for _ in range(max_iters):
        centers[-1] = target - (centers[-1] - centers[-2]) / np.linalg.norm(centers[-1] - centers[-2]) * R

        for i in reversed(range(len(centers)-1)):
            d = centers[i] - centers[i+1]
            centers[i] = centers[i+1] + d / np.linalg.norm(d) * L

        centers[0] = base
        for i in range(1, len(centers)):
            d = centers[i] - centers[i-1]
            centers[i] = centers[i-1] + d / np.linalg.norm(d) * L

        end = centers[-1] + (centers[-1] - centers[-2]) / np.linalg.norm(centers[-1] - centers[-2]) * R
        if np.linalg.norm(end - target) < tol:
            break

    return centers


def compute_angles(centers):
    """
    Zwraca listę kątów między segmentami (w radianach).
    """
    angles = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        angles.append(np.arctan2(dy, dx))
    return np.array(angles)
