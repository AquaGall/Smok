import numpy as np

def drop_nan(signal):
    mask = ~np.isnan(signal)
    return signal[mask]

def standardize(signal):
    mu = np.nanmean(signal)
    sigma = np.nanstd(signal)
    if sigma == 0:
        return signal * 0
    return (signal - mu) / sigma
