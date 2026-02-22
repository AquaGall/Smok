import numpy as np
import matplotlib.pyplot as plt


def extract_real_signal(df, col):
    signal = df[col].values
    mask = ~np.isnan(signal)
    return signal[mask]


def compute_fft_signal(signal):
    if len(signal) < 2:
        return np.array([]), np.array([])
    signal = signal - np.mean(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal))
    mask = fft_freqs > 0
    return fft_freqs[mask], np.abs(fft_vals[mask])


def plot_fft_single(freqs, mags, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(freqs) > 0:
        ax.plot(freqs, mags, color="blue")
    ax.set_xlabel("Częstotliwość")
    ax.set_ylabel("Amplituda")
    ax.set_title(title)
    ax.grid(True)
    return fig
