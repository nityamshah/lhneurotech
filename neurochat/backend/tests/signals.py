"""Synthetic signal helpers shared by the backend tests."""

import numpy as np

FS = 250


def tone(freq, seconds, fs=FS, amp=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def eeg_mix(seconds, theta=0.0, alpha=0.0, beta=0.0, n_channels=2, fs=FS, seed=0):
    """Sum of one theta (5.5 Hz), alpha (9 Hz) and beta (15 Hz) tone per channel plus light noise.

    Amplitudes are in uV; keep them well under 100 so confidence stays high.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_channels, int(seconds * fs)))
    for ch in range(n_channels):
        out[ch] = (
            tone(5.5, seconds, fs, theta)
            + tone(9.0, seconds, fs, alpha)
            + tone(15.0, seconds, fs, beta)
            + rng.normal(0, 0.5, out.shape[1])
        )
    return out
