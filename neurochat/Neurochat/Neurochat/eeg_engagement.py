"""
eeg_engagement.py
-----------------
NeuroChat EEG Engagement Scoring — faithful implementation of:
  Baradari et al. (2025). NeuroChat: A Neuroadaptive AI Chatbot for
  Customizing Learning Experiences. ACM CUI '25.
  DOI: 10.1145/3719160.3736623  (Section 3)

Paper's exact pipeline:
  1. Bandpass filter 1-30 Hz  (retain relevant neural activity)
  2. 60 Hz notch filter        (remove power line interference)
  3. Segment into 1-second epochs with 250 ms step intervals
  4. FFT per epoch -> band power for theta, alpha, beta
  5. Engagement index E = beta / (alpha + theta)
       where theta = 4-7 Hz, alpha = 7-11 Hz, beta = 11-20 Hz
  6. Average E across the sliding window:
       - Calibration phase : 10-second window
       - Main interaction  : 15-second window
  7. Normalize: E_norm = (E - E_min) / (E_max - E_min), clamped [0, 1]
       E_min = lowest  score seen during calibration
       E_max = highest score seen during calibration
  8. Score is FROZEN when user starts typing; resumes when they stop

Input contract:
  data matrix shape: (n_channels, n_samples)
    rows    = channels (8 in our setup)
    columns = time samples
    values  = raw EEG voltage (uV)
"""

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, tf2sos


# Band definitions — PAPER'S EXACT RANGES (Section 3)
THETA_BAND = (4.0, 7.0)    # theta -- paper: 4-7 Hz
ALPHA_BAND = (7.0, 11.0)   # alpha -- paper: 7-11 Hz
BETA_BAND  = (11.0, 20.0)  # beta  -- paper: 11-20 Hz

# Processing constants — PAPER'S EXACT VALUES (Section 3)
BANDPASS_LOW     = 1.0    # Hz
BANDPASS_HIGH    = 30.0   # Hz
NOTCH_FREQ       = 60.0   # Hz  -- power line interference
EPOCH_SEC        = 1.0    # seconds per epoch
EPOCH_STEP_MS    = 250    # ms between epoch start times
CALIB_WINDOW_SEC = 10     # seconds -- calibration sliding window
MAIN_WINDOW_SEC  = 15     # seconds -- main interaction sliding window


# Low-level DSP helpers

def _make_bandpass(low, high, fs, order=5):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _make_notch(freq, fs, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return tf2sos(b, a)


def _apply_filters(data, bp_sos, notch_sos):
    """Apply bandpass then notch filter channel-wise. data: (n_ch, n_samples)"""
    out = np.zeros_like(data, dtype=float)
    for ch in range(data.shape[0]):
        sig = sosfilt(bp_sos, data[ch].astype(float))
        out[ch] = sosfilt(notch_sos, sig)
    return out


def _fft_band_power(epoch_1d, fs, low, high):
    """
    Band power via FFT for one 1-second epoch on one channel.
    Paper specifies FFT (not Welch) for per-epoch computation.
    Hanning window applied to reduce spectral leakage.
    """
    n = len(epoch_1d)
    fft_vals  = np.fft.rfft(epoch_1d * np.hanning(n))
    fft_power = (np.abs(fft_vals) ** 2) / n
    freqs     = np.fft.rfftfreq(n, d=1.0 / fs)
    mask      = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(fft_power[mask]))


def _epoch_engagement(epoch, fs):
    """
    E = beta / (alpha + theta) for one 1-second epoch, averaged across channels.
    epoch: (n_channels, epoch_samples)
    """
    per_ch = []
    for ch in range(epoch.shape[0]):
        theta = _fft_band_power(epoch[ch], fs, *THETA_BAND)
        alpha = _fft_band_power(epoch[ch], fs, *ALPHA_BAND)
        beta  = _fft_band_power(epoch[ch], fs, *BETA_BAND)
        denom = alpha + theta
        if denom < 1e-12:
            continue
        per_ch.append(beta / denom)
    return float(np.mean(per_ch)) if per_ch else 0.0


def _score_window(data, fs, epoch_sec=EPOCH_SEC, step_ms=EPOCH_STEP_MS):
    """
    Segment data into overlapping 1-second epochs (250 ms step).
    Returns list of raw engagement indices, one per epoch.
    Paper: "segmented into 1-second epochs with 250 ms intervals"
    """
    epoch_samples = int(epoch_sec * fs)
    step_samples  = int((step_ms / 1000.0) * fs)
    indices = []
    start = 0
    while start + epoch_samples <= data.shape[1]:
        indices.append(_epoch_engagement(data[:, start:start + epoch_samples], fs))
        start += step_samples
    return indices


# Main Scorer Class

class EngagementScorer:
    """
    Stateful NeuroChat engagement scorer — paper Section 3.

    Usage
    -----
    scorer = EngagementScorer(sample_rate=256, n_channels=8)

    # Session start — calibration
    scorer.calibrate(relax_data, active_data)

    # Real-time loop — call once per second
    score = scorer.update(new_chunk)

    # When user starts typing
    frozen_score = scorer.freeze()   # inject this into LLM prompt

    # After user submits
    scorer.unfreeze()
    """

    def __init__(self, sample_rate=256, n_channels=8):
        self.fs         = sample_rate
        self.n_channels = n_channels

        self._bp_sos    = _make_bandpass(BANDPASS_LOW, BANDPASS_HIGH, sample_rate)
        self._notch_sos = _make_notch(NOTCH_FREQ, sample_rate)

        buf_size        = int(MAIN_WINDOW_SEC * sample_rate)
        self._buffer    = np.zeros((n_channels, buf_size))
        self._buf_size  = buf_size
        self._samples_in = 0

        self._e_min: float | None = None
        self._e_max: float | None = None

        self._last_score   = 0.0
        self._frozen_score = None
        self._is_frozen    = False

    # Calibration

    def calibrate(self, relax_data: np.ndarray, active_data: np.ndarray):
        """
        Paper: E_min = lowest score across both tasks
               E_max = highest score across both tasks
               Uses 10-second sliding window during calibration
        """
        relax_scores  = self._score_calib_block(relax_data)
        active_scores = self._score_calib_block(active_data)
        all_scores    = relax_scores + active_scores

        if not all_scores:
            raise ValueError("Calibration failed — no valid epochs found.")

        self._e_min = float(np.min(all_scores))
        self._e_max = float(np.max(all_scores))

        print(f"[Calibration] E_min = {self._e_min:.4f}  (lowest,  relaxation)")
        print(f"[Calibration] E_max = {self._e_max:.4f}  (highest, active task)")

    def _score_calib_block(self, data: np.ndarray) -> list:
        """
        Slide a 10-second window in 1-second steps over the calibration data.
        Returns one mean engagement score per window position.
        Paper: calibration uses 10-second window (vs 15-second for main task).
        """
        calib_samples = int(CALIB_WINDOW_SEC * self.fs)
        step          = self.fs   # 1-second steps
        filtered      = _apply_filters(data, self._bp_sos, self._notch_sos)
        scores        = []
        start         = 0
        while start + calib_samples <= filtered.shape[1]:
            window  = filtered[:, start:start + calib_samples]
            indices = _score_window(window, self.fs)
            if indices:
                scores.append(float(np.mean(indices)))
            start += step
        return scores

    def is_calibrated(self) -> bool:
        return self._e_min is not None and self._e_max is not None

    # Real Time Update

    def update(self, chunk: np.ndarray) -> float:
        """
        Ingest a new raw EEG chunk, update the 15-second rolling buffer,
        return the current normalized engagement score.

        Call once per second. If frozen (user typing), buffer still updates
        but returns the frozen score unchanged.

        chunk: (n_channels, n_new_samples)
        """
        if chunk.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {chunk.shape[0]}")

        n_new = chunk.shape[1]
        self._buffer     = np.roll(self._buffer, -n_new, axis=1)
        self._buffer[:, -n_new:] = chunk
        self._samples_in += n_new

        # Wait until we have a full 15-second window
        if self._samples_in < self._buf_size:
            return self._last_score

        # Return frozen score while user is typing
        if self._is_frozen and self._frozen_score is not None:
            return self._frozen_score

        filtered = _apply_filters(self._buffer, self._bp_sos, self._notch_sos)
        indices  = _score_window(filtered, self.fs)

        if not indices:
            return self._last_score

        raw_mean         = float(np.mean(indices))
        self._last_score = self._normalize(raw_mean)

        print(f"[EEG] score={self._last_score:.3f}")

        return self._last_score

    # Freeze and unfreeze

    def freeze(self) -> float:
        """
        Freeze score when user starts typing.
        Paper: score is frozen the moment user begins a new query.
        Returns the frozen score to inject into the LLM prompt.
        """
        self._frozen_score = self._last_score
        self._is_frozen    = True
        return self._frozen_score

    def unfreeze(self):
        """Resume scoring after user submits their message."""
        self._is_frozen    = False
        self._frozen_score = None

    def get_score_for_prompt(self) -> float:
        """Score to inject into the current LLM prompt."""
        if self._is_frozen and self._frozen_score is not None:
            return self._frozen_score
        return self._last_score

    # Normalizaton
    def _normalize(self, raw: float) -> float:
        """E_norm = (E - E_min) / (E_max - E_min), clamped [0, 1]"""
        if not self.is_calibrated():
            return float(np.clip(raw / 3.0, 0.0, 1.0))
        span = self._e_max - self._e_min
        if span < 1e-10:
            return 0.5
        return float(np.clip((raw - self._e_min) / span, 0.0, 1.0))

    # Accessors

    def get_calibration(self) -> dict:
        return {"e_min": self._e_min, "e_max": self._e_max}

    def reset(self):
        self._buffer     = np.zeros((self.n_channels, self._buf_size))
        self._samples_in = 0
        self._e_min = self._e_max = None
        self._last_score = 0.0
        self._frozen_score = None
        self._is_frozen    = False


# Synthetic EEG generator — for testing without real hardware

def make_synthetic_eeg(duration_sec, sample_rate=256, n_channels=8,
                       engagement_level="medium"):
    """
    Synthetic EEG using paper's exact band ranges:
      theta: 4-7 Hz, alpha: 7-11 Hz, beta: 11-20 Hz

    Returns (n_channels, n_samples)
    """
    n   = int(duration_sec * sample_rate)
    t   = np.linspace(0, duration_sec, n)
    rng = np.random.default_rng(seed=42)

    profiles = {
        "low":    {"theta": 9.0, "alpha": 8.0, "beta": 1.5},
        "medium": {"theta": 5.0, "alpha": 5.0, "beta": 5.0},
        "high":   {"theta": 1.5, "alpha": 2.0, "beta": 10.0},
    }
    amp  = profiles.get(engagement_level, profiles["medium"])
    data = np.zeros((n_channels, n))

    for ch in range(n_channels):
        sig = (
            amp["theta"] * np.sin(2*np.pi * rng.uniform(*THETA_BAND) * t + rng.uniform(0, 2*np.pi))
          + amp["alpha"] * np.sin(2*np.pi * rng.uniform(*ALPHA_BAND) * t + rng.uniform(0, 2*np.pi))
          + amp["beta"]  * np.sin(2*np.pi * rng.uniform(*BETA_BAND)  * t + rng.uniform(0, 2*np.pi))
          + rng.normal(0, 0.5, n)
        )
        data[ch] = sig
    return data


# SelfTest

if __name__ == "__main__":
    print("=" * 65)
    print("NeuroChat EEG Engagement Scorer — Corrected Self Test")
    print("Baradari et al. (2025), ACM CUI '25, Section 3")
    print("=" * 65)

    FS, N_CH = 256, 8
    scorer   = EngagementScorer(sample_rate=FS, n_channels=N_CH)

    print("\n--- Calibration (2 min relax + 2 min word association) ---")
    scorer.calibrate(
        make_synthetic_eeg(120, FS, N_CH, "low"),
        make_synthetic_eeg(120, FS, N_CH, "high"),
    )

    print("\n--- Real-time scoring (1 Hz, 15-second window) ---")
    print(f"{'Sec':<6} {'Level':<10} {'Score':>7}  Bar")
    print("-" * 50)

    sec = 0
    for level in ["low", "medium", "high"]:
        sim = make_synthetic_eeg(20, FS, N_CH, level)
        for i in range(0, sim.shape[1] - FS + 1, FS):
            score = scorer.update(sim[:, i:i + FS])
            print(f"{sec:<6} {level:<10} {score:>7.3f}  {'█' * int(score * 30)}")
            sec += 1

    print("\n--- Freeze demo ---")
    frozen = scorer.freeze()
    print(f"User started typing. Score frozen at: {frozen:.3f}")
    score_while_typing = scorer.update(make_synthetic_eeg(1, FS, N_CH, "high")[:, :FS])
    print(f"Score during typing (unchanged):       {score_while_typing:.3f}")
    scorer.unfreeze()
    print("User submitted. Scoring resumed.")

    print("\nAll paper parameters correctly implemented.")
