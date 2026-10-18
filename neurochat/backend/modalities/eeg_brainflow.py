"""BrainFlow EEG source plus all DSP for the engagement index.

Follows Baradari et al. (2025), section 3.2: bandpass 1-30 Hz, 60 Hz notch, then segment into
1-second epochs with a 250 ms step, FFT per epoch for theta/alpha/beta band power, and
E = beta / (alpha + theta) per epoch. The score is the mean of E over the sliding window (15 s
live, 10 s for calibration), averaged over the frontal channels.
"""

import asyncio
import contextlib
import logging

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from numpy.lib.stride_tricks import sliding_window_view
from scipy.integrate import trapezoid
from scipy.signal import butter, get_window, iirnotch, sosfiltfilt, tf2sos

from .base import EngagementModality, ModalityOutput

logger = logging.getLogger(__name__)

THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (7.0, 11.0)
BETA_BAND = (11.0, 20.0)
EPOCH_SECONDS = 1.0
EPOCH_STEP_SECONDS = 0.25
CALIBRATION_WINDOW_SECONDS = 10
CLEAN_AMPLITUDE_UV = 100.0

BoardShim.disable_board_logger()


def bandpass(data, fs, low=1.0, high=30.0, order=4):
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch(data, fs, freq=60.0, q=30.0):
    """Zero-phase notch filter along the last axis."""
    b, a = iirnotch(freq, q, fs)
    return sosfiltfilt(tf2sos(b, a), data, axis=-1)


def preprocess(data, fs):
    """Bandpass 1-30 Hz, then 60 Hz notch."""
    return notch(bandpass(data, fs), fs)


def epoch_signal(data, fs, epoch_seconds=EPOCH_SECONDS, step_seconds=EPOCH_STEP_SECONDS):
    """Cut (n_channels, n_samples) into overlapping epochs: (n_channels, n_epochs, epoch_samples).

    The step is rounded to whole samples (62 at 250 Hz, i.e. 248 ms instead of 250 ms).
    """
    data = np.atleast_2d(data)
    length = int(round(epoch_seconds * fs))
    step = max(1, int(round(step_seconds * fs)))
    if data.shape[-1] < length:
        return np.empty((data.shape[0], 0, length))
    return sliding_window_view(data, length, axis=-1)[:, ::step, :]


def fft_band_power(epochs, fs, band):
    """Band power of each epoch (last axis): Hann-windowed FFT, PSD integrated over [low, high] Hz.

    Integration is trapezoidal over the bins inside the band, so a bin that sits exactly on a
    band edge (7 Hz, 11 Hz at 1 Hz resolution) is split half-and-half between the two bands.
    """
    epochs = np.asarray(epochs, dtype=float)
    n = epochs.shape[-1]
    window = get_window("hann", n)
    spectrum = np.fft.rfft(epochs * window, axis=-1)
    psd = 2.0 * np.abs(spectrum) ** 2 / (fs * np.sum(window**2))  # one-sided, units^2 / Hz
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= band[0] - 1e-9) & (freqs <= band[1] + 1e-9)
    return trapezoid(psd[..., mask], freqs[mask], axis=-1)


def epoch_engagement(epochs, fs):
    """E = beta / (alpha + theta + 1e-9) for every epoch: (n_channels, n_epochs)."""
    theta = fft_band_power(epochs, fs, THETA_BAND)
    alpha = fft_band_power(epochs, fs, ALPHA_BAND)
    beta = fft_band_power(epochs, fs, BETA_BAND)
    return beta / (alpha + theta + 1e-9)


def engagement_index(filtered, fs, channel_indices):
    """Mean per-epoch engagement over all epochs of `filtered` and the given (valid) channels."""
    filtered = np.atleast_2d(filtered)
    idx = [i for i in channel_indices if 0 <= i < filtered.shape[0]]
    if not idx:
        raise ValueError("no valid channels for the engagement index")
    epochs = epoch_signal(filtered[idx], fs)
    if epochs.shape[1] == 0:
        raise ValueError("signal is shorter than one epoch")
    return float(np.mean(epoch_engagement(epochs, fs)))


def signal_confidence(filtered, channel_indices=None):
    """Fraction of samples with |amplitude| < 100 uV (on the bandpassed signal)."""
    filtered = np.atleast_2d(filtered)
    if channel_indices is not None:
        idx = [i for i in channel_indices if 0 <= i < filtered.shape[0]]
        filtered = filtered[idx]
    return float(np.mean(np.abs(filtered) < CLEAN_AMPLITUDE_UV))


class EEGBrainflowModality(EngagementModality):
    name = "eeg"

    def __init__(
        self,
        board_id,
        serial_port="",
        window_seconds=15,
        frontal_channel_indices=(0, 1),
        poll_interval=1.0,
    ):
        self.board_id = board_id
        self.serial_port = serial_port
        self.window_seconds = window_seconds
        self.frontal_channel_indices = list(frontal_channel_indices)
        self.poll_interval = poll_interval
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.started = False
        self._max_samples = int(window_seconds * self.sampling_rate)
        self._buffer = np.empty((0, 0))
        self._output: ModalityOutput | None = None
        self._board = None
        self._task = None

    async def start(self):
        if self._board is not None:
            return
        board = None
        try:
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            board = BoardShim(self.board_id, params)
            await asyncio.to_thread(board.prepare_session)
            await asyncio.to_thread(board.start_stream)
        except Exception as exc:
            logger.warning("EEG board %s failed to start: %s", self.board_id, exc)
            if board is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(board.release_session)
            return
        self._board = board
        self.started = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        board, self._board = self._board, None
        self.started = False
        self._output = None
        if board is None:
            return
        for step in (board.stop_stream, board.release_session):
            try:
                await asyncio.to_thread(step)
            except Exception as exc:
                logger.warning("EEG board shutdown step failed: %s", exc)

    def get_output(self) -> ModalityOutput | None:
        return self._output

    def ingest(self, eeg_chunk):
        """Feed one poll's worth of EEG (n_channels, n_samples) and refresh the output.

        An empty chunk means the stream broke: the cached score is dropped rather than served
        stale, and the buffer is cleared so epochs never span the gap (the window refills).
        """
        eeg_chunk = np.asarray(eeg_chunk, dtype=float)
        if eeg_chunk.ndim != 2 or eeg_chunk.shape[1] == 0:
            self._output = None
            self._buffer = np.empty((0, 0))
            return
        if self._buffer.shape[1] == 0:
            self._buffer = eeg_chunk
        else:
            self._buffer = np.concatenate([self._buffer, eeg_chunk], axis=1)
        self._buffer = self._buffer[:, -self._max_samples :]
        self._refresh()

    def _refresh(self):
        if self._buffer.shape[1] < self._max_samples:
            self._output = None
            return
        idx = [i for i in self.frontal_channel_indices if i < self._buffer.shape[0]]
        if not idx:
            self._output = None
            return
        fs = self.sampling_rate
        filtered = preprocess(self._buffer[idx], fs)
        channels = range(len(idx))
        score = engagement_index(filtered, fs, channels)
        # The paper calibrates on a shorter (10 s) sliding window than the live 15 s one.
        calibration_score = engagement_index(filtered[:, -int(CALIBRATION_WINDOW_SECONDS * fs) :], fs, channels)
        if not (np.isfinite(score) and np.isfinite(calibration_score)):
            self._output = None
            return
        self._output = ModalityOutput(
            score=score,
            confidence=signal_confidence(filtered),
            calibration_score=calibration_score,
        )

    async def _poll_loop(self):
        try:
            while True:
                await asyncio.sleep(self.poll_interval)
                data = await asyncio.to_thread(self._board.get_board_data)
                self.ingest(data[self.eeg_channels])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("EEG polling stopped")
            self._output = None
