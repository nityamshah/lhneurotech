import asyncio

import numpy as np
import pytest
from brainflow.board_shim import BoardIds, BoardShim

from modalities.eeg_brainflow import EEGBrainflowModality
from signals import FS, eeg_mix, tone

SYNTHETIC = int(BoardIds.SYNTHETIC_BOARD)
CYTON = int(BoardIds.CYTON_BOARD)
WINDOW = 15


def make(board_id=SYNTHETIC, **kw):
    return EEGBrainflowModality(board_id, **kw)


async def test_start_failure_is_graceful(monkeypatch):
    def boom(self):
        raise RuntimeError("no board here")

    monkeypatch.setattr(BoardShim, "prepare_session", boom)
    modality = make()
    await modality.start()
    assert modality.started is False
    assert modality.get_output() is None
    await modality.stop()


async def test_cyton_without_hardware_does_not_crash():
    modality = make(CYTON, serial_port="COM99" if __import__("sys").platform == "win32" else "/dev/ttyNOPE")
    await modality.start()
    assert modality.started is False
    assert modality.get_output() is None
    await modality.stop()


async def test_stop_is_safe_when_never_started():
    modality = make()
    await modality.stop()
    await modality.stop()
    assert modality.started is False


def test_returns_none_until_a_full_window_is_buffered():
    modality = make()
    modality.ingest(eeg_mix(5, beta=5))
    assert modality.get_output() is None
    modality.ingest(eeg_mix(WINDOW - 5, beta=5)[:, :-1])  # one sample short of 15 s
    assert modality.get_output() is None
    modality.ingest(eeg_mix(1, beta=5)[:, :1])  # now exactly 15 s
    assert modality.get_output() is not None


def test_buffer_keeps_only_the_latest_window():
    modality = make()
    modality.ingest(eeg_mix(20, beta=5))
    assert modality._buffer.shape[1] == WINDOW * FS


def test_full_pipeline_on_mixed_theta_alpha_beta():
    engaged, relaxed = make(), make()
    engaged.ingest(eeg_mix(16, theta=2, alpha=2, beta=10))
    relaxed.ingest(eeg_mix(16, theta=10, alpha=10, beta=2))
    a, r = engaged.get_output(), relaxed.get_output()
    assert a is not None and r is not None
    assert np.isfinite(a.score) and a.score > 0
    assert a.score > 3 * r.score
    assert a.confidence == pytest.approx(1.0)


def test_artifact_bursts_lower_confidence():
    clean, noisy = make(), make()
    data = eeg_mix(16, theta=2, alpha=2, beta=10)
    clean.ingest(data)
    burst = data.copy()
    burst[:, :1000] += tone(10.0, 1000 / FS, amp=400.0)
    noisy.ingest(burst)
    assert noisy.get_output().confidence < 0.9
    assert noisy.get_output().confidence < clean.get_output().confidence


def test_a_poll_with_no_new_samples_drops_the_cached_score():
    modality = make()
    modality.ingest(eeg_mix(16, beta=5))
    assert modality.get_output() is not None
    modality.ingest(np.empty((2, 0)))
    assert modality.get_output() is None


def test_calibration_score_uses_the_shorter_ten_second_window():
    # 5 s of beta-heavy signal followed by 10 s of relaxed signal.
    mixed = np.concatenate(
        [eeg_mix(5, theta=2, alpha=2, beta=10), eeg_mix(10, theta=10, alpha=10, beta=2)], axis=1
    )
    modality, relaxed_only = make(), make()
    modality.ingest(mixed)
    relaxed_only.ingest(eeg_mix(15, theta=10, alpha=10, beta=2))
    out = modality.get_output()
    assert out.calibration_score < out.score  # the 15 s score still includes the beta-heavy start
    assert out.calibration_score == pytest.approx(relaxed_only.get_output().score, rel=0.25)


def test_score_and_calibration_score_agree_on_a_stationary_signal():
    modality = make()
    modality.ingest(eeg_mix(15, theta=3, alpha=3, beta=6))
    out = modality.get_output()
    assert out.calibration_score == pytest.approx(out.score, rel=0.05)


def test_a_gap_in_the_stream_clears_the_buffer_so_epochs_never_span_it():
    modality = make()
    modality.ingest(eeg_mix(16, beta=5))
    modality.ingest(np.empty((2, 0)))
    assert modality.get_output() is None
    modality.ingest(eeg_mix(5, beta=5))
    assert modality._buffer.shape[1] == 5 * FS
    assert modality.get_output() is None  # a full 15 s window must be rebuilt


def test_frontal_indices_beyond_the_channel_count_are_dropped():
    modality = make(frontal_channel_indices=[0, 1, 99])
    modality.ingest(eeg_mix(16, beta=5))
    assert modality.get_output() is not None
    none_valid = make(frontal_channel_indices=[99])
    none_valid.ingest(eeg_mix(16, beta=5))
    assert none_valid.get_output() is None


async def test_real_synthetic_board_streams_and_stops_cleanly():
    modality = make(poll_interval=0.2)
    await modality.start()
    try:
        assert modality.started is True
        assert modality.sampling_rate == 250
        assert modality.eeg_channels == list(range(1, 17))
        await asyncio.sleep(1.2)
        assert modality._buffer.shape[1] > 0
        assert modality.get_output() is None  # far less than 15 s buffered
    finally:
        await modality.stop()
    assert modality.started is False
