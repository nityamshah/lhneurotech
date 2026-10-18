import numpy as np
import pytest

from modalities.eeg_brainflow import (
    ALPHA_BAND,
    BETA_BAND,
    THETA_BAND,
    band_power,
    bandpass,
    engagement_index,
    notch,
    preprocess,
)
from signals import FS, eeg_mix, rms, tone

SECONDS = 20
TRIM = 2 * FS  # ignore filter edge effects


def gain(filter_fn, freq, **kw):
    x = tone(freq, SECONDS, amp=10.0)
    y = filter_fn(x, FS, **kw)
    return rms(y[TRIM:-TRIM]) / rms(x[TRIM:-TRIM])


def test_bandpass_preserves_in_band():
    assert gain(bandpass, 10.0) > 0.9


@pytest.mark.parametrize("freq", [0.2, 50.0])
def test_bandpass_attenuates_out_of_band(freq):
    assert gain(bandpass, freq) < 0.05


def test_notch_removes_60hz_and_keeps_10hz():
    assert gain(notch, 60.0) < 0.05
    assert gain(notch, 10.0) > 0.95


@pytest.mark.parametrize(
    "freq, band_name",
    [(5.0, "theta"), (9.0, "alpha"), (15.0, "beta")],
)
def test_band_power_lands_in_the_right_band(freq, band_name):
    x = tone(freq, 15, amp=10.0)
    powers = {
        "theta": band_power(x, FS, THETA_BAND),
        "alpha": band_power(x, FS, ALPHA_BAND),
        "beta": band_power(x, FS, BETA_BAND),
    }
    assert max(powers, key=powers.get) == band_name
    others = [p for name, p in powers.items() if name != band_name]
    assert powers[band_name] > 20 * max(others)


def test_band_power_is_in_signal_units_and_scales_with_amplitude_squared():
    x = tone(9.0, 15, amp=3.0)
    # A sine of amplitude A carries power A^2 / 2.
    assert band_power(x, FS, ALPHA_BAND) == pytest.approx(4.5, rel=0.05)
    assert band_power(2 * x, FS, ALPHA_BAND) == pytest.approx(4 * band_power(x, FS, ALPHA_BAND))


def test_band_power_is_per_channel_for_2d_input():
    data = np.vstack([tone(9.0, 15, amp=1.0), tone(9.0, 15, amp=2.0)])
    low, high = band_power(data, FS, ALPHA_BAND)
    assert high == pytest.approx(4 * low, rel=0.01)


def test_engagement_index_is_higher_for_beta_heavy_signal():
    beta_heavy = preprocess(eeg_mix(15, theta=2, alpha=2, beta=10), FS)
    relaxed = preprocess(eeg_mix(15, theta=10, alpha=10, beta=2), FS)
    assert engagement_index(beta_heavy, FS, [0, 1]) > 3 * engagement_index(relaxed, FS, [0, 1])


def test_engagement_index_ignores_out_of_range_channels_and_rejects_none_valid():
    data = preprocess(eeg_mix(15, theta=2, alpha=2, beta=10), FS)
    assert engagement_index(data, FS, [0, 1, 99]) == pytest.approx(engagement_index(data, FS, [0, 1]))
    with pytest.raises(ValueError):
        engagement_index(data, FS, [99])
