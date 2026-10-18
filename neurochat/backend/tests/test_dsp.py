import numpy as np
import pytest

from modalities.eeg_brainflow import (
    ALPHA_BAND,
    BETA_BAND,
    THETA_BAND,
    bandpass,
    engagement_index,
    epoch_engagement,
    epoch_signal,
    fft_band_power,
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


def band_powers(x):
    """Mean per-epoch band power of a 1-D signal in each band."""
    epochs = epoch_signal(x, FS)
    return {
        name: float(np.mean(fft_band_power(epochs, FS, band)))
        for name, band in (("theta", THETA_BAND), ("alpha", ALPHA_BAND), ("beta", BETA_BAND))
    }


# --- filtering -------------------------------------------------------------------------------


def test_bandpass_preserves_in_band():
    assert gain(bandpass, 10.0) > 0.9


@pytest.mark.parametrize("freq", [0.2, 50.0])
def test_bandpass_attenuates_out_of_band(freq):
    assert gain(bandpass, freq) < 0.05


def test_notch_removes_60hz_and_keeps_10hz():
    assert gain(notch, 60.0) < 0.05
    assert gain(notch, 10.0) > 0.95


# --- epoching --------------------------------------------------------------------------------


def test_epochs_are_one_second_long_with_a_250ms_step():
    x = np.arange(15 * FS, dtype=float)[None, :]
    epochs = epoch_signal(x, FS)
    assert epochs.shape == (1, 57, FS)  # 1 s epochs, step 62 samples (~250 ms) over 15 s
    assert epochs[0, 0, 0] == 0
    assert epochs[0, 1, 0] == 62
    assert epochs[0, 2, 0] == 124
    np.testing.assert_array_equal(epochs[0, 5], x[0, 5 * 62 : 5 * 62 + FS])


def test_epoching_is_exact_when_the_step_is_a_whole_number_of_samples():
    x = np.zeros((2, 15 * 200))  # Ganglion: 200 Hz -> 50-sample step = exactly 250 ms
    epochs = epoch_signal(x, 200)
    assert epochs.shape == (2, 57, 200)


def test_signal_shorter_than_one_epoch_gives_no_epochs():
    assert epoch_signal(np.zeros((2, FS - 1)), FS).shape == (2, 0, FS)


# --- per-epoch FFT band power ----------------------------------------------------------------


@pytest.mark.parametrize(
    "freq, band_name",
    [(5.0, "theta"), (9.0, "alpha"), (15.0, "beta")],
)
def test_band_power_lands_in_the_right_band(freq, band_name):
    powers = band_powers(tone(freq, 15, amp=10.0))
    assert max(powers, key=powers.get) == band_name
    others = [p for name, p in powers.items() if name != band_name]
    assert powers[band_name] > 20 * max(others)


def test_band_power_is_in_signal_units_and_scales_with_amplitude_squared():
    # A sine of amplitude A carries power A^2 / 2.
    assert band_powers(tone(9.0, 15, amp=3.0))["alpha"] == pytest.approx(4.5, rel=0.05)
    assert band_powers(tone(9.0, 15, amp=6.0))["alpha"] == pytest.approx(
        4 * band_powers(tone(9.0, 15, amp=3.0))["alpha"]
    )


def test_a_tone_on_a_band_edge_is_shared_between_the_two_bands():
    powers = band_powers(tone(7.0, 15, amp=5.0))  # exactly on the theta/alpha boundary
    assert powers["theta"] == pytest.approx(powers["alpha"], rel=0.05)
    assert powers["beta"] < 0.01 * powers["alpha"]


def test_band_power_is_per_channel_and_per_epoch():
    data = np.vstack([tone(9.0, 15, amp=1.0), tone(9.0, 15, amp=2.0)])
    power = fft_band_power(epoch_signal(data, FS), FS, ALPHA_BAND)
    assert power.shape == (2, 57)
    assert power[1].mean() == pytest.approx(4 * power[0].mean(), rel=0.01)


# --- engagement ------------------------------------------------------------------------------


def test_engagement_per_epoch_is_beta_over_alpha_plus_theta():
    epochs = epoch_signal(tone(15.0, 5, amp=4.0) + tone(9.0, 5, amp=2.0) + tone(5.0, 5, amp=2.0), FS)
    theta = fft_band_power(epochs, FS, THETA_BAND)
    alpha = fft_band_power(epochs, FS, ALPHA_BAND)
    beta = fft_band_power(epochs, FS, BETA_BAND)
    np.testing.assert_allclose(epoch_engagement(epochs, FS), beta / (alpha + theta + 1e-9))
    # amplitudes 4 vs 2 and 2 -> beta power 8, alpha 2, theta 2 -> E close to 2
    assert epoch_engagement(epochs, FS).mean() == pytest.approx(2.0, rel=0.1)


def test_engagement_index_is_the_mean_over_epochs_and_channels():
    filtered = preprocess(eeg_mix(15, theta=3, alpha=3, beta=6), FS)
    per_epoch = epoch_engagement(epoch_signal(filtered, FS), FS)
    assert engagement_index(filtered, FS, [0, 1]) == pytest.approx(per_epoch.mean())
    assert engagement_index(filtered, FS, [0]) == pytest.approx(per_epoch[0].mean())


def test_engagement_index_averages_ratios_not_band_powers():
    # Half the window is beta-only, half is theta-only: the mean of per-epoch ratios is dominated
    # by the beta epochs, unlike a ratio of window-averaged band powers.
    x = np.concatenate([tone(15.0, 7.5, amp=5.0), tone(5.0, 7.5, amp=5.0)])[None, :]
    epochs = epoch_signal(x, FS)
    ratios = epoch_engagement(epochs, FS)[0]
    assert engagement_index(x, FS, [0]) == pytest.approx(ratios.mean())
    mean_beta = fft_band_power(epochs, FS, BETA_BAND).mean()
    mean_alpha_theta = (fft_band_power(epochs, FS, ALPHA_BAND) + fft_band_power(epochs, FS, THETA_BAND)).mean()
    assert ratios.mean() > 5 * (mean_beta / mean_alpha_theta)


def test_engagement_index_is_higher_for_beta_heavy_signal():
    beta_heavy = preprocess(eeg_mix(15, theta=2, alpha=2, beta=10), FS)
    relaxed = preprocess(eeg_mix(15, theta=10, alpha=10, beta=2), FS)
    assert engagement_index(beta_heavy, FS, [0, 1]) > 3 * engagement_index(relaxed, FS, [0, 1])


def test_engagement_index_ignores_out_of_range_channels_and_rejects_none_valid():
    data = preprocess(eeg_mix(15, theta=2, alpha=2, beta=10), FS)
    assert engagement_index(data, FS, [0, 1, 99]) == pytest.approx(engagement_index(data, FS, [0, 1]))
    with pytest.raises(ValueError):
        engagement_index(data, FS, [99])


def test_engagement_index_rejects_a_signal_shorter_than_one_epoch():
    with pytest.raises(ValueError):
        engagement_index(np.zeros((2, FS - 1)), FS, [0, 1])
