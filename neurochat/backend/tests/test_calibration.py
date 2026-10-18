import math

import pytest

from calibration import COMPLETE, IDLE, PHASE1, PHASE2, Calibration


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cal(clock):
    return Calibration(["eeg"], phase_seconds=120, clock=clock)


def calibrate(cal, lo=1.0, hi=3.0):
    cal.start(1)
    cal.record("eeg", lo)
    cal.record("eeg", (lo + hi) / 2)
    cal.stop()
    cal.start(2)
    cal.record("eeg", hi)
    cal.record("eeg", (lo + hi) / 2)
    cal.stop()


def test_starts_idle_with_no_bounds(cal):
    assert cal.state == IDLE
    assert cal.e_min == {"eeg": None}
    assert cal.e_max == {"eeg": None}


def test_phase1_keeps_the_lowest_score_as_e_min(cal):
    cal.start(1)
    assert cal.state == PHASE1
    for score in (2.0, 0.8, 1.5):
        cal.record("eeg", score)
    assert cal.e_min["eeg"] == 0.8
    assert cal.e_max["eeg"] is None


def test_phase2_keeps_the_highest_score_as_e_max(cal):
    cal.start(2)
    assert cal.state == PHASE2
    for score in (2.0, 3.5, 1.5):
        cal.record("eeg", score)
    assert cal.e_max["eeg"] == 3.5


def test_full_flow_reaches_complete(cal):
    calibrate(cal, lo=1.0, hi=3.0)
    assert cal.state == COMPLETE
    assert (cal.e_min["eeg"], cal.e_max["eeg"]) == (1.0, 3.0)


def test_record_is_ignored_when_idle_or_complete(cal):
    cal.record("eeg", 5.0)
    assert cal.e_min["eeg"] is None and cal.e_max["eeg"] is None
    calibrate(cal)
    cal.record("eeg", 99.0)
    cal.record("eeg", -99.0)
    assert (cal.e_min["eeg"], cal.e_max["eeg"]) == (1.0, 3.0)


def test_stop_finalizes_phase1_back_to_idle_keeping_e_min(cal):
    cal.start(1)
    cal.record("eeg", 1.2)
    cal.stop()
    assert cal.state == IDLE
    assert cal.e_min["eeg"] == 1.2


def test_stop_when_idle_is_a_noop(cal):
    cal.stop()
    assert cal.state == IDLE


def test_phase1_times_out_after_120_seconds(cal, clock):
    cal.start(1)
    cal.record("eeg", 1.0)
    clock.advance(119.9)
    assert cal.state == PHASE1
    clock.advance(0.2)
    assert cal.state == IDLE
    assert cal.e_min["eeg"] == 1.0


def test_scores_after_the_deadline_are_not_recorded(cal, clock):
    cal.start(1)
    cal.record("eeg", 1.0)
    clock.advance(121)
    cal.record("eeg", 0.1)
    assert cal.e_min["eeg"] == 1.0


def test_phase2_timeout_completes_calibration(cal, clock):
    cal.start(1)
    cal.record("eeg", 1.0)
    cal.stop()
    cal.start(2)
    cal.record("eeg", 3.0)
    clock.advance(120)
    assert cal.state == COMPLETE


def test_starting_phase2_during_phase1_finalizes_phase1(cal):
    cal.start(1)
    cal.record("eeg", 1.0)
    cal.start(2)
    assert cal.state == PHASE2
    assert cal.e_min["eeg"] == 1.0
    cal.record("eeg", 3.0)
    cal.stop()
    assert cal.state == COMPLETE


def test_starting_phase1_again_resets_everything(cal):
    calibrate(cal)
    cal.start(1)
    assert cal.state == PHASE1
    assert cal.e_min["eeg"] is None and cal.e_max["eeg"] is None
    assert cal.normalize(2.0, "eeg") is None


def test_phase2_without_samples_does_not_complete(cal):
    cal.start(1)
    cal.record("eeg", 1.0)
    cal.stop()
    cal.start(2)
    cal.stop()
    assert cal.state == IDLE
    assert cal.normalize(1.0, "eeg") is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_scores_are_ignored(cal, bad):
    cal.start(1)
    cal.record("eeg", bad)
    assert cal.e_min["eeg"] is None


def test_unknown_modality_scores_are_ignored(cal):
    cal.start(1)
    cal.record("webcam", 1.0)
    assert cal.e_min == {"eeg": None}


def test_invalid_phase_raises(cal):
    with pytest.raises(ValueError):
        cal.start(3)


def test_normalize_maps_bounds_to_0_and_1(cal):
    calibrate(cal, lo=1.0, hi=3.0)
    assert cal.normalize(1.0, "eeg") == 0.0
    assert cal.normalize(2.0, "eeg") == 0.5
    assert cal.normalize(3.0, "eeg") == 1.0


def test_normalize_clips_to_unit_interval(cal):
    calibrate(cal, lo=1.0, hi=3.0)
    assert cal.normalize(-5.0, "eeg") == 0.0
    assert cal.normalize(50.0, "eeg") == 1.0


@pytest.mark.parametrize("lo, hi", [(2.0, 2.0), (3.0, 1.0)])
def test_normalize_returns_half_when_bounds_do_not_span(cal, lo, hi):
    cal.start(1)
    cal.record("eeg", lo)
    cal.stop()
    cal.start(2)
    cal.record("eeg", hi)
    cal.stop()
    assert cal.state == COMPLETE
    assert cal.normalize(7.0, "eeg") == 0.5


def test_normalize_is_none_for_unknown_or_uncalibrated(cal):
    assert cal.normalize(1.0, "eeg") is None  # uncalibrated
    calibrate(cal)
    assert cal.normalize(1.0, "webcam") is None  # unknown


def test_normalize_is_none_until_calibration_is_complete(cal):
    cal.start(1)
    cal.record("eeg", 1.0)
    cal.stop()
    cal.start(2)
    cal.record("eeg", 3.0)
    assert cal.normalize(2.0, "eeg") is None  # phase 2 still running


def test_complete_requires_every_modality_to_have_both_bounds(clock):
    cal = Calibration(["eeg", "webcam"], clock=clock)
    cal.start(1)
    cal.record("eeg", 1.0)
    cal.stop()
    cal.start(2)
    cal.record("eeg", 3.0)
    cal.stop()
    assert cal.state == IDLE  # webcam never produced a score
