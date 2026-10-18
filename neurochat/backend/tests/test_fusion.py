import json

import pytest

from calibration import Calibration
from fusion import EngagementFuser
from modalities.base import EngagementModality, ModalityOutput


class StubModality(EngagementModality):
    def __init__(self, name, score=None, confidence=1.0):
        self.name = name
        self.output = None if score is None else ModalityOutput(score, confidence)

    async def start(self): ...

    async def stop(self): ...

    def get_output(self):
        return self.output


def calibrated(names, lo=0.0, hi=10.0):
    cal = Calibration(names)
    cal.start(1)
    for n in names:
        cal.record(n, lo)
    cal.stop()
    cal.start(2)
    for n in names:
        cal.record(n, hi)
    cal.stop()
    return cal


def fuser_for(mods, weights, cal):
    return EngagementFuser({m.name: m for m in mods}, weights, cal)


def test_single_modality_passes_through_normalized_score():
    eeg = StubModality("eeg", score=5.0, confidence=0.9)
    payload = fuser_for([eeg], {"eeg": 1.0}, calibrated(["eeg"])).get_engagement()
    assert payload["fused"] == pytest.approx(0.5)
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["sources"] == {"eeg": {"score": 0.5, "confidence": 0.9}}


def test_multi_modality_weighted_average():
    a, b = StubModality("a", score=10.0), StubModality("b", score=0.0)
    f = fuser_for([a, b], {"a": 0.6, "b": 0.4}, calibrated(["a", "b"]))
    assert f.get_engagement()["fused"] == pytest.approx(0.6)


def test_confidence_scales_each_modalitys_influence():
    a, b = StubModality("a", score=10.0, confidence=0.5), StubModality("b", score=0.0, confidence=1.0)
    f = fuser_for([a, b], {"a": 0.6, "b": 0.4}, calibrated(["a", "b"]))
    payload = f.get_engagement()
    assert payload["fused"] == pytest.approx((1.0 * 0.6 * 0.5) / (0.6 * 0.5 + 0.4 * 1.0))
    assert payload["confidence"] == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)


def test_zero_confidence_modality_is_excluded():
    a, b = StubModality("a", score=10.0, confidence=1.0), StubModality("b", score=0.0, confidence=0.0)
    f = fuser_for([a, b], {"a": 0.5, "b": 0.5}, calibrated(["a", "b"]))
    assert f.get_engagement()["fused"] == pytest.approx(1.0)


def test_all_zero_confidence_gives_none():
    a = StubModality("a", score=10.0, confidence=0.0)
    payload = fuser_for([a], {"a": 1.0}, calibrated(["a"])).get_engagement()
    assert payload["fused"] is None


def test_all_modalities_returning_none():
    a, b = StubModality("a"), StubModality("b")
    payload = fuser_for([a, b], {"a": 0.5, "b": 0.5}, calibrated(["a", "b"])).get_engagement()
    assert payload["fused"] is None
    assert payload["confidence"] is None
    assert payload["sources"] == {}


def test_partial_none_uses_only_modalities_with_output():
    a, b = StubModality("a", score=10.0), StubModality("b")
    payload = fuser_for([a, b], {"a": 0.5, "b": 0.5}, calibrated(["a", "b"])).get_engagement()
    assert payload["fused"] == pytest.approx(1.0)
    assert list(payload["sources"]) == ["a"]


def test_empty_modalities():
    payload = EngagementFuser({}, {}, Calibration([])).get_engagement()
    assert payload["fused"] is None
    assert payload["confidence"] is None
    assert payload["calibration"] == {"state": "idle", "e_min": None, "e_max": None}


def test_fused_is_null_before_calibration_but_confidence_is_reported():
    eeg = StubModality("eeg", score=5.0, confidence=0.8)
    payload = fuser_for([eeg], {"eeg": 1.0}, Calibration(["eeg"])).get_engagement()
    assert payload["fused"] is None
    assert payload["sources"]["eeg"] == {"score": None, "confidence": 0.8}
    assert payload["confidence"] == pytest.approx(0.8)
    assert payload["calibration"]["state"] == "idle"


def test_each_tick_feeds_raw_scores_to_calibration_while_a_phase_is_active():
    eeg = StubModality("eeg", score=2.0)
    cal = Calibration(["eeg"])
    f = fuser_for([eeg], {"eeg": 1.0}, cal)
    cal.start(1)
    f.get_engagement()
    eeg.output = ModalityOutput(1.5, 1.0)
    f.get_engagement()
    assert cal.e_min["eeg"] == 1.5


def test_payload_shape_and_json_serializable():
    eeg = StubModality("eeg", score=5.0)
    payload = fuser_for([eeg], {"eeg": 1.0}, calibrated(["eeg"], lo=0.0, hi=10.0)).get_engagement()
    assert set(payload) == {"fused", "confidence", "sources", "calibration", "timestamp"}
    assert payload["calibration"] == {"state": "complete", "e_min": 0.0, "e_max": 10.0}
    assert isinstance(payload["timestamp"], float)
    json.dumps(payload)
