"""Per-modality E_min / E_max calibration and score normalization.

State machine: idle -> phase1 -> phase2 -> complete.
Phase 1 (relaxation) keeps the lowest raw score as e_min; phase 2 (word association) keeps the
highest as e_max. Each phase finalizes on stop() or after `phase_seconds`. When phase 1 ends the
state returns to idle with e_min set and e_max unset ("phase 1 done, waiting for phase 2").
"""

import math
import time

IDLE = "idle"
PHASE1 = "phase1"
PHASE2 = "phase2"
COMPLETE = "complete"

PHASE_SECONDS = 120


class Calibration:
    def __init__(self, modality_names, phase_seconds=PHASE_SECONDS, clock=time.monotonic):
        self._names = list(modality_names)
        self._phase_seconds = phase_seconds
        self._clock = clock
        self._state = IDLE
        self._deadline = None
        self._e_min = {name: None for name in self._names}
        self._e_max = {name: None for name in self._names}

    @property
    def state(self):
        self._check_timeout()
        return self._state

    @property
    def e_min(self):
        return dict(self._e_min)

    @property
    def e_max(self):
        return dict(self._e_max)

    def start(self, phase):
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase!r}")
        self._check_timeout()
        if self._state in (PHASE1, PHASE2):
            self._finalize()
        if phase == 1:
            self._e_min = {name: None for name in self._names}
            self._e_max = {name: None for name in self._names}
            self._state = PHASE1
        else:
            self._e_max = {name: None for name in self._names}
            self._state = PHASE2
        self._deadline = self._clock() + self._phase_seconds

    def stop(self):
        self._check_timeout()
        if self._state in (PHASE1, PHASE2):
            self._finalize()

    def record(self, modality_name, raw_score):
        self._check_timeout()
        if self._state not in (PHASE1, PHASE2) or modality_name not in self._e_min:
            return
        if not math.isfinite(raw_score):
            return
        if self._state == PHASE1:
            current = self._e_min[modality_name]
            self._e_min[modality_name] = raw_score if current is None else min(current, raw_score)
        else:
            current = self._e_max[modality_name]
            self._e_max[modality_name] = raw_score if current is None else max(current, raw_score)

    def normalize(self, score, modality_name):
        if self.state != COMPLETE or modality_name not in self._e_min:
            return None
        lo, hi = self._e_min[modality_name], self._e_max[modality_name]
        if lo is None or hi is None:
            return None
        if hi <= lo:
            return 0.5
        return min(1.0, max(0.0, (score - lo) / (hi - lo)))

    def _check_timeout(self):
        if self._state in (PHASE1, PHASE2) and self._clock() >= self._deadline:
            self._finalize()

    def _finalize(self):
        if self._state == PHASE2 and all(
            self._e_min[n] is not None and self._e_max[n] is not None for n in self._names
        ):
            self._state = COMPLETE
        else:
            self._state = IDLE
        self._deadline = None
