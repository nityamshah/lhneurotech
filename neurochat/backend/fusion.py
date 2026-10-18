import time


class EngagementFuser:
    def __init__(self, modalities: dict, weights: dict, calibration):
        self.modalities = modalities
        self.weights = weights
        self.calibration = calibration

    def get_engagement(self) -> dict:
        sources = {}
        weighted_score = 0.0
        weighted_conf = 0.0  # sum of weight * confidence over modalities with a normalized score
        conf_sum = 0.0
        weight_sum = 0.0

        for name, modality in self.modalities.items():
            output = modality.get_output()
            if output is None:
                continue
            weight = self.weights.get(name, 0.0)
            self.calibration.record(name, output.score)
            normalized = self.calibration.normalize(output.score, name)
            sources[name] = {"score": normalized, "confidence": output.confidence}

            conf_sum += weight * output.confidence
            weight_sum += weight
            if normalized is not None:
                weighted_score += normalized * weight * output.confidence
                weighted_conf += weight * output.confidence

        fused = weighted_score / weighted_conf if weighted_conf > 0 else None
        confidence = conf_sum / weight_sum if weight_sum > 0 else None

        # The calibration state is read last so a phase that just timed out is reported correctly.
        state = self.calibration.state
        e_min, e_max = self._primary_bounds()
        return {
            "fused": fused,
            "confidence": confidence,
            "sources": sources,
            "calibration": {"state": state, "e_min": e_min, "e_max": e_max},
            "timestamp": time.time(),
        }

    def _primary_bounds(self):
        """The payload carries scalar bounds: those of the first modality."""
        for name in self.modalities:
            return self.calibration.e_min.get(name), self.calibration.e_max.get(name)
        return None, None
