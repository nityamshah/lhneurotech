from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModalityOutput:
    score: float  # RAW engagement index (normalization happens in calibration/fusion)
    confidence: float  # signal quality 0-1
    features: np.ndarray | None = field(default=None)  # reserved for future fusion
    # Raw score over the paper's shorter calibration window; None means "calibrate on `score`".
    calibration_score: float | None = None


class EngagementModality(ABC):
    name: str

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    def get_output(self) -> ModalityOutput | None: ...
