"""Environment-driven settings for the NeuroChat backend."""

import os
import sys

from brainflow.board_shim import BoardIds

BOARD_IDS = {
    "SYNTHETIC": int(BoardIds.SYNTHETIC_BOARD),
    "CYTON": int(BoardIds.CYTON_BOARD),
    "GANGLION": int(BoardIds.GANGLION_BOARD),
}

WINDOW_SECONDS = 15
FRONTAL_CHANNEL_INDICES = [0, 1]  # Fp1, Fp2 on the default Cyton montage
ACTIVE_MODALITIES = [{"type": "eeg", "weight": 1.0}]
WS_HOST = "0.0.0.0"


def _default_serial_port() -> str:
    return "COM3" if sys.platform == "win32" else "/dev/ttyUSB0"


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.board_name = env.get("NEUROCHAT_BOARD", "SYNTHETIC").upper()
        if self.board_name not in BOARD_IDS:
            raise ValueError(
                f"Unknown NEUROCHAT_BOARD {self.board_name!r}; expected one of {', '.join(BOARD_IDS)}"
            )
        self.board_id = BOARD_IDS[self.board_name]
        self.serial_port = env.get("NEUROCHAT_SERIAL_PORT", _default_serial_port())
        self.ws_port = int(env.get("NEUROCHAT_WS_PORT", "8765"))
        self.ws_host = WS_HOST
        self.window_seconds = WINDOW_SECONDS
        self.frontal_channel_indices = list(FRONTAL_CHANNEL_INDICES)
        self.modalities = [dict(m) for m in ACTIVE_MODALITIES]


def load_config(env=None) -> Config:
    return Config(env)
