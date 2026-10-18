import asyncio
import logging

from calibration import Calibration
from config import load_config
from fusion import EngagementFuser
from modalities.eeg_brainflow import EEGBrainflowModality
from websocket_server import EngagementServer


def build_modality(spec, config):
    if spec["type"] == "eeg":
        return EEGBrainflowModality(
            board_id=config.board_id,
            serial_port=config.serial_port,
            window_seconds=config.window_seconds,
            frontal_channel_indices=config.frontal_channel_indices,
        )
    raise ValueError(f"Unknown modality type: {spec['type']!r}")


async def main():
    try:
        config = load_config()
    except ValueError as exc:
        print(f"Config error: {exc}", flush=True)
        return
    modalities = {}
    weights = {}
    for spec in config.modalities:
        modality = build_modality(spec, config)
        modalities[modality.name] = modality
        weights[modality.name] = spec["weight"]

    for modality in modalities.values():
        await modality.start()
        if getattr(modality, "started", False):
            print(f"EEG started: board_id={modality.board_id}, channels={modality.eeg_channels}", flush=True)
        else:
            print(f"WARNING: {modality.name} did not start; running without a signal (see log)", flush=True)
    for name, weight in weights.items():
        print(f"Loaded modality: {name} (weight={weight:.2f})", flush=True)

    calibration = Calibration(list(modalities))
    fuser = EngagementFuser(modalities, weights, calibration)
    server = EngagementServer(fuser, calibration, host=config.ws_host, port=config.ws_port)

    print(f"Starting WebSocket server on ws://{config.ws_host}:{config.ws_port}", flush=True)
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()
        for modality in modalities.values():
            await modality.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
