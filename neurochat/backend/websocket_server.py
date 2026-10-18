"""WebSocket server: broadcasts the engagement payload and accepts calibration commands."""

import asyncio
import contextlib
import json
import logging

from websockets import broadcast
from websockets.asyncio.server import serve

logger = logging.getLogger(__name__)


def handle_command(raw, calibration) -> bool:
    """Apply one inbound command. Bad input is logged and ignored, never raised."""
    try:
        msg = json.loads(raw)
    except ValueError:
        logger.warning("Ignoring malformed JSON command: %r", raw)
        return False
    if not isinstance(msg, dict):
        logger.warning("Ignoring non-object command: %r", msg)
        return False

    cmd = msg.get("cmd")
    if cmd == "start_calibration":
        phase = msg.get("phase")
        if isinstance(phase, int) and not isinstance(phase, bool) and phase in (1, 2):
            calibration.start(phase)
            return True
        logger.warning("Ignoring start_calibration with invalid phase: %r", phase)
        return False
    if cmd == "stop_calibration":
        calibration.stop()
        return True
    logger.warning("Ignoring unknown command: %r", cmd)
    return False


class EngagementServer:
    def __init__(self, fuser, calibration, host="0.0.0.0", port=8765, interval=1.0):
        self.fuser = fuser
        self.calibration = calibration
        self.host = host
        self.port = port
        self.interval = interval
        self._clients = set()
        self._server = None
        self._task = None

    async def start(self):
        self._server = await serve(self._handler, self.host, self.port)
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self):
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for message in websocket:
                handle_command(message, self.calibration)
        finally:
            self._clients.discard(websocket)

    async def _broadcast_loop(self):
        # Runs even with no clients connected: each tick also feeds calibration.
        while True:
            try:
                broadcast(self._clients, json.dumps(self.fuser.get_engagement()))
            except Exception:
                logger.exception("Failed to build or send the engagement payload")
            await asyncio.sleep(self.interval)
