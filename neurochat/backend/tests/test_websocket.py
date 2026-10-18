import asyncio
import json
import time

import pytest
import pytest_asyncio
from websockets.asyncio.client import connect

from calibration import Calibration
from fusion import EngagementFuser
from modalities.base import EngagementModality, ModalityOutput
from websocket_server import EngagementServer, handle_command


class StubModality(EngagementModality):
    name = "eeg"

    async def start(self): ...

    async def stop(self): ...

    def get_output(self):
        return ModalityOutput(2.0, 0.9)


async def start_server(interval):
    cal = Calibration(["eeg"])
    fuser = EngagementFuser({"eeg": StubModality()}, {"eeg": 1.0}, cal)
    server = EngagementServer(fuser, cal, host="127.0.0.1", port=0, interval=interval)
    await server.start()
    return server, cal


@pytest_asyncio.fixture
async def served():
    server, cal = await start_server(interval=0.05)
    yield server, cal
    await server.stop()


async def recv_json(ws, timeout=2.0):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def recv_until(ws, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = await recv_json(ws, timeout)
        if predicate(payload):
            return payload
    raise AssertionError("condition never met in the broadcast")


async def test_payload_is_broadcast_with_calibration_state(served):
    server, _ = served
    async with connect(f"ws://127.0.0.1:{server.port}") as ws:
        payload = await recv_json(ws)
    assert set(payload) == {"fused", "confidence", "sources", "calibration", "timestamp"}
    assert payload["fused"] is None
    assert payload["calibration"] == {"state": "idle", "e_min": None, "e_max": None}
    assert payload["sources"]["eeg"]["confidence"] == 0.9


async def test_every_client_receives_the_broadcast(served):
    server, _ = served
    uri = f"ws://127.0.0.1:{server.port}"
    async with connect(uri) as a, connect(uri) as b:
        pa, pb = await asyncio.gather(recv_json(a), recv_json(b))
    assert pa["sources"] and pb["sources"]


async def test_default_interval_is_one_second_and_cadence_matches():
    cal = Calibration(["eeg"])
    fuser = EngagementFuser({"eeg": StubModality()}, {"eeg": 1.0}, cal)
    assert EngagementServer(fuser, cal).interval == 1.0

    server, _ = await start_server(interval=1.0)
    try:
        async with connect(f"ws://127.0.0.1:{server.port}") as ws:
            await recv_json(ws)
            t0 = time.monotonic()
            await recv_json(ws, timeout=3)
            assert 0.8 < time.monotonic() - t0 < 1.4
    finally:
        await server.stop()


async def test_calibration_commands_drive_the_state(served):
    server, cal = served
    async with connect(f"ws://127.0.0.1:{server.port}") as ws:
        await ws.send(json.dumps({"cmd": "start_calibration", "phase": 1}))
        await recv_until(ws, lambda p: p["calibration"]["state"] == "phase1")
        # While phase 1 runs, the fuser tick feeds raw scores into e_min.
        payload = await recv_until(ws, lambda p: p["calibration"]["e_min"] == 2.0)
        assert payload["fused"] is None

        await ws.send(json.dumps({"cmd": "stop_calibration"}))
        await recv_until(ws, lambda p: p["calibration"]["state"] == "idle")

        await ws.send(json.dumps({"cmd": "start_calibration", "phase": 2}))
        await recv_until(ws, lambda p: p["calibration"]["state"] == "phase2")
        await recv_until(ws, lambda p: p["calibration"]["e_max"] == 2.0)
        await ws.send(json.dumps({"cmd": "stop_calibration"}))
        done = await recv_until(ws, lambda p: p["calibration"]["state"] == "complete")
    assert done["calibration"]["e_min"] == 2.0
    assert cal.state == "complete"


async def test_bad_commands_are_ignored_and_the_connection_survives(served):
    server, cal = served
    async with connect(f"ws://127.0.0.1:{server.port}") as ws:
        for raw in (
            "not json at all",
            "[1, 2]",
            '"just a string"',
            json.dumps({"cmd": "explode"}),
            json.dumps({"cmd": "start_calibration", "phase": 7}),
            json.dumps({"cmd": "start_calibration", "phase": True}),
            json.dumps({"cmd": "start_calibration"}),
            b"\xff\xfe",
        ):
            await ws.send(raw)
        payload = await recv_json(ws)
        assert payload["calibration"]["state"] == "idle"
        # Still alive and still accepting valid commands afterwards.
        await ws.send(json.dumps({"cmd": "start_calibration", "phase": 1}))
        await recv_until(ws, lambda p: p["calibration"]["state"] == "phase1")
    assert cal.state == "phase1"


def test_handle_command_return_values():
    cal = Calibration(["eeg"])
    assert handle_command('{"cmd": "start_calibration", "phase": 2}', cal) is True
    assert cal.state == "phase2"
    assert handle_command('{"cmd": "stop_calibration"}', cal) is True
    assert handle_command("{oops", cal) is False
    assert handle_command('{"cmd": "nope"}', cal) is False
    assert handle_command('{"cmd": "start_calibration", "phase": 0}', cal) is False
