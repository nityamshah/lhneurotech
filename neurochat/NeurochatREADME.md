# NeuroChat

A neuroadaptive AI tutor. It reads real-time engagement from an EEG headset, turns it into a normalized 0–1 score, and silently prepends that score to every message sent to a local LLM, so the model adapts its teaching style. The user never sees the tag in the chat, and the model is told never to mention it.

Based on: Baradari et al. (2025), [NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences](https://doi.org/10.1145/3719160.3736623), ACM CUI '25. The paper is in `docs/neurochat_paper.pdf`.

By default it runs on BrainFlow's **synthetic board**, so you can run everything without hardware. The synthetic signal is fake and nearly flat, so its score carries no meaning (see [Known limitations](#known-limitations)).

---

## How it works

1. **EEG source.** An OpenBCI Cyton or Ganglion (or the synthetic board) streams raw EEG through BrainFlow.
2. **Python backend.** Every second it filters the last 15 s, cuts it into 1 s epochs (250 ms step), takes an FFT per epoch, computes the engagement index per epoch, and averages it. It normalizes the result with the user's own calibration range and broadcasts it over WebSocket (`ws://localhost:8765`).
3. **React frontend.** It freezes the score when the input is focused and sends `[Engagement: 0.72]\n<your text>` to Ollama. Only your text is shown in the chat.
4. **Ollama (`qwen3.5:9b`, local).** A fixed system prompt maps the score to a teaching style:

| Engagement | Teaching style |
|---|---|
| **Low (< 0.3)** | Narrative, storytelling, curiosity hooks, a Socratic question |
| **Moderate (0.3–0.6)** | Balanced explanation with examples and analogies |
| **High (> 0.6)** | Technical depth, bullet points, scientific precision |

If the backend is down, chat still works and the tag is `[Engagement: unavailable]` (the model uses a balanced style).

```
EEG headset (Cyton / Ganglion over USB) or BrainFlow synthetic board
        │
        ▼
Python backend   (BrainFlow → DSP → calibration → fusion → WebSocket)
        │  ws://localhost:8765
        ▼
React frontend   (score → hidden tag injected into the user message)
        │  HTTP localhost:11434/api/chat
        ▼
Ollama           (local LLM, qwen3.5:9b)
```

Three native processes: Ollama, the backend, the frontend. The frontend calls Ollama directly; the backend only computes and streams the score.

---

## Project structure

```
neurochat/
├── CLAUDE.md                    # project spec + "as built" notes
├── NeurochatREADME.md           # this file
├── docs/                        # paper, target README, reference DSP
├── backend/
│   ├── modalities/
│   │   ├── base.py              # ModalityOutput + EngagementModality ABC
│   │   └── eeg_brainflow.py     # BrainFlow session + all DSP
│   ├── calibration.py           # E_min / E_max per modality, phase state machine, normalize()
│   ├── fusion.py                # EngagementFuser (confidence-weighted)
│   ├── websocket_server.py      # 1 s broadcast + inbound calibration commands
│   ├── config.py                # env-driven settings
│   ├── main.py                  # wires everything together
│   ├── pyproject.toml           # uv-managed, pinned to Python 3.13
│   └── tests/
└── frontend/
    ├── src/
    │   ├── components/  ChatInterface.jsx  CalibrationModal.jsx  BrainWidget.jsx
    │   ├── hooks/       useEngagementStream.js  useCalibration.js
    │   ├── lib/         ollama.js  systemPrompt.js
    │   ├── test/
    │   ├── App.jsx  main.jsx
    ├── index.html  vite.config.js  package.json
```

---

## EEG signal processing

Implemented in `backend/modalities/eeg_brainflow.py`, following the paper's section 3.2. It runs every second over a 15 s sliding window:

1. **Bandpass** 1–30 Hz (Butterworth, order 4, zero-phase)
2. **Notch** at 60 Hz
3. **Epochs:** 1 s long, 250 ms step (rounded to whole samples, e.g. 62 at 250 Hz)
4. **FFT per epoch** (Hann window, one-sided PSD)
5. **Band power** per epoch, integrated over the band: theta 4–7 Hz, alpha 7–11 Hz, beta 11–20 Hz
6. **Engagement index** per epoch = β / (α + θ + 1e-9)
7. **Score** = mean over all epochs in the window, averaged over the frontal channels (the first two channels, Fp1 and Fp2 on the default Cyton montage)

`confidence` is the fraction of samples with |amplitude| < 100 µV (measured on the bandpassed signal). The backend emits nothing until it has buffered a full 15 s.

### Calibration

Before chatting, the user completes two 2-minute tasks:

- **Phase 1, relaxation:** sit still and clear your mind. The lowest score seen becomes `E_min`.
- **Phase 2, word association:** think of a word, then a new word starting with its last letter. The highest score seen becomes `E_max`.

Calibration scores use the paper's shorter 10 s window. After calibration, live scores are normalized with `E_norm = (E − E_min) / (E_max − E_min)`, clipped to [0, 1] (0.5 if the bounds don't span). Each phase also ends on its own after 120 s, and the frontend moves from phase 1 to phase 2 automatically. Calibration lives in backend memory: restarting the backend asks you to calibrate again.

### Freeze

When you click into the message box the score is frozen at its latest live value, so the model sees your engagement while you read the previous reply, not while you type. It is released once the reply has finished streaming.

---

## Test suite

### Backend: 75 tests (+5 that need Ollama)

```powershell
cd backend
uv run pytest tests/ -v
```

| Test file | Tests | What it covers |
|---|---|---|
| `test_calibration.py` | 25 | Phase state machine, per-phase min/max, timeouts (fake clock), restarts, normalization and clipping, equal bounds, unknown modality |
| `test_dsp.py` | 19 | Bandpass and notch behaviour, epoching, per-epoch FFT band power (units, band edges), per-epoch engagement, ratio-then-average |
| `test_eeg_modality.py` | 13 | Failure without hardware, safe `stop()`, `None` until 15 s, full pipeline on mixed theta/alpha/beta, confidence, 10 s calibration score, stream gaps, a real synthetic-board smoke run |
| `test_fusion.py` | 12 | Passthrough, weighted average, confidence weighting, zero confidence, all/partial/empty `None`, calibration recording |
| `test_websocket.py` | 6 | Broadcast payload and cadence, calibration commands, bad commands ignored |
| `test_llm_adaptation.py` | 5 | Same question at scores 0.1 / 0.5 / 0.9 and "unavailable": structure, length, no score leakage. **Needs Ollama; deselected by default.** |

Run the Ollama test with `uv run pytest -m llm -rs`. Its assertions check non-deterministic text, so it can occasionally fail.

### Frontend: 86 tests

```powershell
cd frontend
npm test
```

| Test file | Tests | What it covers |
|---|---|---|
| `BrainWidget.test.jsx` | 11 | Connected/disconnected text, red/yellow/green thresholds (0.4, 0.7), null handling |
| `CalibrationModal.test.jsx` | 12 | All states, MM:SS formatting, Start callback, word-association example |
| `ChatInterface.test.jsx` | 15 | Tag injected but never shown, system prompt, streaming, errors, locked until calibrated, unavailable tag, focus and send callbacks |
| `ollama.test.js` | 7 | NDJSON parsing, split chunks, non-OK throws, request shape, malformed lines, `thinking` ignored |
| `systemPrompt.test.js` | 10 | Identity, tag format, all three tiers, fallback, never-mention rule, Markdown |
| `useEngagementStream.test.jsx` | 12 | Reconnect, freeze/unfreeze, null when disconnected, malformed messages, cleanup |
| `useCalibration.test.jsx` | 15 | State mapping, countdown, auto-advance, single phase-2 start and stop, resets |
| `App.test.jsx` | 4 | Modal gating, freeze-on-focus end to end, offline and dropped-backend behaviour |

macOS / Linux: the same commands work (`cd backend && uv run pytest tests/ -v`, `cd frontend && npm test`).

---

## Quick start (no headset)

### Prerequisites

- [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), and Node.js (the test tooling needs `^22.22`, `^24.15` or newer; developed on 24).
- On Windows PowerShell, if `npm` is blocked with "running scripts is disabled", run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 1. Pull the model

```powershell
ollama pull qwen3.5:9b
```

### 2. Start Ollama with CORS enabled

The browser can't call Ollama until `OLLAMA_ORIGINS` allows it.

```powershell
$env:OLLAMA_ORIGINS="*"; ollama serve
```

On Windows, Ollama often auto-starts in the background without this setting, so `ollama serve` reports that port 11434 is already in use. Either quit Ollama from the system tray and rerun the command above, or set the variable permanently, then quit Ollama from the tray and reopen it from the Start menu:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", "*", "User")
```

macOS / Linux: `OLLAMA_ORIGINS="*" ollama serve`

### 3. Start the backend

```powershell
cd backend
uv run python main.py
```

You should see:

```
EEG started: board_id=-1, channels=[1, 2, 3, ...]
Loaded modality: eeg (weight=1.00)
Starting WebSocket server on ws://0.0.0.0:8765
```

### 4. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173/`).

### 5. What you should see

1. A full-screen **Calibration** modal and a top-right widget saying "Connected". The chat box is locked ("Complete calibration to begin.").
2. After about 15 s the widget shows `Signal: 100%`.
3. Click **Start**: 2:00 of "Relax", then 2:00 of word association. The modal then closes and chat unlocks.
4. Chat. In the browser devtools (Network tab), the `chat` request shows `[Engagement: 0.xx]` at the start of your message; the page never does.

**Without the backend**, the widget says "Backend not connected" and chat works with `[Engagement: unavailable]`.

---

## Talking to the WebSocket directly

```powershell
cd backend
uv run python -m websockets ws://localhost:8765
```

A payload arrives every second:

```json
{
  "fused": 0.72,
  "confidence": 0.91,
  "sources": { "eeg": { "score": 0.72, "confidence": 0.91 } },
  "calibration": { "state": "complete", "e_min": 0.4, "e_max": 2.1 },
  "timestamp": 1730000000.0
}
```

`fused` and `sources.*.score` are `null` until calibration is complete; `confidence` is reported earlier. Type any of these commands (malformed or unknown input is ignored):

```json
{"cmd": "start_calibration", "phase": 1}
{"cmd": "start_calibration", "phase": 2}
{"cmd": "stop_calibration"}
```

Stopping early is a quick way to test calibration without waiting 4 minutes.

---

## Connecting a real headset

### OpenBCI Cyton / Ganglion

1. Plug in the USB dongle.
2. Find the serial port: Windows, Device Manager (`COM3`, `COM4`, …); macOS, `ls /dev/tty.usbserial-*`; Linux, `ls /dev/ttyUSB*`.
3. Start the backend with hardware mode:

```powershell
cd backend
$env:NEUROCHAT_BOARD="CYTON"; $env:NEUROCHAT_SERIAL_PORT="COM3"; uv run python main.py
```

macOS / Linux:

```bash
cd backend
NEUROCHAT_BOARD=CYTON NEUROCHAT_SERIAL_PORT=/dev/tty.usbserial-XXXX uv run python main.py
```

Use `GANGLION` for the Ganglion. If the board can't connect, the backend logs a warning and keeps running without a signal.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEUROCHAT_BOARD` | `SYNTHETIC` | `SYNTHETIC`, `CYTON` or `GANGLION` |
| `NEUROCHAT_SERIAL_PORT` | `COM3` on Windows, `/dev/ttyUSB0` elsewhere | Serial port for the USB dongle (ignored for `SYNTHETIC`) |
| `NEUROCHAT_WS_PORT` | `8765` | Backend WebSocket port |
| `VITE_WS_URL` | `ws://localhost:8765` | Frontend: where to find the backend (set before `npm run dev`) |

### Checking signal quality

The widget's signal indicator is green above 70%, yellow from 40% to 70%, red below 40%. If it is low, check electrode contact and minimize movement.

---

## Known limitations

- **The synthetic board is not meaningful data.** Each channel is a pure single-frequency sine with no beta, so the score is nearly constant (raw values around 0.046, varying by about 0.05%). After calibration the normalized score hovers near 0.5 or jumps between 0 and 1. Real variation needs real hardware.
- **Network exposure.** The backend binds `0.0.0.0` with no Origin check, so any web page or device on your network can read the stream and send calibration commands.
- **Calibration records noisy windows too.** One artifact (such as a blink) can set `E_min` or `E_max`.
- **State lives in memory.** Calibration resets when the backend restarts, and chat history is lost on page reload. Reloading mid-calibration restarts the countdown display (the backend's own timeout still ends the phase).
- **15 s warm-up.** The backend emits nothing until a full window is buffered.

## Differences from the paper

Same: band ranges, the index β/(α+θ), 1–30 Hz bandpass and 60 Hz notch, 1 s epochs with a 250 ms step and an FFT per epoch, a 15 s window updated every second, a 10 s calibration window, 2-minute relaxation and word-association calibration, min/max normalization, a hidden tag in the prompt, and freezing the score when the user goes to type.

Different:

- **Hardware and architecture.** The paper used a Muse 2 over Web Bluetooth in a browser-only app. This uses OpenBCI boards through BrainFlow and a Python backend.
- **Channels.** The paper doesn't say which channels feed the index. This averages the first two channels.
- **Unspecified details chosen here:** zero-phase order-4 filtering, a Hann window, trapezoid-integrated band power, and a step rounded to whole samples.
- **Min and max.** The paper is ambiguous (relaxation sets the minimum and word association the maximum, or the lowest and highest across both tasks). This uses per-phase values. Scores are also clipped to [0, 1], which the paper doesn't mention.
- **Where calibration is stored.** The paper uses browser session storage. This uses backend memory.
- **Freeze trigger.** The paper freezes when typing starts. This freezes on input focus.
- **Visible score.** The widget shows the engagement score to the user. The paper hid EEG metrics from participants.
- **Model and prompt.** The paper used GPT-4-turbo with its own cognitive-load-framed prompt. This uses local `qwen3.5:9b` and a fixed prompt with 0.3 / 0.6 tiers.
- **Not built from the paper:** Mood Mode and Debug Mode toggles, the chat-history sidebar, EEG and chat export, connect/disconnect/recalibrate controls, and the study protocol.

---

## Tech stack

| Layer | Technology |
|---|---|
| EEG interface | BrainFlow (OpenBCI Cyton and Ganglion, synthetic board) |
| Signal processing | NumPy, SciPy (Butterworth filters, FFT, trapezoid integration) |
| Backend | Python 3.13 (pinned with uv), asyncio, websockets |
| Frontend | React 19, Vite, react-markdown |
| LLM | Ollama (`qwen3.5:9b`, local) |
| Testing | pytest + pytest-asyncio; Vitest + React Testing Library + jsdom |

## Not built (future work)

A webcam engagement modality, neural-network fusion, richer engagement metrics, adaptation presets and trend detection, cloud LLM backends, and session analytics. See `docs/TARGET_README.md` for the longer list. Adding a modality is described in the "as built" section of `CLAUDE.md`.
