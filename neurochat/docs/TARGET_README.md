# NeuroChat

A neuroadaptive AI tutor that reads real-time brain engagement signals from an EEG headset and silently injects them into every LLM prompt, so the model adapts its teaching style on the fly.

Built at the MIT Media Lab. Based on the paper: [NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences](https://doi.org/10.1145/3719160.3736623) (CUI '25).

---

## How It Works

NeuroChat forms a closed-loop system between the learner's brain and the LLM:

1. **EEG headset** captures brainwave data in real time
2. **Python backend** processes the signal (bandpass filter → Welch PSD → engagement index) and streams a normalized engagement score over WebSocket
3. **React frontend** silently injects `[Engagement: 0.72]` into every user message before sending it to the LLM — the user never sees this tag
4. **Ollama (local LLM)** reads the score and adapts its response:

| Engagement | Teaching Style |
|---|---|
| **Low (< 0.3)** | Narrative, storytelling, Socratic questions, curiosity hooks |
| **Moderate (0.3–0.6)** | Balanced explanations with examples and analogies |
| **High (> 0.6)** | Technical depth, bullet points, scientific precision |

The LLM never mentions the score, EEG, or brain signals to the user.

---

## Architecture

```
OpenBCI / Muse EEG Headset (USB serial or Bluetooth)
        │
        ▼
Python Backend (BrainFlow → DSP → engagement score)
        │ WebSocket (ws://localhost:8765)
        ▼
React Frontend (score injection → Ollama prompt)
        │ HTTP (localhost:11434/api/chat)
        ▼
Ollama (local LLM — qwen3.5:9b)
```

The frontend calls Ollama directly. The backend's only job is to compute and stream the engagement score.

---

## Project Structure

```
neurochat/
├── backend/
│   ├── modalities/
│   │   ├── base.py              # ModalityOutput dataclass + EngagementModality ABC
│   │   └── eeg_brainflow.py     # EEG signal processing via BrainFlow
│   ├── calibration.py           # Per-session E_min / E_max normalization
│   ├── fusion.py                # Confidence-weighted multi-modal fusion
│   ├── websocket_server.py      # Broadcasts engagement every 1s
│   ├── config.py                # Board type, serial port, weights
│   ├── main.py                  # Entry point
│   ├── pyproject.toml           # uv-managed dependencies
│   └── tests/
│       ├── test_calibration.py
│       ├── test_dsp.py
│       ├── test_eeg_modality.py
│       ├── test_fusion.py
│       ├── test_websocket.py
│       └── test_llm_adaptation.py
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── ChatInterface.jsx     # Streaming chat with score injection
    │   │   ├── CalibrationModal.jsx  # 2-phase calibration flow
    │   │   └── BrainWidget.jsx       # Live signal quality + engagement display
    │   ├── hooks/
    │   │   ├── useEngagementStream.js # WebSocket → engagement state
    │   │   └── useCalibration.js      # Calibration state machine
    │   ├── lib/
    │   │   ├── ollama.js             # NDJSON streaming fetch wrapper
    │   │   └── systemPrompt.js       # Adaptation rules for the LLM
    │   ├── test/
    │   │   ├── BrainWidget.test.jsx
    │   │   ├── CalibrationModal.test.jsx
    │   │   ├── ChatInterface.test.jsx
    │   │   ├── ollama.test.js
    │   │   └── systemPrompt.test.js
    │   └── App.jsx
    ├── vite.config.js
    └── package.json
```

---

## EEG Signal Processing Pipeline

The DSP pipeline in `eeg_brainflow.py` runs every 1 second over a 15-second sliding window:

1. **Bandpass filter** (1–30 Hz, Butterworth order 4) — retains relevant neural activity
2. **60 Hz notch filter** — removes power line interference
3. **Welch PSD** — extracts power spectral density
4. **Band power extraction:**
   - Theta (4–7 Hz) — relaxation, drowsiness
   - Alpha (7–11 Hz) — disengagement, passive states
   - Beta (11–20 Hz) — active focus, cognitive processing
5. **Engagement index** = β / (α + θ) — averaged over frontal channels (Fp1, Fp2)
6. **Confidence** = fraction of clean samples (|amplitude| < 100 µV)

After calibration, the raw score is normalized: `E_norm = (E - E_min) / (E_max - E_min)`, clipped to [0, 1].

### Calibration

Before each session, users complete two 2-minute tasks:
- **Phase 1 (Relaxation):** Eyes open, minimal cognition → establishes E_min
- **Phase 2 (Word Association):** Think of words chained by last letter → establishes E_max

---

## Test Suite

### Backend — 39 tests

Run with: `cd neurochat/backend && uv run pytest tests/ -v`

| Test File | Tests | What It Covers |
|---|---|---|
| `test_calibration.py` | 13 | Phase state machine (idle → phase1 → phase2 → complete), E_min/E_max recording, normalization with clipping, timeout auto-finalization, edge cases (equal bounds, unknown modality) |
| `test_dsp.py` | 5 | Bandpass filter (in-band preservation, out-of-band attenuation), 60 Hz notch filter removal, band power extraction across frequency ranges |
| `test_eeg_modality.py` | 5 | Graceful failure without hardware, insufficient buffer returns None, full pipeline with synthetic EEG data (mixed theta/alpha/beta), safe stop when not started |
| `test_fusion.py` | 6 | Single modality passthrough, multi-modal weighted average, confidence-based weighting (zero confidence excluded), all-None handling, partial-None, empty modalities |
| `test_websocket.py` | 3 | Server broadcasts engagement payload every second, calibration commands processed correctly, calibration state included in broadcast |
| `test_llm_adaptation.py` | 5 | Sends same question at scores 0.1, 0.5, 0.9 — verifies low uses questions/hooks, high uses structure/technical terms, high is longer than low, score never leaked to user. **Requires Ollama running.** |

### Frontend — 44 tests

Run with: `cd neurochat/frontend && npm test`

| Test File | Tests | What It Covers |
|---|---|---|
| `BrainWidget.test.jsx` | 9 | Connected/disconnected display, signal quality indicators (red/yellow/green thresholds at 0.4 and 0.7), engagement score rendering, null state handling |
| `CalibrationModal.test.jsx` | 8 | All states rendered correctly (idle/phase1/phase2/complete), countdown timer formatting (MM:SS), start button fires callback, word association example shown |
| `ChatInterface.test.jsx` | 13 | Disabled when uncalibrated, score injection into Ollama messages (`[Engagement: X.XX]`), engagement tag hidden from UI, freeze on input focus, unfreeze after send, streamed response display, error handling, system prompt included |
| `ollama.test.js` | 5 | NDJSON stream parsing, partial chunk buffering across reads, non-ok HTTP response throws, correct request format (model, stream, messages), malformed JSON lines skipped |
| `systemPrompt.test.js` | 9 | Prompt contains NeuroChat identity, engagement tag format, all three score tiers defined, never-mention-EEG instruction, Markdown instruction, unavailable score fallback, low/high engagement strategies present |

### Running All Tests

```bash
# Backend (fast — no Ollama needed)
cd neurochat/backend && uv run pytest tests/ -v --ignore=tests/test_llm_adaptation.py

# Backend (full — requires Ollama running)
cd neurochat/backend && uv run pytest tests/ -v

# Frontend
cd neurochat/frontend && npm test
```

---

## Quick Start (Without EEG Headset)

The backend defaults to a **synthetic BrainFlow board** for development, so you can run the full system without hardware.

### Prerequisites

- [Ollama](https://ollama.com) installed
- [uv](https://docs.astral.sh/uv/) installed (Python package manager)
- Node.js 18+

### 1. Pull the LLM model

```bash
ollama pull qwen3.5:9b
```

### 2. Start Ollama

```bash
OLLAMA_ORIGINS="*" ollama serve
```

### 3. Start the backend

```bash
cd neurochat/backend
uv run python main.py
```

You should see:
```
EEG started: board_id=-1, channels=[1, 2, ...]
Loaded modality: eeg (weight=1.00)
Starting WebSocket server on ws://0.0.0.0:8765
```

### 4. Start the frontend

```bash
cd neurochat/frontend
npm install
npm run dev
```

Open the URL from Vite (typically `http://localhost:5173`). Complete the calibration flow, then start chatting.

**Without the backend running**, the frontend degrades gracefully — BrainWidget shows "Backend not connected" and the chat works with `[Engagement: unavailable]` injected.

---

## Connecting a Real EEG Headset

### OpenBCI Cyton

1. Connect the Cyton board via USB dongle
2. Find the serial port:
   - macOS: `ls /dev/tty.usbserial-*`
   - Linux: `ls /dev/ttyUSB*`
   - Windows: Check Device Manager for COM port
3. Start the backend with hardware mode:

```bash
cd neurochat/backend
NEUROCHAT_BOARD=CYTON NEUROCHAT_SERIAL_PORT=/dev/tty.usbserial-XXXX uv run python main.py
```

### OpenBCI Ganglion

```bash
NEUROCHAT_BOARD=GANGLION NEUROCHAT_SERIAL_PORT=/dev/tty.usbserial-XXXX uv run python main.py
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEUROCHAT_BOARD` | `SYNTHETIC` | Board type: `SYNTHETIC`, `CYTON`, or `GANGLION` |
| `NEUROCHAT_SERIAL_PORT` | `/dev/ttyUSB0` | Serial port for the USB dongle |
| `NEUROCHAT_WS_PORT` | `8765` | WebSocket server port |

### Verifying Signal Quality

After connecting:
1. Check the **BrainWidget** (top-right corner) for connection status
2. Signal quality indicator: green (>70%), yellow (40–70%), red (<40%)
3. Complete the 4-minute calibration before chatting
4. If signal quality is low, check electrode contact and minimize movement

---

## Tech Stack

| Layer | Technology |
|---|---|
| EEG Interface | BrainFlow (supports OpenBCI, Muse, and 20+ boards) |
| Signal Processing | SciPy (Butterworth filters, Welch PSD) |
| Backend | Python 3.13, asyncio, websockets |
| Frontend | React 19, Vite |
| LLM | Ollama (qwen3.5:9b, local inference) |
| Package Management | uv (Python), npm (JavaScript) |
| Testing | pytest + pytest-asyncio (backend), Vitest + React Testing Library (frontend) |

---

## Future Plans

### Webcam Engagement Modality

Add a second engagement signal using webcam-based facial analysis via MediaPipe FaceMesh:

- **Eye aspect ratio** (both eyes averaged) → drowsiness detection
- **Head pose** (yaw + pitch magnitude) → on-screen attention tracking
- **Blink rate** over a 15-second window → fatigue estimation

The webcam score will be computed as: `score = 0.5 × attention + 0.3 × eye_openness + 0.2 × (1 − fatigue)`

The modality system is already designed for this — a new `modalities/webcam_mediapipe.py` implementing `EngagementModality` will plug directly into `EngagementFuser` with no changes to other files. The webcam will run at weight 0.4 alongside EEG at 0.6.

### Neural Network Fusion

Replace the current weighted-average fusion with a learned fusion model:

- **KernelFusion** — a small neural network trained on paired EEG + webcam data to predict a single engagement score that captures cross-modal interactions the linear formula misses
- Train on per-user calibration data to personalize the fusion function
- The `EngagementFuser` class is designed to be swapped out when this is ready

### Advanced Engagement Metrics

Extend beyond the basic β/(α+θ) engagement index:

- **Cognitive Load Index** — Theta Fz / Alpha Pz, an indicator of working memory demands
- **Alpha asymmetry** — difference in alpha power between hemispheres, linked to approach motivation and active engagement
- **Alpha peak frequency** — individual alpha frequency as an indicator of processing efficiency
- **Event-Related Potentials (ERPs)** — P300 for attentional allocation, N200 for conflict detection
- **Multi-band engagement** — incorporate gamma (30–50 Hz) for high-level cognitive binding

### Richer Adaptation Strategies

Based on findings from the user study (n=24):

- **User preference profiles** — some learners prefer structured/factual responses even at low engagement; allow configurable adaptation presets (e.g., "explorer" vs "textbook" mode)
- **Engagement trend detection** — adapt based on whether engagement is rising, falling, or stable over time, not just the current snapshot
- **Cognitive load awareness** — distinguish between productive engagement (learning) and overload (confusion), using complementary signals like response time and question complexity
- **Persistent learner memory** — track engagement patterns across sessions to build a long-term model of each learner's preferences

### Platform Expansion

- **Muse 2 support** — the paper's original hardware; connect via Web Bluetooth directly in the browser, eliminating the Python backend for consumer deployments
- **Cloud LLM support** — add OpenAI/Anthropic API backends alongside Ollama for users without local GPU
- **Mobile-friendly UI** — responsive layout for tablet use in classroom settings
- **Session analytics dashboard** — visualize engagement trends, topic coverage, and learning patterns over time
