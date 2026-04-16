#A real time neuroadaptive system that adjusts LLM responses using EEG-derived engagement scores.

Overview:
- Processes EEG signals in real time
- Computes an engagement score from frequency bands
- Continuously updates a rolling score
- Injects that score into OpenWebUI prompts to adapt LLM responses

Files:
- eeg_engagement.py: Processes EEG signals, extracts frequency bands (alpha/beta/theta), computes engagement score, maintains rolling buffer, and handles calibration and normalization.
- engagement_runtime.py: Initializes the system, runs calibration, starts the background EEG thread, and exposes get_score(), freeze(), and unfreeze() APIs.
- eeg_streamer.py: Continuously feeds EEG data into the system by calling scorer.update() every second; currently uses synthetic data but can be replaced with real hardware input.
- neurochat_engagement.py: Reads the latest engagement score on each user message and injects adaptive instructions into the LLM system prompt.
- docker-compose.yml: Defines and connects services (Ollama, OpenWebUI, Pipelines), sets networking and ports, and runs the full system with a single command.

Running (Terminal Test)
in terminal: python -c "import engagement_runtime; import time; time.sleep(25)"
Expected output:
- Calibration values
- Stream start message
- Continuous engagement scores

Running (Full System)
Docker desktop must be running
download desired model: ex: docker exec -it ollama ollama pull qwen3.5:9b
in terminal: docker compose up -d
Then open: http://localhost:3000

Notes
- Synthetic EEG is used by default
- Replace eeg_streamer.py with real EEG input for hardware use
- Modeled from https://dl.acm.org/doi/10.1145/3719160.3736623
