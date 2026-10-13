# Nyra‑AI Architecture Overview

**Nyra‑AI** is a personal voice assistant built on top of the open‑source
[RealtimeSTT](https://github.com/koljab/RealtimeSTT) library for low‑latency
speech‑to‑text, a local LLM powered by Ollama (`llama3.2:3b`), and a handful of
Python utilities for web lookup, file I/O, and system information.

The project is intentionally lightweight – the core runtime lives in a single
`main.py` script that orchestrates the components.

---

## Directory layout

```
nyra-ai/
├─ assets/                  # Pre‑trained .onnx models for Piper TTS
│  └─ en_GB‑…onx and .json
├─ Config/                  # Runtime configuration values
│  └─ Configurations.py
├─ src/                     # Runtime code (mostly wrappers around libraries)
│  ├─ app.py                # Entry‑point used by the server build
│  ├─ TranscriptionEngine/  # Simple wrapper that exposes RealtimeSTT APIs
│  │  └─ transcribe.py
│  ├─ OllamaEngine/         # Helper that streams Ollama responses
│  │  └─ OllamaInference.py
│  └─ main.py               # Voice‑assistant application loop
├─ Tools/                   # User‑defined tools (search, file ops, etc.)
│  └─ UserTools.py
├─ pyproject.toml            # Project metadata & dependencies
├─ README.md                 # (currently empty – use this file)
├─ run_server.py (?)         # Optional FastAPI server wrapper (not shipped)
└─ …                        # other configuration & lock files
```

| Folder | Purpose |
|--------|---------|
| `assets/` | The ONNX models used by [Piper TTS](https://github.com/rhasspy/piper) for speech synthesis. |
| `Config/` | Holds default name of assistant, model and system prompt. |
| `src/` | Implementation details – audio recording, transcription, LLM inference. |
| `Tools/` | Exposed to the LLM as "tools"; includes a web searcher (`ddgs`), file utilities, and a system‑time helper. |
| `pyproject.toml` | Describes dependencies, Python version, and the build configuration. |

---

## Core flow

1. **Voice capture** – `RealtimeSTT.AudioToTextRecorder` runs in a background thread. It emits raw audio chunks, runs VAD, forwards speech to the chosen transcription engine (`faster_whisper` in our default config), and returns text via the `recorder.text()` method.

2. **Processing loop** – `main.py` continuously polls the recorder. When a user speaks, the gathered text is appended to a LLM prompt (the system prompt from `Configurations.py`).

3. **LLM inference** – `src/OllamaEngine/OllamaInference.py` sends the prompt to a local Ollama instance. Streaming tokens are printed to the console and also queued for speech synthesis.

4. **Text‑to‑speech** – `src/main.py` uses `piper` (loaded with the assets in `assets/`) to synthesize the LLM response. The output is streamed through a `sounddevice.RawOutputStream` to the default audio output device.

5. **Tools** – If the LLM emits a tool call, `src/main.py` dispatches to the appropriate function in `Tools/UserTools.py`. The result is fed back into the LLM conversation as a *tool* message.

---

## Dependencies

```toml
[project]
name = "nyra-ai"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "ddgs>=9.16.0",          # duckduckgo search API
    "ollama>=0.6.3",          # local LLM
    "openwakeword==0.3.0",    # optional wake‑word support
    "piper-tts>=1.8.0",       # offline TTS engine
    "psutil>=7.2.2",          # system information
    "realtimestt[faster-whisper,porcupine]>=1.1.2",  # real‑time ASR
    "sounddevice>=0.5.6",      # audio capture & playback
    "gpuinfo",                # optional GPU info
]
```

All packages are installed by `uv sync` (recommended) or via `pip install .`.

---

## Extending the assistant

| Area | How to add |
|------|------------|
| **New voice model** | Replace the `.onnx` file in `assets/` and update `src/main.py` accordingly. |
| **New tool** | Implement a function in `Tools/UserTools.py`, add it to the `TOOLS` dictionary in `main.py`, and invoke it from the LLM. |
| **Different LLM** | Swap the model string in `Config/Configurations.py` to a local or remote Ollama model. |
| **Different transcription engine** | Change the `transcription_engine` or `model` arguments passed to `AudioToTextRecorder`. |

---

## Running the assistant

```bash
# 1. Create a virtual environment (recommended)
uv venv                # or: python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv sync                # or: pip install .

# 3. Make sure the local Ollama server is running
ollama serve

# 4. Start the assistant
python src/main.py
```

You should hear a startup greeting and the assistant will start listening once a pause in speech longer than 1.5 s occurs.

---

## Key files

| File | Purpose |
|------|---------|
| `src/main.py` | Main event loop; orchestrates audio, transcription, inference, and speech synthesis. |
| `src/OllamaEngine/OllamaInference.py` | Handles streaming LLM responses and maintains the conversation history. |
| `src/TranscriptionEngine/transcribe.py` | Thin wrapper around `RealtimeSTT` that exposes `recorder.text()`. |
| `Tools/UserTools.py` | Set of helpers available to the LLM (`web_search`, `save_to_file`, etc.). |
| `Config/Configurations.py` | Defines global constants like `MODEL` and `SYSTEM_PROMPT`. |

---

> **Note** – The repository currently does not ship a FastAPI web‑socket server; the server can be added by creating an entry‑point that forwards the same logic in an ASGI application.
