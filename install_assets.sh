#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VOICES=("en_GB-cori-medium")        # add more here, e.g. "en_GB-cori-high"
ASSETS_DIR="assets"
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

echo "==> System packages"
if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev portaudio19-dev libportaudio2 curl
else
  echo "Not apt-based: install a C compiler, Python headers and PortAudio dev files yourself."
fi

echo "==> uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Python dependencies"
uv sync

echo "==> Piper voices"
mkdir -p "$ASSETS_DIR"
for voice in "${VOICES[@]}"; do
  # en_GB-cori-medium  ->  en/en_GB/cori/medium/en_GB-cori-medium
  IFS='-' read -r locale name quality <<< "$voice"
  lang="${locale%%_*}"
  url="$HF_BASE/$lang/$locale/$name/$quality/$voice"
  for ext in onnx onnx.json; do
    file="$ASSETS_DIR/$voice.$ext"
    if [ -f "$file" ]; then
      echo "have $file"
    else
      echo "downloading $file"
      curl -L --fail -o "$file" "$url.$ext"
    fi
  done
done

echo "==> Whisper model (otherwise it downloads on first run)"
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small.en')"

echo "==> Ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
MODEL=$(uv run python -c "from Config.Configurations import MODEL; print(MODEL)")
ollama pull "$MODEL"

echo "Done."