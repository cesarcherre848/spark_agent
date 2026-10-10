#!/bin/sh
# whisper-entrypoint.sh
# Descarga el modelo ggml-small.bin al volumen si no existe,
# luego arranca el servidor whisper.cpp.
set -e

MODEL_PATH="/models/ggml-small.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[whisper-entrypoint] Modelo no encontrado. Descargando ggml-small.bin (~488 MB)..."
    wget -q -O "$MODEL_PATH" "$MODEL_URL" \
        && echo "[whisper-entrypoint] Descarga completada." \
        || { echo "[whisper-entrypoint] ERROR: descarga fallida."; rm -f "$MODEL_PATH"; exit 1; }
else
    echo "[whisper-entrypoint] Modelo ya existe en volumen, omitiendo descarga."
fi

echo "[whisper-entrypoint] Iniciando whisper-server en 0.0.0.0:9000..."
exec /app/build/bin/whisper-server \
    --model    "$MODEL_PATH" \
    --host     "0.0.0.0" \
    --port     "9000" \
    --language "es"
