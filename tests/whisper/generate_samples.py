"""
Script para generar audio samples WAV directamente desde texto usando gTTS.
Usa pydub si está disponible (necesita ffmpeg), o ffmpeg subprocess directamente.

Uso:
    python3 tests/whisper/generate_samples.py
"""

import io
import os
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    print("ERROR: Instala gtts primero: python3 -m pip install gtts")
    sys.exit(1)

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

OUTPUT_DIR = Path(__file__).parent / "audio_samples"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLES = [
    ("es_01_saludo.wav",
     "Hola, ¿cómo estás? Buenos días, que tengas un excelente día.",
     "es"),
    ("es_02_numeros.wav",
     "El número de teléfono es cero uno ocho tres cuatro cinco seis siete.",
     "es"),
    ("es_03_oracion_larga.wav",
     "El rápido zorro marrón saltó sobre el perro perezoso que dormía tranquilamente bajo el árbol.",
     "es"),
    ("es_04_nombres_propios.wav",
     "Me llamo Alejandro García y vivo en la Ciudad de México desde hace diez años.",
     "es"),
    ("es_05_negocio.wav",
     "Por favor, necesito una factura electrónica para esta compra. El total es doscientos cincuenta pesos.",
     "es"),
    ("es_06_preguntas.wav",
     "¿Cuándo llega el próximo vuelo a Buenos Aires? Necesito llegar antes de las seis de la tarde.",
     "es"),
    ("es_07_tecnico.wav",
     "La inteligencia artificial está transformando la manera en que las empresas procesan y analizan sus datos.",
     "es"),
    ("es_08_coloquial.wav",
     "Oye, ya viste la película? Está buenísima, te la recomiendo mucho, en serio.",
     "es"),
    ("en_01_control.wav",
     "The quick brown fox jumps over the lazy dog. Speech recognition works well in English.",
     "en"),
]


def mp3_bytes_to_wav(mp3_bytes: bytes, target_rate: int = 16000) -> bytes:
    """Convierte MP3 bytes → WAV 16kHz mono."""
    if HAS_PYDUB:
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        audio = audio.set_frame_rate(target_rate).set_channels(1).set_sample_width(2)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue()

    # Fallback: ffmpeg subprocess
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3f:
        mp3f.write(mp3_bytes)
        mp3_path = mp3f.name
    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", str(target_rate), "-ac", "1", "-sample_fmt", "s16",
             wav_path],
            check=True, capture_output=True
        )
        with open(wav_path, "rb") as wf:
            return wf.read()
    finally:
        os.unlink(mp3_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def generate_all():
    print(f"Generando {len(SAMPLES)} audio samples en: {OUTPUT_DIR}\n")
    for filename, text, lang in SAMPLES:
        wav_path = OUTPUT_DIR / filename
        if wav_path.exists():
            print(f"  ✓ {filename} ya existe, omitiendo.")
            continue
        print(f"  → Generando {filename} ({lang})...")
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_bytes = mp3_buf.getvalue()
        wav_bytes = mp3_bytes_to_wav(mp3_bytes)
        wav_path.write_bytes(wav_bytes)
        print(f"    ✓ Guardado ({len(wav_bytes)//1024} KB): {wav_path.name}")
    print("\n✅ Todos los samples generados.")


if __name__ == "__main__":
    generate_all()
