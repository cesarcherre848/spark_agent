"""
src/agent_service/api/whatsapp/audio_transcriber.py - Pipeline de transcripción OGG/Opus/MP3 -> WAV -> Whisper STT
"""

import asyncio
import io
import logging
import os
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_MIME_PREFIXES = (
    "audio/ogg",
    "audio/opus",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/wav",
    "audio/x-wav",
    "audio/m4a",
    "audio/x-m4a",
    "audio/aac",
)


def get_whisper_server_url() -> str:
    """Obtiene la URL base del servidor whisper desde variables de entorno."""
    return os.getenv("WHISPER_SERVER_URL", "http://localhost:9001").rstrip("/")


async def ogg_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """Convierte cualquier flujo de audio (OGG/Opus, MP3, AAC) a WAV 16kHz mono PCM16 en memoria usando ffmpeg."""
    if not audio_bytes:
        raise ValueError("El búfer de audio está vacío.")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i", "pipe:0",
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            "-f", "wav",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        wav_bytes, stderr = await proc.communicate(input=audio_bytes)
        if proc.returncode != 0:
            error_msg = stderr.decode(errors="replace")[:300]
            logger.error(f"[AudioTranscriber] Error en ffmpeg (código {proc.returncode}): {error_msg}")
            raise RuntimeError(f"Error procesando audio con ffmpeg: {error_msg}")
        return wav_bytes
    except FileNotFoundError:
        logger.error("[AudioTranscriber] 'ffmpeg' no está instalado en el sistema.")
        raise RuntimeError("ffmpeg no está disponible para la conversión de audio.")


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
    language: str = "es",
    whisper_url: Optional[str] = None,
) -> str:
    """Pipeline completo: valida formato, convierte a WAV 16kHz si es necesario y envía a Whisper.

    Args:
        audio_bytes: Bytes del archivo de audio descargado.
        mime_type: Tipo MIME reportado por Meta (ej: 'audio/ogg; codecs=opus').
        language: Código ISO de idioma (default: 'es').
        whisper_url: URL base opcional de whisper-server.

    Returns:
        Texto transcrito en string.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes no puede estar vacío.")

    base_mime = (mime_type or "").split(";")[0].strip().lower()
    is_supported = any(base_mime.startswith(prefix) or prefix.startswith(base_mime) for prefix in SUPPORTED_AUDIO_MIME_PREFIXES)

    if not is_supported and base_mime:
        logger.warning(f"[AudioTranscriber] Tipo MIME '{mime_type}' no estándar. Se intentará conversión con ffmpeg.")

    # Convertir a WAV 16kHz PCM16 a menos que ya sea WAV con cabecera
    if base_mime in ("audio/wav", "audio/x-wav") and audio_bytes.startswith(b"RIFF"):
        wav_bytes = audio_bytes
    else:
        logger.info(f"[AudioTranscriber] Convirtiendo {base_mime or 'audio'} ({len(audio_bytes)} bytes) a WAV 16kHz...")
        wav_bytes = await ogg_to_wav_bytes(audio_bytes)

    target_url = (whisper_url or get_whisper_server_url()).rstrip("/")
    inference_endpoint = f"{target_url}/inference"

    logger.info(f"[AudioTranscriber] Enviando audio ({len(wav_bytes)} bytes) a {inference_endpoint} (lang={language})...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            inference_endpoint,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language": language, "response_format": "json"},
        )
        if response.status_code != 200:
            logger.error(f"[AudioTranscriber] Error HTTP de Whisper ({response.status_code}): {response.text}")
            response.raise_for_status()

    body = response.json()
    # Whisper.cpp devuelve {"text": "..."} o {"transcription": [{"text": "..."}, ...]}
    text = body.get("text", "").strip()
    if not text and "transcription" in body:
        segments = body.get("transcription", [])
        text = " ".join(seg.get("text", "") for seg in segments if isinstance(seg, dict)).strip()

    logger.info(f"[AudioTranscriber] Transcripción completada: '{text[:100]}...'")
    return text
