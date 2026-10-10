"""
Fixtures compartidas para la suite de pruebas de whisper.cpp.
"""

from pathlib import Path
import pytest
import httpx

# ─── Constantes ───────────────────────────────────────────────────────────────
WHISPER_BASE_URL = "http://localhost:9001"
AUDIO_SAMPLES_DIR = Path(__file__).parent / "audio_samples"

# Referencia: (nombre_archivo, transcripción_exacta, idioma)
SPANISH_SAMPLES = [
    ("es_01_saludo.wav",
     "Hola, cómo estás? Buenos días, que tengas un excelente día.",
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
]

ENGLISH_SAMPLES = [
    ("en_01_control.wav",
     "The quick brown fox jumps over the lazy dog. Speech recognition works well in English.",
     "en"),
]

ALL_SAMPLES = SPANISH_SAMPLES + ENGLISH_SAMPLES


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def whisper_client() -> httpx.Client:
    """Cliente HTTP reutilizable para toda la sesión de tests."""
    with httpx.Client(base_url=WHISPER_BASE_URL, timeout=90.0) as client:
        yield client


@pytest.fixture(scope="session")
def audio_samples_dir() -> Path:
    """Directorio de audio samples; falla si no existen los archivos."""
    if not AUDIO_SAMPLES_DIR.exists() or not any(AUDIO_SAMPLES_DIR.glob("*.wav")):
        pytest.skip(
            "No se encontraron audio samples. "
            "Ejecuta primero: python tests/whisper/generate_samples.py"
        )
    return AUDIO_SAMPLES_DIR
