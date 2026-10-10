"""
test_performance.py – Suite de tests de rendimiento para whisper.cpp.

Métricas medidas:
  - Latencia total (segundos de wall-clock por transcripción)
  - RTF  (Real-Time Factor = latencia / duración_audio)
        RTF < 1.0 → más rápido que tiempo real ✅
        RTF > 1.0 → más lento que tiempo real ⚠
  - CPU pico del proceso Docker (via docker stats)
  - RAM pico del contenedor    (via docker stats)

Uso:
    pytest tests/whisper/test_performance.py -v -s
"""

import time
import json
import subprocess
import threading
import statistics
import pytest
import httpx
import wave
import contextlib
from pathlib import Path
from typing import Optional

from tests.whisper.conftest import (
    WHISPER_BASE_URL,
    AUDIO_SAMPLES_DIR,
    ALL_SAMPLES,
    SPANISH_SAMPLES,
)

WHISPER_CONTAINER = "whisper_server"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_audio_duration_seconds(wav_path: Path) -> float:
    """Duración de un WAV en segundos."""
    with contextlib.closing(wave.open(str(wav_path), "r")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def transcribe_timed(client: httpx.Client, audio_path: Path, language: str) -> dict:
    """Transcribe y retorna texto + tiempo de wall-clock."""
    with open(audio_path, "rb") as f:
        t0 = time.perf_counter()
        response = client.post(
            "/inference",
            files={"file": (audio_path.name, f, "audio/wav")},
            data={"language": language, "response_format": "json"},
            timeout=120.0,
        )
        elapsed = time.perf_counter() - t0

    assert response.status_code == 200, (
        f"HTTP {response.status_code} para {audio_path.name}: {response.text}"
    )
    body = response.json()
    text = body.get("text", "").strip() or " ".join(
        seg.get("text", "") for seg in body.get("transcription", [])
    ).strip()
    return {"text": text, "elapsed_s": elapsed}


def get_container_stats(container: str) -> Optional[dict]:
    """
    Obtiene CPU% y RAM del contenedor usando `docker stats --no-stream`.
    Retorna None si Docker no está disponible o el contenedor no existe.
    """
    try:
        result = subprocess.run(
            [
                "docker", "stats", container,
                "--no-stream",
                "--format", '{"cpu":"{{.CPUPerc}}","mem":"{{.MemUsage}}","mem_pct":"{{.MemPerc}}"}',
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        raw = json.loads(line)
        cpu_str = raw["cpu"].replace("%", "").strip()
        mem_parts = raw["mem"].split("/")
        mem_used = mem_parts[0].strip()
        return {
            "cpu_pct": float(cpu_str) if cpu_str else 0.0,
            "mem_used": mem_used,
            "mem_pct": raw["mem_pct"].replace("%", "").strip(),
        }
    except Exception:
        return None


class ContainerMonitor(threading.Thread):
    """
    Hilo que sondea `docker stats` cada 0.5s durante una transcripción
    para capturar pico de CPU y RAM.
    """
    def __init__(self, container: str):
        super().__init__(daemon=True)
        self.container = container
        self.cpu_samples: list[float] = []
        self.mem_samples: list[str] = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            stats = get_container_stats(self.container)
            if stats:
                self.cpu_samples.append(stats["cpu_pct"])
                self.mem_samples.append(stats["mem_used"])
            time.sleep(0.5)

    def stop(self):
        self._stop_event.set()

    @property
    def peak_cpu(self) -> Optional[float]:
        return max(self.cpu_samples) if self.cpu_samples else None

    @property
    def avg_cpu(self) -> Optional[float]:
        return statistics.mean(self.cpu_samples) if self.cpu_samples else None

    @property
    def peak_mem(self) -> Optional[str]:
        return self.mem_samples[-1] if self.mem_samples else None


# ─── Tests individuales de latencia ──────────────────────────────────────────

@pytest.mark.parametrize("filename,reference,lang", ALL_SAMPLES)
def test_latency_per_sample(
    whisper_client: httpx.Client,
    audio_samples_dir: Path,
    filename: str,
    reference: str,
    lang: str,
):
    """
    Latencia < 60s y RTF razonable (< 5.0) para cada sample.
    En CPU con el modelo small el RTF típico es 0.3–0.8.
    """
    audio_path = audio_samples_dir / filename
    if not audio_path.exists():
        pytest.skip(f"Audio no encontrado: {audio_path}")

    duration_s = get_audio_duration_seconds(audio_path)
    result = transcribe_timed(whisper_client, audio_path, lang)
    elapsed = result["elapsed_s"]
    rtf = elapsed / duration_s if duration_s > 0 else float("inf")

    print(
        f"\n  {filename}  |  duración={duration_s:.1f}s  "
        f"latencia={elapsed:.2f}s  RTF={rtf:.2f}  "
        f"({'✅ RT' if rtf < 1 else '⚠ SLOW'})"
    )

    assert elapsed < 60, f"Latencia ({elapsed:.1f}s) > 60s para '{filename}'"
    assert rtf < 5.0, f"RTF ({rtf:.2f}) > 5.0 para '{filename}'"


# ─── Reporte consolidado de rendimiento ──────────────────────────────────────

def test_performance_report(
    whisper_client: httpx.Client,
    audio_samples_dir: Path,
):
    """
    Genera la tabla completa de rendimiento:
    latencia, RTF, CPU% y RAM por sample, más estadísticas agregadas.
    """
    rows = []

    for filename, reference, lang in ALL_SAMPLES:
        audio_path = audio_samples_dir / filename
        if not audio_path.exists():
            continue

        duration_s = get_audio_duration_seconds(audio_path)

        # Iniciar monitor de recursos del contenedor
        monitor = ContainerMonitor(WHISPER_CONTAINER)
        monitor.start()

        try:
            result = transcribe_timed(whisper_client, audio_path, lang)
        finally:
            monitor.stop()
            monitor.join(timeout=2)

        elapsed = result["elapsed_s"]
        rtf = elapsed / duration_s if duration_s > 0 else float("inf")

        rows.append({
            "filename": filename,
            "lang": lang,
            "duration_s": duration_s,
            "latency_s": elapsed,
            "rtf": rtf,
            "cpu_peak": monitor.peak_cpu,
            "cpu_avg": monitor.avg_cpu,
            "mem_peak": monitor.peak_mem,
        })

    if not rows:
        pytest.skip("No hay samples disponibles para el reporte de rendimiento.")

    # ── Estadísticas agregadas ─────────────────────────────────────────────────
    latencies = [r["latency_s"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    cpu_peaks = [r["cpu_peak"] for r in rows if r["cpu_peak"] is not None]

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_rtf = statistics.mean(rtfs)
    avg_cpu = statistics.mean(cpu_peaks) if cpu_peaks else None
    max_cpu = max(cpu_peaks) if cpu_peaks else None

    # ── Tabla de salida ────────────────────────────────────────────────────────
    sep = "─" * 85
    print(f"\n\n{'═'*85}")
    print(f"{'WHISPER.CPP (small) – REPORTE DE RENDIMIENTO':^85}")
    print(f"{'Contenedor: ' + WHISPER_CONTAINER + ' | CPU mode':^85}")
    print(f"{'═'*85}")
    print(f"  {'Sample':<28} {'Lang':^5} {'Dur':>5} {'Lat':>6} {'RTF':>5}  {'CPU%':>6}  {'RAM'}")
    print(sep)
    for r in rows:
        cpu_str = f"{r['cpu_peak']:.1f}%" if r["cpu_peak"] is not None else "  N/A"
        mem_str = r["mem_peak"] or "N/A"
        rtf_flag = "✅" if r["rtf"] < 1.0 else "⚠️ "
        print(
            f"  {r['filename']:<28} {r['lang']:^5} "
            f"{r['duration_s']:>4.1f}s {r['latency_s']:>5.2f}s {r['rtf']:>4.2f}{rtf_flag} "
            f"{cpu_str:>6}  {mem_str}"
        )
    print(sep)
    print(f"\n  📊 Estadísticas globales:")
    print(f"     Latencia promedio : {avg_latency:.2f}s")
    print(f"     Latencia p95      : {p95_latency:.2f}s")
    print(f"     RTF promedio      : {avg_rtf:.2f}  ({'✅ real-time' if avg_rtf < 1 else '⚠ slow'})")
    if avg_cpu is not None:
        print(f"     CPU promedio      : {avg_cpu:.1f}%")
        print(f"     CPU pico máx      : {max_cpu:.1f}%")
    else:
        print(f"     CPU/RAM           : (Docker stats no disponible en este entorno)")
    print(f"{'═'*85}\n")

    # Umbral: latencia promedio no debe superar 45s en CPU
    assert avg_latency < 45, (
        f"Latencia promedio ({avg_latency:.1f}s) supera umbral de 45s"
    )
