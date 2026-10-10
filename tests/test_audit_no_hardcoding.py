"""
tests/test_audit_no_hardcoding.py - Suite de Auditoría Automatizada Anti-Hardcoding.
Verifica que ningún archivo en 'src/' contenga nombres de clientes de prueba,
teléfonos fijos o heurísticas acopladas a personas.
"""

import os
import re
import pytest

SRC_DIR = "src"

FORBIDDEN_PATTERNS = [
    (
        "Nombres de clientes de prueba en código",
        re.compile(r"\b(silvana|dika|charito|renzo|miriam\s+pacheco)\b", re.IGNORECASE),
    ),
    (
        "Teléfonos específicos en código ejecutable",
        re.compile(r"\b(51975674799|51983689215|51953320946)\b"),
    ),
    (
        "Submarcas específicas de catálogo o notas hardcodeadas en lógica",
        re.compile(r"\b(soy\s+sexy|soy\s+glow|soy\s+única|soy\s+unica|soy\s+poderosa|mix\s*&\s*chic|peach\s+tonic|berry\s+tonic|l['’]essence)\b", re.IGNORECASE),
    ),
]


def test_audit_no_hardcoded_customer_data_in_src():
    """Escaneo estricto de auditoría: prohíbe que el código fuente en src/
    tenga nombres propios de personas o números de teléfono incrustados."""
    violations = []

    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if not f.endswith(".py"):
                continue
            file_path = os.path.join(root, f)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                for line_no, line in enumerate(fp, 1):
                    # Ignorar comentarios puros si no son código
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for label, pattern in FORBIDDEN_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            violations.append(
                                f"[{label}] {file_path}:{line_no} -> '{line.strip()}' (encontrado: '{match.group(0)}')"
                            )

    assert not violations, (
        f"Se encontraron {len(violations)} violaciones de hardcoding en src/:\n"
        + "\n".join(violations)
    )
