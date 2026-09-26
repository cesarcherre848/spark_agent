"""
src/agent_service/core/formatters/currency.py - Catálogo y formateador de monedas de Spark Agent.
"""

import re
from typing import Union, Optional, Dict

# Catálogo declarativo y extensible de símbolos de moneda
CURRENCY_SYMBOLS: Dict[str, str] = {
    "PEN": "S/.",
    "SOL": "S/.",
    "SOLES": "S/.",
    "NUEVO SOL": "S/.",
    "NUEVOS SOLES": "S/.",
    "S/": "S/.",
    "S/.": "S/.",
    "USD": "$",
    "DOLAR": "$",
    "DOLARES": "$",
    "EUR": "€",
    "EURO": "€",
    "EUROS": "€",
}


def get_currency_symbol(currency: Optional[str] = "PEN") -> str:
    """Retorna el símbolo oficial para la divisa especificada (default: S/.)."""
    if not currency:
        return "S/."
    key = str(currency).strip().upper()
    return CURRENCY_SYMBOLS.get(key, key)


def format_currency(
    amount: Union[float, int, str, None],
    currency: Optional[str] = "PEN",
) -> str:
    """Formatea un monto numérico con el símbolo oficial de divisa y 2 decimales.

    Ejemplos:
        format_currency(35, "PEN") -> "S/. 35.00"
        format_currency(1250.5, "PEN") -> "S/. 1,250.50"
        format_currency(50, "USD") -> "$ 50.00"
    """
    symbol = get_currency_symbol(currency)
    try:
        if amount is None:
            num = 0.0
        elif isinstance(amount, (float, int)):
            num = float(amount)
        else:
            cleaned = str(amount).replace(",", "").strip()
            num = float(cleaned)
    except (ValueError, TypeError):
        num = 0.0

    return f"{symbol} {num:,.2f}"


def _repl_amount_to_sol(match: re.Match) -> str:
    raw_val = match.group(1).replace(",", "").strip()
    try:
        num = float(raw_val)
        return f"S/. {num:,.2f}"
    except ValueError:
        return f"S/. {match.group(1)}"


def normalize_currencies_in_text(text: str) -> str:
    """Normaliza menciones heterogéneas de divisas peruanas hacia el formato estándar 'S/. XX.XX'.

    Corrige:
        - '$35.00 (PEN)' o '$ 35 (PEN)' -> 'S/. 35.00'
        - '35.00 PEN' o '35 PEN' -> 'S/. 35.00'
        - 'PEN 35.00' o 'PEN 35' -> 'S/. 35.00'
        - 'S/ 35.00' o 'S/ 35' (sin punto) -> 'S/. 35.00'
        - 'S/.35.00' (sin espacio) -> 'S/. 35.00'
    """
    if not text:
        return ""

    # 1. Patrón: $35.00 (PEN) o $35 (PEN) o $ 35.00 (SOL)
    res = re.sub(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*\((?:PEN|SOL|SOLES)\)",
        _repl_amount_to_sol,
        text,
        flags=re.IGNORECASE,
    )

    # 2. Patrón: PEN 35.00 o PEN 35
    res = re.sub(
        r"\bPEN\s+([\d,]+(?:\.\d+)?)\b",
        _repl_amount_to_sol,
        res,
        flags=re.IGNORECASE,
    )

    # 3. Patrón: 35.00 PEN o 35 PEN
    res = re.sub(
        r"\b([\d,]+(?:\.\d+)?)\s*PEN\b",
        _repl_amount_to_sol,
        res,
        flags=re.IGNORECASE,
    )

    # 4. Patrón: S/ 35.00 o S/ 35 (evitando S/. ya existente)
    res = re.sub(
        r"(?<![A-Za-z0-9])S/(?!\.)\s*([\d,]+(?:\.\d+)?)\b",
        _repl_amount_to_sol,
        res,
    )

    # 5. Patrón: S/.35.00 (sin espacio después del punto)
    res = re.sub(
        r"(?<![A-Za-z0-9])S/\.(?![\s\.])([\d,]+(?:\.\d+)?)\b",
        _repl_amount_to_sol,
        res,
    )

    return res
