"""
src/agent_service/core/guardrails/sku_grounding.py - Marco Determinista de Validación y Blindaje Anti-Alucinación de SKUs.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SKUGroundingResult:
    """Resultado de la auditoría determinista de grounding de SKUs."""
    is_valid: bool
    mentioned_skus: List[str] = field(default_factory=list)
    valid_skus: List[str] = field(default_factory=list)
    invalid_skus: List[str] = field(default_factory=list)
    critique_message: Optional[str] = None
    suggested_improvements: List[str] = field(default_factory=list)
    replacement_map: Dict[str, str] = field(default_factory=dict)


class SKUExtractor:
    """Extractor multi-patrón de códigos SKU y menciones de producto en texto libre,
    con filtrado estricto de falsos positivos (precios, cantidades, edades, SPF, órdenes)."""

    # Patrones específicos de captura de SKU
    SKU_PATTERNS = [
        re.compile(r"\[(?:SKU|sku|Código|código|Ref|ref)[:\s]+([A-Za-z0-9_-]+)\]", re.IGNORECASE),
        re.compile(r"\[([A-Za-z0-9_-]+)\]"),
        re.compile(r"\b(?:SKU|sku|código|ref)[:\s]+#?([A-Za-z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\b([0-9]{3,7})\b"),  # Códigos numéricos de 3 a 7 dígitos (ej: 150, 2037, 79934)
    ]

    # Patrones de exclusión (Falsos positivos)
    EXCLUSION_PATTERNS = [
        re.compile(r"S/\.?\s*\d+(?:\.\d{2})?"),             # Precios en soles S/. 40.00
        re.compile(r"\b\d+\s*(?:soles|dólares|usd|pen)\b", re.IGNORECASE),
        re.compile(r"\b(?:S00\d+|SO\d+|G00\d+)\b", re.IGNORECASE),  # Códigos de orden S00015
        re.compile(r"\b\d+\s*(?:años|año|meses|días)\b", re.IGNORECASE),  # 50 años, 32 años
        re.compile(r"\b(?:SPF|FPS)\s*\d+\b", re.IGNORECASE),        # SPF 30, SPF 100
        re.compile(r"\b\d+\s*(?:g|gr|gramos|ml|oz|fl\.?\s*oz|unidades?|und|unid|pza|piezas?)\b", re.IGNORECASE),  # 100 g, 50 ml
        re.compile(r"\b\d+x\b", re.IGNORECASE),                     # 1x, 2x
        re.compile(r"\b(?:\+?51)?9\d{8}\b"),                       # Números telefónicos
        re.compile(r"\bPág\.?\s*\d+\b", re.IGNORECASE),            # Menciones de página
        re.compile(r"\b(?:C-?\d{1,2}|Campaña\s*\d{1,2})\b", re.IGNORECASE),  # Campaña C10
        re.compile(r"\b(?:N[°ºo.]?|Nro\.?)\s*\d+\b", re.IGNORECASE),        # Numeración de brochas/tonos: N° 101, N° 201
        re.compile(r"\b\d+\s*%\b"),                                         # Porcentajes: 100%, 50%
    ]

    @classmethod
    def extract_skus(cls, text: str) -> List[str]:
        """Extrae de forma exhaustiva y limpia todos los códigos SKU mencionados en el texto."""
        if not text:
            return []

        # 1. Enmascarar exclusiones para no capturar precios, cantidades ni órdenes
        masked_text = text
        for exc_pat in cls.EXCLUSION_PATTERNS:
            masked_text = exc_pat.sub(" ", masked_text)

        found_candidates: Set[str] = set()

        # 2. Extraer menciones con los patrones
        for pat in cls.SKU_PATTERNS:
            matches = pat.findall(masked_text)
            for m in matches:
                clean_m = str(m).strip()
                # Filtrar códigos demasiado cortos o palabras comunes
                if len(clean_m) >= 2 and not clean_m.lower() in {"sku", "ref", "pag", "c10", "c11", "c12", "c15"}:
                    found_candidates.add(clean_m)

        # 3. También buscar patrones tipo [150] en el texto original (si no es precio)
        bracket_nums = re.findall(r"\[([A-Za-z0-9_-]+)\]", text)
        for b in bracket_nums:
            b_clean = b.strip()
            if b_clean.isdigit() and int(b_clean) > 0 and len(b_clean) <= 7:
                found_candidates.add(b_clean)
            elif re.match(r"^(?:SKU:\s*)?([A-Za-z0-9_-]+)$", b_clean, re.IGNORECASE):
                sku_val = re.sub(r"^SKU:\s*", "", b_clean, flags=re.IGNORECASE).strip()
                if sku_val:
                    found_candidates.add(sku_val)

        return sorted(list(found_candidates))


class SKUGroundingValidator:
    """Validador determinista de códigos SKU contra los productos autorizados en catálogo."""

    @classmethod
    def build_authorized_sku_index(
        cls, candidate_products: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Construye un índice de SKUs autorizados a partir de los productos candidatos del turno,
        incluyendo las variantes/tonos oficiales de la página del catálogo (catalog_variants)."""
        index: Dict[str, Dict[str, Any]] = {}
        for p in candidate_products:
            if isinstance(p, dict):
                sku = str(p.get("sku") or "").strip()
                if sku:
                    index[sku] = p
                # También indexar default_code si viene
                d_code = str(p.get("default_code") or "").strip()
                if d_code:
                    index[d_code] = p
                # Indexar todas las variantes/tonos de la página del catálogo
                for cv in p.get("catalog_variants") or []:
                    if isinstance(cv, dict) and cv.get("sku"):
                        cv_sku = str(cv["sku"]).strip()
                        if cv_sku and cv_sku not in index:
                            index[cv_sku] = p
        return index

    @classmethod
    def validate_draft_grounding(
        cls,
        draft_response: str,
        candidate_products: List[Dict[str, Any]],
        strict_mode: bool = True,
    ) -> SKUGroundingResult:
        """Audita el borrador contra los productos candidatos autorizados.
        Si detecta SKUs no autorizados, rechaza el borrador y genera retroalimentación de corrección."""
        if not draft_response:
            return SKUGroundingResult(is_valid=True)

        authorized_index = cls.build_authorized_sku_index(candidate_products)
        authorized_skus = set(authorized_index.keys())

        # Si no hay candidatos con SKU en el turno, no podemos validar grounding estricto
        if not authorized_skus:
            return SKUGroundingResult(is_valid=True)

        extracted_skus = SKUExtractor.extract_skus(draft_response)
        if not extracted_skus:
            return SKUGroundingResult(is_valid=True)

        invalid_skus: List[str] = []
        valid_skus: List[str] = []
        replacement_map: Dict[str, str] = {}

        for sku in extracted_skus:
            if sku in authorized_skus:
                valid_skus.append(sku)
            else:
                invalid_skus.append(sku)

        if not invalid_skus:
            return SKUGroundingResult(
                is_valid=True,
                mentioned_skus=extracted_skus,
                valid_skus=valid_skus,
            )

        # Construir tabla de reemplazo inteligente buscando por similitud de nombre
        for inv_sku in invalid_skus:
            # Buscar si el número coincide con la página de algún producto autorizado
            for auth_sku, prod in authorized_index.items():
                p_page = str(prod.get("pagina") or "")
                if p_page and p_page == inv_sku:
                    replacement_map[inv_sku] = auth_sku
                    break

        critique_parts = [
            f"ALUCINACIÓN DE SKUs DETECTADA: Se encontraron códigos inválidos o inexistentes en el borrador: {invalid_skus}."
        ]
        suggestions = []

        if replacement_map:
            critique_parts.append("TABLA DE REEMPLAZO OBLIGATORIO:")
            for inv_s, real_s in replacement_map.items():
                prod_name = authorized_index.get(real_s, {}).get("name", "Producto")
                critique_parts.append(f"- El código '{inv_s}' es el número de página; el SKU oficial real para '{prod_name}' es [{real_s}].")
                suggestions.append(f"Reemplazar [{inv_s}] por [{real_s}] para '{prod_name}'")

        critique_parts.append(
            f"Los únicos SKUs válidos y autorizados en el catálogo son: {sorted(list(authorized_skus))}. "
            f"Corrige la respuesta inmediatamente sustituyendo los códigos erróneos por los SKUs reales."
        )

        return SKUGroundingResult(
            is_valid=False,
            mentioned_skus=extracted_skus,
            valid_skus=valid_skus,
            invalid_skus=invalid_skus,
            critique_message="\n".join(critique_parts),
            suggested_improvements=suggestions or [f"Usar exclusivamente los SKUs válidos: {sorted(list(authorized_skus))}"],
            replacement_map=replacement_map,
        )


class SKUFailSafeSanitizer:
    """Sanitizador final previo al canal: neutraliza o corrige cualquier SKU no autorizado."""

    @classmethod
    def sanitize_text(
        cls,
        text: str,
        authorized_skus: Set[str],
        replacement_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """Sustituye códigos erróneos por sus reemplazos o limpia corchetes espurios."""
        if not text:
            return text

        repl = replacement_map or {}
        sanitized = text

        # 1. Aplicar reemplazos explícitos
        for inv_sku, real_sku in repl.items():
            # Reemplazar [SKU: inv_sku] o [inv_sku] por [real_sku]
            sanitized = re.sub(
                rf"\[(?:SKU:\s*)?{re.escape(inv_sku)}\]",
                f"[{real_sku}]",
                sanitized,
                flags=re.IGNORECASE,
            )

        # 2. Para cualquier [SKU: X] o [X] que siga siendo inválido y no esté en authorized_skus
        if authorized_skus:
            def replace_unknown_bracket(match: re.Match) -> str:
                inner = match.group(1).strip()
                clean_code = re.sub(r"^SKU:\s*", "", inner, flags=re.IGNORECASE).strip()
                if clean_code in authorized_skus:
                    return f"[{clean_code}]"
                if clean_code.isdigit() and len(clean_code) <= 7 and clean_code not in authorized_skus:
                    # Es un número inventado o página; remover el corchete engañoso
                    return ""
                return match.group(0)

            sanitized = re.sub(r"\[([A-Za-z0-9_:\s-]+)\]", replace_unknown_bracket, sanitized)
            # Limpiar posibles dobles espacios resultantes
            sanitized = re.sub(r"[ ]{2,}", " ", sanitized)

        return sanitized
