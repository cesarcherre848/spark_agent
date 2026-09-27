#!/usr/bin/env python3
"""
scripts/generate_missing_embeddings.py

Sincroniza y genera embeddings faltantes para productos en PostgreSQL (product_vector_embedding)
utilizando el modelo oficial 'BAAI/bge-m3' (1024 dimensiones).
"""

import sys
import os
import json
import hashlib
import logging
import psycopg
from typing import Dict, Any, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(".env.qa")
load_dotenv(".env.dev")

from src.agent_service.config.database import get_database_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_missing_embeddings")


def clean_odoo_translatable(val: Any) -> str:
    """Extrae la traducción al español o inglés de campos JSON de Odoo."""
    if not val:
        return ""
    if isinstance(val, dict):
        return str(val.get("es_PE") or val.get("es_ES") or val.get("en_US") or next(iter(val.values()), "")).strip()
    if isinstance(val, str) and val.startswith("{"):
        try:
            d = json.loads(val)
            if isinstance(d, dict):
                return str(d.get("es_PE") or d.get("es_ES") or d.get("en_US") or next(iter(d.values()), "")).strip()
        except Exception:
            pass
    return str(val).strip()


def build_source_text(name: str, desc_sale: str, desc: str) -> str:
    """Construye el texto estándar para vectorización léxica y semántica."""
    clean_name = clean_odoo_translatable(name)
    clean_desc = clean_odoo_translatable(desc_sale) or clean_odoo_translatable(desc)
    lines = [f"Producto: {clean_name}"]
    if clean_desc:
        lines.append(f"Descripción: {clean_desc}")
    return "\n".join(lines)


def sync_missing_embeddings(
    ai_provider: str = "huggingface",
    ai_model: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> int:
    """Detecta e inserta embeddings para productos que carecen de ellos."""
    ds = get_database_settings()
    logger.info(f"Conectando a base de datos {ds.dbname} en {ds.host}:{ds.port}...")

    with psycopg.connect(ds.conninfo) as conn:
        with conn.cursor() as cur:
            # 1. Obtener productos faltantes
            cur.execute("""
                SELECT 
                    pp.id AS product_id,
                    pp.product_tmpl_id,
                    pt.name,
                    pt.description_sale,
                    pt.description
                FROM product_product pp
                JOIN product_template pt ON pt.id = pp.product_tmpl_id
                WHERE pp.id NOT IN (
                    SELECT DISTINCT product_id 
                    FROM product_vector_embedding 
                    WHERE ai_provider = %s AND ai_model = %s AND active = TRUE
                )
                ORDER BY pp.id;
            """, (ai_provider, ai_model))

            missing_records = cur.fetchall()
            total_missing = len(missing_records)
            logger.info(f"Productos sin embeddings detectados: {total_missing}")

            if total_missing == 0:
                logger.info("El catálogo está 100% vectorizado. No hay productos pendientes.")
                return 0

            # 2. Inicializar modelo de embeddings
            logger.info(f"Cargando modelo de embeddings: {ai_model} ({ai_provider})...")
            embeddings_service = HuggingFaceEmbeddings(
                model_name=ai_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # 3. Procesar en lotes
            inserted_count = 0
            for i in range(0, total_missing, batch_size):
                batch = missing_records[i : i + batch_size]
                texts_to_embed: List[str] = []
                batch_metadata: List[Dict[str, Any]] = []

                for row in batch:
                    product_id, tmpl_id, name_raw, desc_sale_raw, desc_raw = row
                    source_text = build_source_text(name_raw, desc_sale_raw, desc_raw)
                    clean_name = clean_odoo_translatable(name_raw)
                    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

                    texts_to_embed.append(source_text)
                    batch_metadata.append({
                        "product_id": product_id,
                        "product_tmpl_id": tmpl_id,
                        "name": clean_name,
                        "source_text": source_text,
                        "content_hash": content_hash,
                    })

                # Generar vectores
                vectors = embeddings_service.embed_documents(texts_to_embed)

                # Insertar en base de datos
                for meta, vec in zip(batch_metadata, vectors):
                    vec_str = f"[{','.join(map(str, vec))}]"
                    cur.execute("""
                        INSERT INTO product_vector_embedding (
                            product_tmpl_id,
                            product_id,
                            dimension,
                            name,
                            source_text,
                            content_hash,
                            ai_provider,
                            ai_model,
                            embedding_1024,
                            active,
                            create_uid,
                            write_uid,
                            create_date,
                            write_date
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, TRUE, 1, 1, NOW(), NOW()
                        );
                    """, (
                        meta["product_tmpl_id"],
                        meta["product_id"],
                        len(vec),
                        meta["name"],
                        meta["source_text"],
                        meta["content_hash"],
                        ai_provider,
                        ai_model,
                        vec_str,
                    ))
                    inserted_count += 1

                conn.commit()
                logger.info(f"Progreso: {inserted_count}/{total_missing} productos insertados.")

            logger.info(f"Completado exitosamente: {inserted_count} productos vectorizados.")
            return inserted_count


if __name__ == "__main__":
    count = sync_missing_embeddings()
    sys.exit(0)
