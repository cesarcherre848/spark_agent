import os
from typing import Optional
from dataclasses import dataclass
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv(".env.dev")


@dataclass
class DatabaseSettings:
    host: str = os.getenv("PG_HOST") or os.getenv("ODOO_PG_HOST", "134.199.209.31")
    port: str = os.getenv("PG_PORT") or os.getenv("ODOO_PG_PORT", "5432")
    user: str = os.getenv("PG_USER") or os.getenv("ODOO_PG_USER", "spark_erp_app")
    password: str = os.getenv("PG_PASSWORD") or os.getenv("ODOO_PG_PASSWORD", "Sp4rk_Erp#2026!kL")
    dbname: str = os.getenv("PG_DATABASE") or os.getenv("ODOO_PG_DATABASE", "spark_erp_db_dev")
    connect_timeout: int = 5

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname} "
            f"connect_timeout={self.connect_timeout}"
        )


def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


_GLOBAL_DB_POOL: Optional[AsyncConnectionPool] = None


def get_db_pool(
    min_size: int = 1,
    max_size: int = 10,
    conninfo: Optional[str] = None,
) -> AsyncConnectionPool:
    """Retorna la instancia global compartida de AsyncConnectionPool (singleton)."""
    global _GLOBAL_DB_POOL
    if _GLOBAL_DB_POOL is None:
        info = conninfo or get_database_settings().conninfo
        _GLOBAL_DB_POOL = AsyncConnectionPool(
            conninfo=info,
            min_size=min_size,
            max_size=max_size,
            open=False,
        )
    return _GLOBAL_DB_POOL


async def close_db_pool() -> None:
    """Cierra el pool global de conexiones si está abierto."""
    global _GLOBAL_DB_POOL
    if _GLOBAL_DB_POOL is not None:
        await _GLOBAL_DB_POOL.close()
        _GLOBAL_DB_POOL = None
