"""Database engine and session factory.

Supports PostgreSQL (asyncpg) and SQLite (aiosqlite) via the same async API.
Config priority: POSTGRES_URL env var → SQLite fallback at db/dragon_engine.db.
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.pool import NullPool

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_DB_DIR = Path(__file__).resolve().parent
_DEFAULT_SQLITE = str(_DB_DIR / "dragon_engine.db")


def _build_db_url() -> tuple[str, str]:
    """Return (url, dialect_label)."""
    from shared.configs.settings import get_settings

    settings = get_settings()
    explicit = settings.db_url
    if explicit:
        if "postgres" in explicit and "asyncpg" not in explicit:
            explicit = explicit.replace("postgresql://", "postgresql+asyncpg://", 1)
        return explicit, "postgresql" if "postgres" in explicit else "sqlite"

    pg_url = settings.postgres_url
    if pg_url:
        # Convert postgresql:// → postgresql+asyncpg://
        if pg_url.startswith("postgresql://"):
            pg_url = pg_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif pg_url.startswith("postgres://"):
            pg_url = pg_url.replace("postgres://", "postgresql+asyncpg://", 1)
        return pg_url, "postgresql"

    sqlite_url = f"sqlite+aiosqlite:///{_DEFAULT_SQLITE}"
    return sqlite_url, "sqlite"


_db_url, _dialect = _build_db_url()

logger.info("[db] dialect=%s url=%s", _dialect, _db_url.split("@")[-1] if "@" in _db_url else _db_url)

_connect_args: dict = {}
_poolclass = None

if _dialect == "sqlite":
    _connect_args = {"check_same_thread": False}
    _poolclass = NullPool

engine = create_async_engine(
    _db_url,
    echo=False,
    pool_pre_ping=True,
    connect_args=_connect_args,
    poolclass=_poolclass,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """Yield an async session — use as FastAPI dependency or context manager."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables (idempotent). Call once at startup."""
    from db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("[db] tables created/verified (dialect=%s)", _dialect)


async def close_db() -> None:
    """Dispose engine. Call at shutdown."""
    await engine.dispose()
    logger.info("[db] engine disposed")
