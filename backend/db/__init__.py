"""Database setup: engine, session factory, declarative base."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    """Build a dialect-appropriate engine.

    SQLite (local dev): check_same_thread=False so worker threads can share it.
    PostgreSQL (production): a connection pool with pre-ping to survive dropped
    connections (e.g. across container restarts).
    """
    url = settings.database_url
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=settings.db_pool_pre_ping,
            future=True,
        )
    return create_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=settings.db_pool_pre_ping,
        future=True,
    )


engine = _make_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Ensure the schema exists.

    Production uses Alembic migrations (run by the container entrypoint), so this
    is a no-op there. In development/test we create tables directly for
    convenience. Either way the data directory is ensured.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.is_production:
        return
    # Import models so they register on Base.metadata before create_all.
    from models import video  # noqa: F401
    from models import event  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope for use inside background jobs/scripts."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# FastAPI dependency
def db_dependency() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
