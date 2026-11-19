"""Engine, sessões e utilitários de persistência idempotente."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from fachadaguard.config import get_settings
from fachadaguard.domain.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn: Any, _: Any) -> None:  # pragma: no cover - trivial
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def init_engine(url: str | None = None) -> Engine:
    global _engine, _session_factory
    _engine = make_engine(url or get_settings().database_url)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    return _engine or init_engine()


def create_schema(engine: Engine | None = None, attempts: int = 30) -> None:
    """Cria tabelas de forma segura com vários serviços subindo juntos (advisory lock no PostgreSQL)."""
    engine = engine or get_engine()
    for attempt in range(attempts):
        try:
            if engine.dialect.name == "postgresql":
                with engine.begin() as conn:
                    conn.execute(text("SELECT pg_advisory_xact_lock(724001)"))
                    Base.metadata.create_all(conn)
            else:
                Base.metadata.create_all(engine)
            return
        except (OperationalError, IntegrityError, ProgrammingError):
            if attempt == attempts - 1:
                raise
            time.sleep(2)


@contextmanager
def session_scope() -> Iterator[Session]:
    if _session_factory is None:
        init_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_ignore_duplicates(
    session: Session, table: Any, rows: Sequence[dict[str, Any]], key: str = "event_id"
) -> int:
    """INSERT ... ON CONFLICT (key) DO NOTHING. Retorna quantas linhas foram realmente inseridas."""
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    insert_fn = postgresql.insert if dialect == "postgresql" else sqlite.insert
    stmt = insert_fn(table).values(list(rows)).on_conflict_do_nothing(index_elements=[key])
    result = session.execute(stmt.returning(table.c[key]))
    return len(result.fetchall())
