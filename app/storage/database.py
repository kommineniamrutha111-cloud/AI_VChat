"""Database session factory for SQLite persistence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.config import load_config

_CONFIG = load_config()
_ENGINE = create_engine(_CONFIG.database.url, echo=False, future=True)
_SessionFactory = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)



def init_db() -> None:
    """Create database tables if they do not exist."""
    from app.storage import models  # Local import to avoid circular import

    inspector = inspect(_ENGINE)
    table_names = inspector.get_table_names()
    schema_reset = False
    if "chat_messages" in table_names:
        columns = {column["name"] for column in inspector.get_columns("chat_messages")}
        if "session_id" not in columns:
            with _ENGINE.begin() as connection:
                connection.exec_driver_sql("DROP TABLE chat_messages")
            schema_reset = True
    if schema_reset and "chat_sessions" in table_names:
        with _ENGINE.begin() as connection:
            connection.exec_driver_sql("DROP TABLE chat_sessions")

    models.Base.metadata.create_all(_ENGINE)

@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transactional session scope."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

