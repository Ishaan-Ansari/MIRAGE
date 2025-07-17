from contextlib import contextmanager
from typing import List

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import POSTGRES_DATABASE_URL

postgres_engine = create_engine(POSTGRES_DATABASE_URL)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=postgres_engine,
)

class PostGresBase(DeclarativeBase):
    pass

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_existing_tables() -> List[str]:
    inspector = inspect(subject=postgres_engine)
    return inspector.get_table_names()