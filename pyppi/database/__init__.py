
"""
Top-level of module database. This file instantiates a project-level
session that should be used for all transactions.
"""

import os
import atexit
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.ext.declarative import declarative_base

from ..base.file_paths import default_db_path

logger = logging.getLogger("pyppi")
db_engine = create_engine(
    "sqlite:///" + os.path.normpath(default_db_path),
    convert_unicode=True, connect_args={'check_same_thread': False}
)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False,  bind=db_engine)
)
Base = declarative_base()
Base.query = db_session.query_property()


def init_database(engine):
    from .models import (
        Protein, Interaction, Pubmed, Psimi, Reference
    )
    Base.metadata.create_all(bind=engine, checkfirst=True)


def create_session(db_path, echo=False):
    from .models import (
        Protein, Interaction, Pubmed, Psimi, Reference
    )
    try:
        engine = create_engine(
            "sqlite:///" + os.path.normpath(db_path),
            convert_unicode=True, connect_args={'check_same_thread': False}
        )
        session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=engine)
        )
        Base.query = session.query_property()
        init_database(engine)

        Protein.query = session.query_property()
        Interaction.query = session.query_property()
        Pubmed.query = session.query_property()
        Psimi.query = session.query_property()
        Reference.query = session.query_property()

        return session, engine
    except:
        raise


def cleanup_module():
    logger.info("Closing database session.")
    global db_session
    global db_engine
    db_session.remove()
    db_session.close_all()
    db_engine.dispose()


def cleanup_database(session, engine):
    session.remove()
    session.close_all()
    engine.dispose()


def delete_database(session):
    from ..database.models import (
        Protein, Interaction, Pubmed, Psimi, Reference
    )

    session.query(Protein).delete()
    session.query(Interaction).delete()
    session.query(Pubmed).delete()
    session.query(Psimi).delete()
    session.query(Reference).delete()

    try:
        session.commit()
    except:
        session.rollback()
        raise
