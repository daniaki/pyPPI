
"""
Top-level of module database. This file instantiates a project-level
session that should be used for all transactions.
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from ..data import default_db_path

logger = logging.getLogger("pyppi")
Base = declarative_base()


def __init_database(engine):
    from .models import (
        Protein, Interaction, Pubmed, Psimi,
        psimi_interactions, pmid_interactions
    )
    Base.metadata.create_all(bind=engine)
    psimi_interactions.create(bind=engine, checkfirst=True)
    pmid_interactions.create(bind=engine, checkfirst=True)


try:
    ENGINE = create_engine("sqlite:///" + os.path.normpath(default_db_path))
    __init_database(ENGINE)
except:
    logger.exception(
        "Could not create an engine for {}.".format(default_db_path)
    )
    raise


@contextmanager
def begin_transaction(db_path=None, echo=False):
    session = None
    try:
        if db_path is not None:
            engine = create_engine("sqlite:///" + os.path.normpath(db_path))
            __init_database(engine)
        else:
            engine = ENGINE

        session = Session(bind=engine)
        yield session
        session.commit()
        session.close()
    except:
        if session is not None:
            session.rollback()
        raise


def make_session(db_path=None, echo=False):
    try:
        if db_path is not None and db_path != default_db_path:
            engine = create_engine("sqlite:///" + os.path.normpath(db_path))
            __init_database(engine)
        else:
            engine = ENGINE
        session = Session(bind=engine)
        return session
    except:
        raise


def delete_database(session=None, db_path=None):
    from ..database.models import (
        Protein, Interaction, Pubmed, Psimi,
        psimi_interactions, pmid_interactions
    )
    if db_path is None:
        db_path = default_db_path

    if session is None:
        with begin_transaction(db_path=db_path) as session:
            session.query(Protein).delete()
            session.query(Interaction).delete()
            session.query(Pubmed).delete()
            session.query(Psimi).delete()

            session.query(psimi_interactions).delete(synchronize_session=False)
            session.query(pmid_interactions).delete(synchronize_session=False)

            try:
                session.commit()
            except:
                session.rollback()
                raise
    else:
        session.query(Protein).delete()
        session.query(Interaction).delete()
        session.query(Pubmed).delete()
        session.query(Psimi).delete()

        session.query(psimi_interactions).delete(synchronize_session=False)
        session.query(pmid_interactions).delete(synchronize_session=False)

        try:
            session.commit()
        except:
            session.rollback()
            raise
