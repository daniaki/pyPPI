
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
    from .models import Protein, Interaction
    Base.metadata.create_all(bind=engine)


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
    try:
        if db_path is not None:
            engine = create_engine("sqlite:///" + os.path.normpath(db_path))
            __init_database(engine)
        else:
            engine = ENGINE

        session = Session(bind=engine)
        yield session
        session.close()

    except:
        raise
