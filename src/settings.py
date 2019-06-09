import logging

from pathlib import Path

import peewee


PROJECT_NAME = "tamago"
LOGGER_NAME = PROJECT_NAME

BASE_DIR = Path().absolute()
HOME_DIR = Path().home() / f".{PROJECT_NAME}"
DATA_DIR = HOME_DIR / "data"

_dirs = [HOME_DIR, DATA_DIR]
for _dir in _dirs:
    if not _dir.exists():
        _dir.mkdir()

DATABASE_PATH = HOME_DIR / f"{PROJECT_NAME.lower()}.sqlite"
DATABASE = peewee.SqliteDatabase(DATABASE_PATH)


def create_tables():
    from . import models

    with DATABASE:
        DATABASE.create_tables(models.MODELS)
