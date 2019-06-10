from logging.config import dictConfig

from pathlib import Path

import peewee


PROJECT_NAME = "tamago"
LOGGER_NAME = PROJECT_NAME
STREAM_LOGGER = f"{PROJECT_NAME}_stream"

BASE_DIR = Path().absolute()
HOME_DIR = Path().home() / f".{PROJECT_NAME}"
DATA_DIR = HOME_DIR / "data"
NETWORKS_DIR = HOME_DIR / "networks"
MODELS_DIR = HOME_DIR / "models"
LOG_DIR = HOME_DIR / "logs"


_dirs = [HOME_DIR, DATA_DIR, NETWORKS_DIR, MODELS_DIR]
for _dir in _dirs:
    if not _dir.exists():
        _dir.mkdir()


DATABASE_PATH = HOME_DIR / f"{PROJECT_NAME.lower()}.sqlite"
DATABASE = peewee.SqliteDatabase(DATABASE_PATH)


def create_tables():
    from .database import models

    with DATABASE:
        DATABASE.create_tables(models.MODELS)


# Set up logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[%(levelname)s] %(asctime)s %(module)s %(message)s"
        },
        "simple": {"format": "%(levelname)s %(message)s"},
    },
    "handlers": {
        PROJECT_NAME: {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": HOME_DIR / f"{PROJECT_NAME}.log",
            "formatter": "verbose",
        },
        STREAM_LOGGER: {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        PROJECT_NAME: {
            "handlers": [PROJECT_NAME],
            "level": "INFO",
            "propagate": True,
        },
        STREAM_LOGGER: {
            "handlers": [PROJECT_NAME],
            "level": "INFO",
            "propagate": True,
        },
    },
}

dictConfig(LOGGING)
