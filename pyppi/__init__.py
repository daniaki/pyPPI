# import atexit
import logging

__all__ = [
    'base',
    'data_mining',
    'database',
    'model_selection',
    'models',
    'network_analysis',
    'predict'
]

logger = logging.getLogger("pyppi")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# atexit.register(cleanup_module)
def wrap_init():
    from .database import init_database, cleanup_module, db_engine
    init_database(db_engine)


wrap_init()
