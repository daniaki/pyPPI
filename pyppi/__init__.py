import logging

logger = logging.getLogger("pyppi")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


__all__ = [
    'base',
    'data',
    'data_mining',
    'model_selection',
    'models',
    'network_analysis',
    'output',
    'tests'
]
