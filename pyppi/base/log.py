"""
Use this module to import the `create_logger` function, which is simply
a helper function automate a logger setup.
"""


import logging

__all__ = [
    'create_logger'
]


def create_logger(name, level):
    """Create a logger with the given name and logging level

    Returns
    ------
    logging.Logger
        A logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
