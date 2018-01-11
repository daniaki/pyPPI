"""
Exceptions for dealing with database retrieval and insertion.
"""


class ObjectNotFound(Exception):
    """
    This exception should be raised when a database entry could not
    be found.
    """


class ObjectAlreadyExists(Exception):
    """
    This exception should be raised when a database entry already
    exists and the creation of another instance may cause issues.
    """
