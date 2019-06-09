import os
import gzip


def open_file(path, mode="rt"):
    if os.path.splitext(path)[-1] == "gz":
        func = gzip.open
    else:
        func = open

    return func(path, mode=mode)
