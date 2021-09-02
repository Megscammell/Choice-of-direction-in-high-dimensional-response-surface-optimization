import numpy as np


def divide_abs_max_value(direction):
    """
    Divide all entries of the direction array by the largest absolute entry.
    Consequently, all elements of upd_direction will be between [-1, 1].

    Parameters
    ----------
    direction : 1-D array
                Search direction

    Returns
    -------
    upd_direction : 1-D array
                    Updated search direction, where all elements
                    of upd_direction will be between [-1, 1].
    """

    upd_direction = direction / np.max(abs(direction))
    return upd_direction
