import numpy as np


def coeffs_dir(direction):
    return direction / np.max(abs(direction))
