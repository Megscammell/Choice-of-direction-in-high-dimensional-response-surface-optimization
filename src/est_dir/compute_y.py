import numpy as np
import random


def compute_y(centre_point, design, positions, n, m, f, func_args, region):
    """Compute response function values."""
    func_evals = 0
    y = np.zeros((n, ))
    for j in range(n):
        adj_point = np.zeros((m, ))
        adj_point[positions] = region * design[j, :]
        y[j] = f(adj_point + centre_point, *func_args)
        func_evals += 1
    return y, func_evals

