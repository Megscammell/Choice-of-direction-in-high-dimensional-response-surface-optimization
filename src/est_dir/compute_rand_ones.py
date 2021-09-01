import numpy as np
import random
import est_dir

def compute_shuffle_cols(temp):
    """Shuffle elements in each columns."""
    return np.take_along_axis(temp,
                              np.random.rand(*temp.shape).argsort(axis=0),axis=0)


def compute_rand_ones(n, m, centre_point, no_vars, f, func_args,
                      region):
    """
    Compute response function values using design matrix from
    construct_design().
    """
    if (n % 2) != 0:
        raise ValueError('n must be even.')

    if no_vars == m:
        positions = np.arange(m)
    else:
        positions = np.sort(np.random.choice(np.arange(m), no_vars,
                            replace=False))
    assert(np.unique(positions).shape[0] == no_vars)

    arr = np.repeat(1, n)
    arr[:int(n/2)] = -1
    temp = np.tile(arr, (no_vars, 1)).T
    design = compute_shuffle_cols(temp)
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n, m, f,
                                      func_args, region)
    return design, y, positions, func_evals 

