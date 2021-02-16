import numpy as np
import random


def compute_shuffle_cols(temp):
    """Shuffle elements in each columns."""
    return np.take_along_axis(temp,
                              np.random.rand(*temp.shape).argsort(axis=0),axis=0)


def compute_y(centre_point, design, positions, n, m, f, func_args):
    """Compute response function values."""
    func_evals = 0
    y = np.zeros((n, ))
    for j in range(n):
        adj_point = np.zeros((m, ))
        adj_point[positions] = design[j, :]
        y[j] = f(adj_point + centre_point, *func_args)
        func_evals += 1
    return y, func_evals


def compute_frac_rand_ones(n, m, positions, centre_point,
                           f, func_args):
    """
    Compute response function values using design matrix from
    construct_design().
    """
    arr = np.repeat(1, n)
    arr[:int(n/2)] = -1
    temp = np.tile(arr, (m, 1)).T
    design = compute_shuffle_cols(temp)
    y, func_evals = compute_y(centre_point, design, positions, n, m, f,
                              func_args)
    return design, y, func_evals


def compute_frac_fact(n, m, positions, centre_point, f, func_args):
    """
    Compute response function values using a 2^{10-6} design matrix.
    """
    design = np.array([[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  +1,  +1],
                       [+1,  -1,  -1,  -1,  +1,  -1,  +1,  +1,  -1,  -1],
                       [-1,  +1,  -1,  -1,  +1,  +1,  -1,  +1,  -1,  -1],
                       [+1,  +1,  -1,  -1,  -1,  +1,  +1,  -1,  +1,  +1],
                       [-1,  -1,  +1,  -1,  +1,  +1,  +1,  -1,  -1,  +1],
                       [+1,  -1,  +1,  -1,  -1,  +1,  -1,  +1,  +1,  -1],
                       [-1,  +1,  +1,  -1,  -1,  -1,  +1,  +1,  +1,  -1],
                       [+1,  +1,  +1,  -1,  +1,  -1,  -1,  -1,  -1,  +1],
                       [-1,  -1,  -1,  +1,  -1,  +1,  +1,  +1,  -1,  +1],
                       [+1,  -1,  -1,  +1,  +1,  +1,  -1,  -1,  +1,  -1],
                       [-1,  +1,  -1,  +1,  +1,  -1,  +1,  -1,  +1,  -1],
                       [+1,  +1,  -1,  +1,  -1,  -1,  -1,  +1,  -1,  +1],
                       [-1,  -1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1],
                       [+1,  -1,  +1,  +1,  -1,  -1,  +1,  -1,  -1,  -1],
                       [-1,  +1,  +1,  +1,  -1,  +1,  -1,  -1,  -1,  -1],
                       [+1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1]])
    y, func_evals = compute_y(centre_point, design, positions, 16, m, f,
                              func_args)
    return design, y, func_evals


def create_design_matrix(n, m, centre_point, choice,
                         f, func_args):
    if choice == 'random_cols':
        if (n % 2) != 0:
            raise ValueError('n must be even.')
        positions = np.arange(m)
        X, y, func_evals = compute_frac_rand_ones(n, m, positions,
                                                  centre_point, f,
                                                  func_args)
        return X, y, positions, func_evals
    elif choice == 'fractional_fact':
        no_vars = 10
        positions = np.sort(np.random.choice(np.arange(m), no_vars,
                            replace=False))
        assert(np.unique(positions).shape[0] == 10)
        X, y, func_evals = compute_frac_fact(n, m, positions, centre_point,
                                             f, func_args)
        return X, y, positions, func_evals
    else:
        raise ValueError('Incorrect choice')