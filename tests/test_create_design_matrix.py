import numpy as np
import pytest

import est_dir


def test_1():
    """Check outputs of compute_shuffle_cols."""
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20]])
    shuffled_arr = est_dir.compute_shuffle_cols(arr)
    assert(arr.shape == (5, 4))
    assert(np.any(arr != shuffled_arr))
    for j in range(4):
        assert(np.all(np.sort(shuffled_arr[:, j]) == arr[:, j]))


def test_2():
    """Check outputs of compute_y."""
    n = 16
    m = 10
    no_vars = m
    positions = np.sort(np.random.choice(np.arange(m), no_vars,
                       replace=False))
    assert(np.unique(positions).shape[0] == no_vars)
    f = est_dir.sphere_f_noise
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
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 1, m)
    minimizer = np.ones((m,))
    func_args = (minimizer, matrix, 0, 5)
    region = 1
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n, m,
                                      f, func_args, region)
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(np.all(y > 0))


def test_3():
    """
    Check outputs of compute_frac_rand_ones with no_vars = m.
    """
    n = 20
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 1, m)
    minimizer = np.ones((m,))
    func_args = (minimizer, matrix, 0, 5)
    region = 1
    (design, y,
     positions,
     func_evals)  = est_dir.compute_rand_ones(n, m, centre_point,
                                                   no_vars, f, func_args,
                                                   region)
    assert(np.unique(positions).shape[0] == no_vars)
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(np.all(y > 0))
    assert(np.all(design != 0))
    assert(design.shape == (n, m))

    for j in range(m):
        assert(np.all(np.sum(design[:, j]) == 0))


def test_4():
    """
    Check outputs of compute_frac_rand_ones with no_vars < m.
    """
    n = 20
    m = 100
    no_vars = 20
    f = est_dir.sphere_f_noise
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 1, m)
    region = 1
    minimizer = np.ones((m,))
    func_args = (minimizer, matrix, 0, 5)
    (design, y,
     positions,
     func_evals)  = est_dir.compute_rand_ones(n, m, centre_point,
                                                   no_vars, f, func_args,
                                                   region)
    assert(np.unique(positions).shape[0] == no_vars)
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(np.all(y > 0))
    assert(np.all(design != 0))
    assert(design.shape == (n, no_vars))

    for j in range(no_vars):
        assert(np.all(np.sum(design[:, j]) == 0))


def test_5():
    """
    Check outputs of compute_frac_fact.
    """
    n = 16
    m = 200
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 5)
    np.random.seed(90)
    no_vars = 10
    region = 1
    set_all_positions = np.arange(m)
    (design, y,
     positions,
     func_evals) = est_dir.compute_frac_fact(n, m, centre_point, no_vars,
                                             f, func_args, region,
                                             set_all_positions)
    assert(positions.shape[0] == no_vars)
    assert(y.shape == (16, ))
    assert(func_evals == 16)
    assert(np.all(design == np.array([[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  +1,  +1],
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
                                    [+1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1]])))
    assert(np.all(y > 0))


def test_6():
    """
    Asserts error message when n is not even for choice =
    'random_1_-1_cols.
    """
    n = 5
    m = 2
    f = est_dir.sphere_f_noise
    centre_point = np.array([1, 3])
    minimizer = np.array([7.5, 9])
    matrix = est_dir.sphere_func_params(1, 4, m)
    func_args = (minimizer, matrix, 0, 0.001)
    no_vars = None
    region = 1
    with pytest.raises(ValueError):
        est_dir.compute_rand_ones(n, m, centre_point,
                                        no_vars, f, func_args,
                                        region)


def test_7():
    """
    Asserts error message when no_vars is not correct
    choice for choice = 'fractional_fact'.
    """
    n = 20
    m = 200
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    no_vars = 50
    set_all_positions = np.arange(m)
    region = 1
    np.random.seed(90)
    with pytest.raises(ValueError):
        est_dir.compute_frac_fact(n, m, centre_point, no_vars, f, func_args,
                                  region, set_all_positions)
