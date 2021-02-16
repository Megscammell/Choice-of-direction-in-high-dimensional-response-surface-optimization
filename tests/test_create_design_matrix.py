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
    positions = np.arange(m)
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
    matrix = est_dir.sphere_func_params(1, 10, m)
    minimizer = np.ones((m,))
    func_args = (minimizer, matrix, 0, 5)
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n, m,
                                f, func_args)
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(np.all(y > 0))


def test_3():
    """
    Check outputs of compute_frac_rand_ones.
    """
    n = 20
    m = 100
    positions = np.arange(m)
    f = est_dir.sphere_f_noise
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    minimizer = np.ones((m,))
    func_args = (minimizer, matrix, 0, 5)
    design, y, func_evals = est_dir.compute_frac_rand_ones(n, m,  positions, centre_point,
                                                            f, func_args)
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(np.all(y > 0))
    assert(np.all(design != 0))
    assert(design.shape == (n, m))

    for j in range(m):
        assert(np.all(np.sum(design[:, j]) == 0))


def test_4():
    """
    Check outputs of compute_frac_fact.
    """
    n = 20
    m = 200
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    np.random.seed(90)
    positions = np.sort(np.random.choice(np.arange(m), 10, replace=False))
    X, y, func_evals = est_dir.compute_frac_fact(n, m, positions,
                                                 centre_point, f,
                                                 func_args)

    assert(y.shape == (16, ))
    assert(func_evals == 16)
    assert(np.all(X == np.array([[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  +1,  +1],
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


def test_5():
    """
    Check outputs of create_design_matrix with choice = 'random_cols'.
    """
    n = 20
    m = 200
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    choice = 'random_cols'
    np.random.seed(90)
    X, y, positions, func_evals = est_dir.create_design_matrix(n, m,
                                                               centre_point,
                                                               choice,
                                                               f, func_args)
    assert(X.shape == (n, m))
    assert(y.shape == (n, ))
    assert(func_evals == n)
    assert(positions.shape == (m, ))
    for j in range(m):
        assert(np.sum(X[:, j]) == 0)


def test_6():
    """
    Check outputs of create_design_matrix with choice = 'fractional_fact'.
    """
    n = 20
    m = 200
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    choice = 'fractional_fact'
    np.random.seed(90)
    X, y, positions, func_evals = est_dir.create_design_matrix(n, m,
                                                        centre_point,
                                                        choice,
                                                        f, func_args)
    assert(X.shape == (16, 10))
    assert(y.shape == (16, ))
    assert(func_evals == 16)
    assert(positions.shape == (10, ))



def test_7():
    """Asserts error message when incorrect choice is given."""
    n = 6
    m = 2
    choice = 'repeat_random'
    f = est_dir.sphere_f_noise
    centre_point = np.array([1, 3])
    minimizer = np.array([7.5, 9])
    matrix = est_dir.sphere_func_params(1, 4, m)
    func_args = (minimizer, matrix, 0, 0.001)
    with pytest.raises(ValueError):
        est_dir.create_design_matrix(n, m, centre_point,
                                choice, f,
                                func_args)


def test_8():
    """
    Asserts error message when n is not even for choice =
    'random_1_-1_cols.
    """
    n = 5
    m = 2
    choice = 'random_cols'
    f = est_dir.sphere_f_noise
    centre_point = np.array([1, 3])
    minimizer = np.array([7.5, 9])
    matrix = est_dir.sphere_func_params(1, 4, m)
    func_args = (minimizer, matrix, 0, 0.001)
    with pytest.raises(ValueError):
        est_dir.create_design_matrix(n, m, centre_point,
                                     choice, f,
                                     func_args)
