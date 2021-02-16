import numpy as np
import pytest

import est_dir


def test_1():
    """Test that design matrix has first column of all ones."""
    n = 10
    m = 5
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.array([2.01003596, 3.29020466, 2.96499689, 0.93333668,
                             3.33078812])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    choice = 'random_cols'
    act_design, y,  positions, func_evals = (est_dir.create_design_matrix
                                            (n, m, centre_point,
                                             choice, f, func_args))
    assert(positions.shape == (m,))
    assert(func_evals == n)
    assert(y.shape == (n, ))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    assert(np.all(full_act_design[:, 0] == np.ones(n)))


def test_2():
    """
    Test outputs of compute_direction with option = 'LS', n=16 and m=10.
    """
    np.random.seed(91)
    n = 16
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0,5,(m,))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    direction, func_evals = (est_dir.compute_direction
                             (n, m, centre_point, f, func_args, option))
    assert(func_evals == 16)
    assert(direction.shape == (m, ))
    assert(np.where(direction == 0)[0].shape == (0,))


def test_3():
    """
    Test outputs of compute_direction with option = 'LS', n=100 and m=500.
    """
    np.random.seed(91)
    n = 100
    m = 500
    f = est_dir.sphere_f_noise
    option = 'LS'
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    direction, func_evals = (est_dir.compute_direction
                             (n, m, centre_point, f, func_args, option))
    assert(func_evals == 16)                          
    assert(direction.shape == (m, ))
    assert(np.where(direction == 0)[0].shape[0] == (m - 10))


def test_4():
    """
    Test outputs of compute_direction with option = 'XY', n=10 and m=5.
    """
    np.random.seed(91)
    n = 10
    m = 5
    f = est_dir.sphere_f_noise
    option = 'XY'
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    direction, func_evals = (est_dir.compute_direction
                             (n, m, centre_point, f, func_args, option))
    assert(direction.shape == (m, ))
    assert(func_evals == n)


def test_5():
    """
    Test outputs of compute_direction with option = 'XY', n=20 and m=50.
    """
    np.random.seed(91)
    n = 20
    m = 50
    f = est_dir.sphere_f_noise
    option = 'XY'
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    direction, func_evals = (est_dir.compute_direction
                             (n, m, centre_point, f, func_args, option))
    assert(direction.shape == (m, ))
    assert(func_evals == n)


def test_6():
    """
    Test that option='XY_LS' returns error.
    """
    np.random.seed(91)
    n = 10
    m = 5
    f = est_dir.sphere_f_noise
    option = 'XY_LS'
    minimizer = np.ones((m,))
    centre_point = np.array([1.82147728, 1.08275171, 1.32970991,
                             1, 1.00918191])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    with pytest.raises(ValueError):
        (est_dir.compute_direction
         (n, m, centre_point, f, func_args, option))


def test_7():
    """
    Test outputs of calc_first_phase_RSM with option = 'LS'.
    """
    np.random.seed(92)
    store_centre_points = []
    n = 16
    m = 20
    f = est_dir.sphere_f_noise
    option = 'LS'
    const_back = 0.5
    forward_tol = 1000000
    back_tol = 0.000001
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    store_centre_points.append(centre_point)
    init_func_val = f(centre_point, *func_args)
    step = 1
    (opt_t, upd_point,
    f_new,
    total_func_evals_step,
    total_func_evals_dir,
    flag) = (est_dir.calc_first_phase_RSM
             (store_centre_points, init_func_val, f, func_args,
              option, n, m, const_back, back_tol,
              const_forward, forward_tol, step))
    assert(flag == True)
    assert(upd_point.shape == (m, ))
    assert(f_new < init_func_val)
    assert(total_func_evals_step > 0)
    assert(total_func_evals_dir == n)
    assert(opt_t > 0)


def test_9():
    """
    Raise error if store_centre_points is not a list within
    calc_first_phase_RSM.
    """
    np.random.seed(91)
    n = 10
    m = 5
    f = est_dir.sphere_f_noise
    option = 'XY'
    const_back = 0.5
    back_tol = 0.000001
    forward_tol = 1000000
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    store_centre_points = np.array([1.82147728, 1.08275171, 1.32970991,
                                    1, 1.00918191])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    init_func_val = f(store_centre_points, *func_args)
    step = 1
    with pytest.raises(ValueError):
        (est_dir.calc_first_phase_RSM
        (store_centre_points, init_func_val, f, func_args,
         option, n, m, const_back, back_tol,
         const_forward, forward_tol, step))
