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
    no_vars = m
    region = 0.1
    act_design, y,  positions, func_evals = (est_dir.compute_rand_ones
                                             (n, m, centre_point, no_vars,
                                              f, func_args, region))
    assert(positions.shape == (m,))
    assert(func_evals == n)
    assert(y.shape == (n, ))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    assert(np.all(full_act_design[:, 0] == np.ones(n)))
    assert(np.all(full_act_design[:, 1:] == act_design))


def test_2():
    """
    Test outputs of compute_direction with option = 'LS', n=16, m=10,
    no_vars=m and F-test p-value is greater than 0.1.
    """
    np.random.seed(91)
    n = 16
    m = 10
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.repeat(1.1, m)
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_LS
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 16)
    assert(direction == False)


def test_3():
    """
    Test outputs of compute_direction with option = 'LS', n=16, m=10,
    no_vars=m and F-test p-value is smaller than 0.1.
    """
    np.random.seed(96)
    n = 16
    m = 10
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_LS
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == 0)
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)


def test_4():
    """
    Test outputs of compute_direction with option = 'LS', n=16, m=100,
    no_vars=10 and F-test p-value is smaller than 0.1.
    """ 
    np.random.seed(96)
    n = 16
    m = 100
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_LS
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals >= 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == (m-no_vars))
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)


def test_5():
    """
    Test outputs of compute_direction with option = 'LS', n=16, m=100,
    no_vars=10 and F-test p-value is greater than 0.1.
    """ 
    np.random.seed(100)
    n = 16
    m = 100
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.repeat(1.1, m)
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_LS
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 160)
    assert(direction == False)


def test_6():
    m = 20
    no_vars = 4
    set_all_positions = np.arange(m)
    positions = np.array([[1, 3, 5, 7],
                [0, 2, 10, 15],
                [4, 12, 19, 6],
                [8, 9, 17, 11],
                [13, 14, 16, 18]])
    index = 0
    while True:
        set_all_positions = np.setdiff1d(set_all_positions, positions[index])
        for k in range(index + 1):
            for i in range(no_vars):
                assert(positions[k][i] not in set_all_positions)
        index += 1
        if index >= np.floor(m / no_vars):
            break


def test_7():
    """
    Test outputs of compute_direction with option = 'XY', n=16, m=10,
    no_vars=m.
    """
    np.random.seed(91)
    n = 16
    m = 10
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_XY
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == 0)
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)


def test_8():
    """
    Test outputs of compute_direction with option = 'XY', n=16, m=100,
    no_vars=m.
    """
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_XY
                            (n, m, centre_point, f, func_args,
                             no_vars, region, const))
    assert(func_evals == 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == 0)
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)


def test_9():
    """
    Test outputs of calc_first_phase_RSM with option = 'LS', where F-test
    p-value is less than 0.1.
    """
    np.random.seed(96)
    n = 16
    m = 10
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    init_func_val = f(centre_point, *func_args)
    const_back = 0.5
    forward_tol = 1000000
    back_tol = 0.000001
    const_forward = (1 / const_back)
    step = 1
    no_vars = 10
    (opt_t, upd_point,
    f_new,
    total_func_evals_step,
    total_func_evals_dir,
    flag, mean_norm_grad) = (est_dir.calc_first_phase_RSM_LS
                             (centre_point, init_func_val, f, func_args,
                              n, m, const_back, back_tol,
                              const_forward, forward_tol, step, no_vars, region,
                              const))
    assert(upd_point.shape == (m, ))
    assert(f_new < init_func_val)
    assert(total_func_evals_step > 0)
    assert(total_func_evals_dir == n)
    assert(opt_t > 0)
    assert(flag == True)
    assert(mean_norm_grad > 0)


def test_10():
    """
    Test outputs of calc_first_phase_RSM with option = 'LS', where F-test
    p-value is greater than 0.1.
    """
    np.random.seed(100)
    n = 16
    m = 100
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.repeat(1.1, m)
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    init_func_val = f(centre_point, *func_args)
    const_back = 0.5
    forward_tol = 1000000
    back_tol = 0.000001
    const_forward = (1 / const_back)
    step = 1
    (opt_t, upd_point,
    f_new,
    total_func_evals_step,
    total_func_evals_dir,
    flag, mean_norm_grad) = (est_dir.calc_first_phase_RSM_LS
                             (centre_point, init_func_val, f, func_args,
                              n, m, const_back, back_tol,
                              const_forward, forward_tol, step, no_vars, region,
                              const))
    assert(np.all(np.round(upd_point, 5)== np.round(centre_point, 5)))
    assert(f_new == init_func_val)
    assert(total_func_evals_step == 0)
    assert(total_func_evals_dir == n * (int(m / no_vars)))
    assert(opt_t == 0)
    assert(flag == False)
    assert(mean_norm_grad == 0)


def test_11():
    """
    Test outputs of calc_first_phase_RSM with option = 'XY'.
    """
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    init_func_val = f(centre_point, *func_args)
    const_back = 0.5
    forward_tol = 1000000
    back_tol = 0.000001
    const_forward = (1 / const_back)
    step = 1
    (opt_t, upd_point,
    f_new,
    total_func_evals_step,
    total_func_evals_dir,
    mean_norm_grad) = (est_dir.calc_first_phase_RSM_XY
                        (centre_point, init_func_val, f, func_args,
                         n, m, const_back, back_tol,
                         const_forward, forward_tol, step, no_vars, region,
                         const))
    assert(upd_point.shape == (m, ))
    assert(f_new < init_func_val)
    assert(total_func_evals_step > 0)
    assert(total_func_evals_dir == n)
    assert(opt_t > 0)
    assert(mean_norm_grad > 0)


def test_12():
    direction = np.array([0.8, 0.2, -0.04, 0.5, -0.6, -0.95])
    new_direction = est_dir.coeffs_dir(direction)
    assert(np.all(np.sign(direction) == np.sign(new_direction)))
    assert(np.all(np.round(new_direction * np.max(abs(direction)), 2) == np.round(direction, 2)))


def test_13():
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_MP
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == 0)
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    act_design, y, positions, func_evals = (est_dir.compute_rand_ones
                                                (n, m, centre_point, no_vars,
                                                f, func_args, region))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    direction2 = np.zeros((m,))
    est = np.linalg.pinv(full_act_design.T @ full_act_design) @ full_act_design.T @ y
    direction2[positions] = est_dir.coeffs_dir(est[1:])
    assert(full_act_design.shape == (n, m+1))
    assert(np.all(full_act_design != 0))
    assert(np.all(full_act_design[:, 0] == np.ones(n)))
    assert(np.all(direction2 == direction))


def test_14():
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    direction, func_evals = (est_dir.compute_direction_MP
                            (n, m, centre_point, f, func_args,
                            no_vars, region, const))
    assert(func_evals == 16)
    assert(direction.shape == (m,))
    assert(np.where(direction == 0)[0].shape[0] == m- no_vars)
    assert(np.max(abs(direction)) == 1)
    pos_max = np.argmax(direction)
    for j in range(no_vars):
        if j != pos_max:
            assert(abs(direction[j]) <= 1)

    np.random.seed(91)
    n = 16
    m = 100
    no_vars = 10
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    act_design, y, positions, func_evals = (est_dir.compute_rand_ones
                                                (n, m, centre_point, no_vars,
                                                f, func_args, region))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    direction2 = np.zeros((m,))
    est = np.linalg.pinv(full_act_design.T @ full_act_design) @ full_act_design.T @ y
    direction2[positions] = est_dir.coeffs_dir(est[1:])
    assert(full_act_design.shape == (n, no_vars+1))
    assert(np.all(full_act_design != 0))
    assert(np.all(full_act_design[:, 0] == np.ones(n)))
    assert(np.all(direction2 == direction))


def test_15():
    np.random.seed(91)
    n = 16
    m = 100
    no_vars = m
    f = est_dir.sphere_f_noise
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(-2, 2, (m,))
    matrix = est_dir.sphere_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    region = 0.1
    const = 1
    init_func_val = f(centre_point, *func_args)
    const_back = 0.5
    forward_tol = 1000000
    back_tol = 0.000001
    const_forward = (1 / const_back)
    step = 1
    (opt_t, upd_point,
    f_new,
    total_func_evals_step,
    total_func_evals_dir,
    mean_norm_grad) = (est_dir.calc_first_phase_RSM_MP
                        (centre_point, init_func_val, f, func_args,
                        n, m, const_back, back_tol,
                        const_forward, forward_tol, step, no_vars, region,
                        const))
    assert(upd_point.shape == (m, ))
    assert(f_new < init_func_val)
    assert(total_func_evals_step > 0)
    assert(total_func_evals_dir == n)
    assert(opt_t > 0)
    assert(mean_norm_grad > 0)