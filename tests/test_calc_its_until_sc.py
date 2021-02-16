import numpy as np

import est_dir


def test_1():
    """
    Check outputs are of correct form for calc_its_until_sc with option ='XY'.
    """
    seed = 3053
    np.random.seed(seed)
    f = est_dir.sphere_f_noise
    n = 16
    m = 10
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    const_back = 0.75
    back_tol=0.0000001
    const_forward = (1 / const_back)
    forward_tol=10000000
    option = 'XY'
    max_func_evals = 100
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its) = est_dir.calc_its_until_sc(centre_point, f,
                                          func_args, n, m,
                                          option, const_back,
                                          back_tol, 
                                          const_forward,
                                          forward_tol,
                                          max_func_evals)

    assert(np.all(upd_point != centre_point))
    assert(init_func_val > f_val)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert((total_func_evals_step + total_func_evals_dir) >= max_func_evals)
    assert(no_its > 0)
    assert(full_time >= 0)


def test_2():
    """
    Check outputs are of correct form for calc_its_until_sc with option ='XY'.
    """
    seed = 3053
    np.random.seed(seed)
    f = est_dir.sphere_f_noise
    n = 16
    m = 10
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 10, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    const_back = 0.75
    back_tol=0.0000001
    const_forward = (1 / const_back)
    forward_tol=10000000
    option = 'LS'
    max_func_evals = 100
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its) = est_dir.calc_its_until_sc(centre_point, f,
                                          func_args, n, m,
                                          option, const_back,
                                          back_tol, 
                                          const_forward,
                                          forward_tol,
                                          max_func_evals)

    assert(np.all(upd_point != centre_point))
    assert(init_func_val > f_val)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert((total_func_evals_step + total_func_evals_dir) >= max_func_evals)
    assert(no_its > 0)
    assert(full_time >= 0)


def test_3():
    """
    Check outputs are of correct form for calc_its_until_sc with option
    = 'LS'. Note that as centre_point is close to minimizer,
    check that no further improvment can be made to response function
    value.
    """
    np.random.seed(5489637)
    store_centre_points = []
    n = 16
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    max_func_evals = 100
    const_back = 0.5
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 1000000
    minimizer = np.ones((m,))
    centre_point = np.array([1.82147728, 1.08275171, 1.32970991,
                             1, 1.00918191, 1.82147728, 1.08275171,
                             1.32970991, 1, 1.00918191])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    store_centre_points.append(centre_point)
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its) = est_dir.calc_its_until_sc(centre_point, f,
                                          func_args, n, m,
                                          option, const_back,
                                          back_tol, 
                                          const_forward,
                                          forward_tol,
                                          max_func_evals)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert((total_func_evals_step + total_func_evals_dir) >= max_func_evals)
    assert(f_val < init_func_val)
    assert(no_its > 0)
    assert(full_time >= 0)
