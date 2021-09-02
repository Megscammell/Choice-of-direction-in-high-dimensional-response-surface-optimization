import numpy as np

import est_dir


def test_1():
    """
    Check outputs are of correct form for calc_its_until_sc_LS()
    and flag = False. Note that as centre_point is close to minimizer,
    checks that no further improvment can be made to response function
    value.
    """
    np.random.seed(5489637)
    m = 10
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    const_back = 0.5
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 1000000
    minimizer = np.ones((m,))
    region = 0.1
    centre_point = np.array([1.82147728, 1.08275171, 1.32970991,
                             1, 1.00918191, 1.82147728, 1.08275171,
                             1.32970991, 1, 1.00918191])
    matrix = est_dir.quad_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    func_args_no_noise = (minimizer, matrix)
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its, store_good_dir,
     store_good_dir_norm,
     store_good_dir_func,
     store_norm_grad) = est_dir.calc_its_until_sc_LS(centre_point, f,
                                                     func_args, m,
                                                     f_no_noise,
                                                     func_args_no_noise,
                                                     region,
                                                     const_back,
                                                     back_tol,
                                                     const_forward,
                                                     forward_tol)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert(no_its > 0)
    assert(full_time >= 0)
    assert(store_good_dir <= no_its)
    assert(len(store_good_dir_norm) == store_good_dir)
    assert(len(store_good_dir_func) == store_good_dir)
    if total_func_evals_step == 0:
        assert(f_val == init_func_val)
        assert(np.all(np.round(upd_point, 5) == np.round(centre_point, 5)))
        assert(store_good_dir == 0)
        assert(len(store_norm_grad) == 0)
    else:
        assert(f_val < init_func_val)
        assert(store_good_dir > 0)


def test_2():
    """
    Check outputs are of correct form for calc_its_until_sc_LS(), with
    (Flag = True)
    """
    np.random.seed(5489637)
    m = 100
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    const_back = 0.5
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 1000000
    minimizer = np.zeros((m,))
    region = 0.1
    centre_point = np.random.uniform(-2, 2, (m, ))
    matrix = est_dir.quad_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    func_args_no_noise = (minimizer, matrix)
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its, store_good_dir,
     store_good_dir_norm,
     store_good_dir_func,
     store_norm_grad) = est_dir.calc_its_until_sc_LS(centre_point, f,
                                                     func_args, m,
                                                     f_no_noise,
                                                     func_args_no_noise,
                                                     region,
                                                     const_back,
                                                     back_tol,
                                                     const_forward,
                                                     forward_tol)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert(no_its > 0)
    assert(full_time >= 0)
    assert(store_good_dir <= no_its)
    assert(len(store_good_dir_norm) == store_good_dir)
    assert(len(store_good_dir_func) == store_good_dir)
    if total_func_evals_step == 0:
        assert(f_val == init_func_val)
        assert(np.all(np.round(upd_point, 5) == np.round(centre_point, 5)))
        assert(store_good_dir == 0)
        assert(len(store_norm_grad) == 0)
    else:
        assert(f_val < init_func_val)
        assert(store_good_dir > 0)
        assert(len(store_norm_grad) > 0)


def test_3():
    """
    Check outputs are of correct form for calc_its_until_sc_LS() with
    Flag = True and break from loop.
    """
    np.random.seed(5489637)
    m = 100
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    const_back = 0.5
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 1000000
    minimizer = np.zeros((m,))
    region = 0.1
    centre_point = np.random.uniform(-2, 2, (m, ))
    matrix = est_dir.quad_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    func_args_no_noise = (minimizer, matrix)
    (upd_point,
     init_func_val,
     f_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_its, store_good_dir,
     store_good_dir_norm,
     store_good_dir_func,
     store_norm_grad) = est_dir.calc_its_until_sc_LS(centre_point, f,
                                                     func_args, m,
                                                     f_no_noise,
                                                     func_args_no_noise,
                                                     region,
                                                     const_back,
                                                     back_tol,
                                                     const_forward,
                                                     forward_tol,
                                                     tol_evals=50)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert(no_its > 0)
    assert(full_time >= 0)
    assert(store_good_dir <= no_its)
    assert(len(store_good_dir_norm) == store_good_dir)
    assert(len(store_good_dir_func) == store_good_dir)
    if total_func_evals_step == 0:
        assert(f_val == init_func_val)
        assert(np.all(np.round(upd_point, 5) == np.round(centre_point, 5)))
        assert(store_good_dir == 0)
        assert(len(store_norm_grad) == 0)
    else:
        assert(f_val < init_func_val)
        assert(store_good_dir > 0)
        assert(len(store_norm_grad) > 0)
