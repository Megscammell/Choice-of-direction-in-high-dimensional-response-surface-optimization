import numpy as np
import pytest

import est_dir


def test_1():
    """
    Check outputs are of correct form for
    rsm_alternative_search_direction() - enter while loop.
    """
    seed = 3053
    np.random.seed(seed)
    f = est_dir.quad_f_noise
    n = 16
    m = 10
    minimizer = np.zeros((m,))
    centre_point = np.random.uniform(-1, 1, (m, ))
    matrix = est_dir.quad_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m
    region = 0.1
    (upd_point,
     init_func_val,
     f_val, full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_iterations) = (est_dir.rsm_alternative_search_direction
                       (centre_point, f, func_args, n, m,
                        no_vars, region, max_func_evals,
                        const_back, back_tol,
                        const_forward, forward_tol))

    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert((total_func_evals_step + total_func_evals_dir) >= max_func_evals)
    assert(no_iterations > 0)
    assert(full_time >= 0)
    assert(f_val < init_func_val)


def test_2():
    """
    Check outputs are of correct form for rsm_alternative_search_direction() -
    do not enter while loop.
    """
    seed = 3053
    np.random.seed(seed)
    f = est_dir.quad_f_noise
    n = 16
    m = 10
    minimizer = np.zeros((m,))
    centre_point = np.random.uniform(-1, 1, (m, ))
    matrix = est_dir.quad_func_params(1, 1, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 10
    no_vars = m
    region = 0.1
    (upd_point,
     init_func_val,
     f_val, full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_iterations) = (est_dir.rsm_alternative_search_direction
                       (centre_point, f, func_args, n, m,
                        no_vars, region, max_func_evals,
                        const_back, back_tol,
                        const_forward, forward_tol))
    assert(type(total_func_evals_step) is int)
    assert(type(total_func_evals_dir) is int)
    assert(upd_point.shape == (m, ))
    assert((total_func_evals_step + total_func_evals_dir) >= max_func_evals)
    assert(no_iterations > 0)
    assert(full_time >= 0)
    assert(f_val < init_func_val)


def test_3():
    """
    Check error message if no_vars > m.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m * 2
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))


def test_4():
    """
    Check error message if n is not integer.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = '16'
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))


def test_5():
    """
    Check error message if m is not integer.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, False,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))


def test_6():
    """
    Check error message if centre_point is not of correct shape.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (2, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))


def test_7():
    """
    Check error message if no_vars is not integer.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = False
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))


def test_8():
    """
    Check error message if region is not integer or float.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100
    no_vars = m
    region = False
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))

    
def test_9():
    """
    Check error message if max_func_evals is not integer.
    """
    f = est_dir.quad_f_noise
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 1)
    minimizer = np.random.uniform(*domain, (m, ))
    centre_point = np.random.uniform(*domain, (m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    func_args = (minimizer, matrix, 0, 1)
    const_back = 0.75
    back_tol = 0.0000001
    const_forward = (1 / const_back)
    forward_tol = 10000000
    max_func_evals = 100.5
    no_vars = m
    region = 0.1
    with pytest.raises(ValueError):
        (est_dir.rsm_alternative_search_direction
         (centre_point, f, func_args, n, m,
          no_vars, region, max_func_evals,
          const_back, back_tol,
          const_forward, forward_tol))
