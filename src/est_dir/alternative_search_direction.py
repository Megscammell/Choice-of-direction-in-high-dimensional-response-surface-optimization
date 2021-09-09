import numpy as np
import time

import est_dir


def rsm_alternative_search_direction(centre_point, f, func_args, n, m,
                                     no_vars, region, max_func_evals,
                                     const_back=0.5, back_tol=0.000001,
                                     const_forward=2, forward_tol=100000000):
    """
    Compute iterations of Phase I of response surface methodology
    until some stopping criterion is met.
    The direction is estimated by compute_direction_XY() at each iteration.
    The step length is computed by forward or backward tracking.

    Parameters
    ----------
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
    n : integer
        Number of observations of the design matrix.
    m : integer
        Number of variables.
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
    region : float
             Region of exploration around the centre point.
    max_func_evals : int
                     Maximum number of function evaluations before stopping.
    const_back : float
                 If backward tracking is required, the initial guess of the
                 step size will be multiplied by const_back at each iteration
                 of backward tracking. That is,
                 t <- t * const_back
                 It should be noted that const_back < 1.
    back_tol : float
               It must be ensured that the step size computed by backward
               tracking is not smaller than back_tol. If this is the case,
               iterations of backward tracking are terminated. Typically,
               back_tol is a very small number.
    const_forward : float
                    The initial guess of the
                    step size will be multiplied by const_forward at each
                    iteration of forward tracking. That is,
                    t <- t * const_back
                    It should be noted that const_forward > 1.
    forward_tol : float
                  It must be ensured that the step size computed by forward
                  tracking is not greater than forward_tol. If this is the
                  case, iterations of forward tracking are terminated.

    Returns
    -------
    upd_point : 1-D array
                Updated centre_point after applying local search with
                estimated direction and step length.
    init_func_val : float
                    Initial function value at initial centre_point.
    f_val : float
            Final response function value after stopping criterion
            has been met for phase I of RSM.
    full_time : float
                Total time taken.
    total_func_evals_step : integer
                            Total number of response function evaluations
                            to compute step length for all iterations.
    total_func_evals_dir : integer
                           Total number of response function evaluations
                           to compute direction for all iterations.
    no_iterations : integer
                    Total number of iterations of Phase I of RSM.
    """
    if type(n) is not int:
        raise ValueError('n must be an integer.')
    if type(m) is not int:
        raise ValueError('dimension must be an integer.')
    if centre_point.shape != (m, ):
        raise ValueError('centre_point must be of correct dimension.')
    if type(no_vars) is not int:
        raise ValueError('no_vars must be an integer.')
    if (type(region) is not int) and (type(region) is not float):
        raise ValueError('region must be an integer or float.')
    if type(max_func_evals) is not int:
        raise ValueError('max_func_evals must be an integer.')
    if (no_vars > m):
        raise ValueError('no_vars must be less than or equal to dimension')
    t0 = time.time()
    total_func_evals_step = 0
    total_func_evals_dir = 0
    step = 1
    init_func_val = f(centre_point, *func_args)
    (upd_point, f_val,
     func_evals_step,
     func_evals_dir) = (est_dir.calc_first_phase_RSM_XY
                        (centre_point, np.copy(init_func_val), f, func_args,
                         n, m, const_back, back_tol, const_forward,
                         forward_tol, step, no_vars, region))
    total_func_evals_step += func_evals_step
    total_func_evals_dir += func_evals_dir
    no_iterations = 1

    while (total_func_evals_step + total_func_evals_dir + n) < max_func_evals:
        centre_point = upd_point
        new_func_val = f_val
        step = 1
        (upd_point, f_val,
         func_evals_step,
         func_evals_dir) = (est_dir.calc_first_phase_RSM_XY
                            (centre_point, np.copy(new_func_val), f, func_args,
                             n, m, const_back, back_tol, const_forward,
                             forward_tol, step, no_vars, region))
        total_func_evals_step += func_evals_step
        total_func_evals_dir += func_evals_dir
        no_iterations += 1

    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations)
