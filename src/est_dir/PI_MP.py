import numpy as np
import time

import est_dir


def compute_direction_MP(n, m, centre_point, f, func_args, no_vars, region,
                         type_inverse):
    """
    Compute estimate of the search direction by applying Moore-Penrose
    pseudo inverse.

    Parameters
    ----------
    n : integer
        Number of observations of the design matrix (rows).
    m : integer
        Number of variables of the design matrix (columns).
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
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
    type_inverse : string
                   Determine whether to perform a left or right inverse.

    Returns
    -------
    direction : 1-D array
                Estimated search direction.
    func_evals : integer
                 Number of times the response function has been evaluated
                 to compute the search direction.
    """
    act_design, y, positions, func_evals = (est_dir.compute_random_design
                                            (n, m, centre_point, no_vars,
                                             f, func_args, region))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    direction = np.zeros((m,))
    if type_inverse == 'left':
        est = (np.linalg.pinv(full_act_design.T @ full_act_design) @
               full_act_design.T @ y)
    elif type_inverse == 'right':
        est = (full_act_design.T @
               np.linalg.pinv(full_act_design @ full_act_design.T) @ y)
    direction[positions] = est_dir.divide_abs_max_value(est[1:])
    assert(max(abs(direction) == 1))
    return direction, func_evals


def calc_first_phase_RSM_MP(centre_point, init_func_val, f, func_args,
                            n, m, const_back, back_tol, const_forward,
                            forward_tol, step, no_vars, region,
                            type_inverse):
    """
    Compute iteration of local search, where search direction is estimated
    by compute_direction_MP(), and the step size is computed using either
    forward or backward tracking.

    Parameters
    ----------
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    init_func_val: float
                   Initial function value at centre_point. That is,
                   f(centre_point, *func_args).
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
    step : float
           Initial guess of step size.
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
    type_inverse : string
                   Determine whether to perform a left or right inverse.

    Returns
    -------
    upd_point : 1-D array
                Updated centre_point after applying local search with
                estimated direction and step length.
    f_new : float
            Response function value at upd_point. That is,
            f(upd_point, *func_args).
    total_func_evals_step : integer
                            Total number of response function evaluations
                            to compute step length.
    total_func_evals_dir : integer
                           Total number of response function evaluations
                           to compute direction.
    """
    direction, total_func_evals_dir = (compute_direction_MP
                                       (n, m, centre_point, f,
                                        func_args, no_vars,
                                        region, type_inverse))
    (upd_point,
     f_new, total_func_evals_step) = (est_dir.combine_tracking
                                      (centre_point, init_func_val,
                                       direction, step, const_back,
                                       back_tol, const_forward,
                                       forward_tol, f, func_args))
    return (upd_point, f_new, total_func_evals_step,
            total_func_evals_dir)


def calc_its_until_sc_MP(centre_point, f, func_args, n, m,
                         f_no_noise, func_args_no_noise,
                         no_vars, region, max_func_evals,
                         type_inverse, const_back=0.5, back_tol=0.000001,
                         const_forward=2, forward_tol=100000000):
    """
    Compute iterations of Phase I of response surface methodology
    until some stopping criterion is met.
    The direction is estimated by compute_direction_MP() at each iteration.
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
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.
    func_args_no_noise : tuple
                         Arguments passed to the function f_no_noise.
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
    type_inverse : string
                   Determine whether to perform a left or right inverse.
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
    store_good_dir : integer
                     Number of 'good' search directions. That is, the number
                     of times moving along the estimated search direction
                     improves the response function value with no noise.
    store_good_dir_norm : list
                          If a 'good' direction is determined,
                          distance of point and minimizer at the k-th
                          iteration subtracted by the distance of point
                          and minimizer at the (k+1)-th iteration is
                          stored.
    store_good_dir_func : list
                          If a 'good' direction is determined,
                          store the response function value with point
                          at the k-th iteration, subtracted by response
                          function value with point at the
                          (k+1)-th iteration.
    """

    t0 = time.time()
    if (no_vars > m):
        raise ValueError('Incorrect no_vars choice')
    store_good_dir = 0
    store_good_dir_norm = []
    store_good_dir_func = []
    total_func_evals_step = 0
    total_func_evals_dir = 0
    step = 1
    init_func_val = f(centre_point, *func_args)
    (upd_point, f_val,
     func_evals_step,
     func_evals_dir) = (calc_first_phase_RSM_MP
                        (centre_point, np.copy(init_func_val), f,
                         func_args, n, m, const_back,
                         back_tol, const_forward, forward_tol, step,
                         no_vars, region, type_inverse))
    total_func_evals_step += func_evals_step
    total_func_evals_dir += func_evals_dir
    no_iterations = 1

    if (f_no_noise(centre_point, *func_args_no_noise) >
            f_no_noise(upd_point, *func_args_no_noise)):
        store_good_dir += 1
        store_good_dir_norm.append(np.linalg.norm(centre_point -
                                                  func_args[0]) -
                                   np.linalg.norm(upd_point -
                                                  func_args[0]))
        store_good_dir_func.append(f_no_noise(centre_point,
                                              *func_args_no_noise) -
                                   f_no_noise(upd_point,
                                              *func_args_no_noise))

    while (total_func_evals_step + total_func_evals_dir + n) < max_func_evals:
        centre_point = upd_point
        new_func_val = f_val
        step = 1
        (upd_point, f_val,
         func_evals_step,
         func_evals_dir) = (calc_first_phase_RSM_MP
                            (centre_point, np.copy(new_func_val), f, func_args,
                             n, m, const_back, back_tol, const_forward,
                             forward_tol, step, no_vars, region, type_inverse))
        total_func_evals_step += func_evals_step
        total_func_evals_dir += func_evals_dir
        no_iterations += 1

        if (f_no_noise(centre_point, *func_args_no_noise) >
                f_no_noise(upd_point, *func_args_no_noise)):
            store_good_dir += 1
            store_good_dir_norm.append(np.linalg.norm(centre_point -
                                                      func_args[0]) -
                                       np.linalg.norm(upd_point -
                                                      func_args[0]))
            store_good_dir_func.append(f_no_noise(centre_point,
                                                  *func_args_no_noise) -
                                       f_no_noise(upd_point,
                                                  *func_args_no_noise))

    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations, store_good_dir,
            store_good_dir_norm, store_good_dir_func)
