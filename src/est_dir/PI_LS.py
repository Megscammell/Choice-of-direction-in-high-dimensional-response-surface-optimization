import numpy as np
import time
import statsmodels.api as sm

import est_dir


def compute_frac_fact(m, centre_point, no_vars, f, func_args,
                      region, set_all_positions):
    """
    Compute response function value at each observation of the design matrix
    centred at centre_point, where a 2^{10-6} design matrix is used.

    Parameters
    ----------
    m : integer
        Number of variables of the design matrix (columns).
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
    region : float
             Region of exploration around the centre point.
    set_all_positions : 1-D array
                        If no_vars < m, it is ensured that new variables of
                        centre_point are used to centre the design matrix.
                        That is, the random subset of variables of centre_point
                        will be selected randomly from set_all_positions.

    Returns
    -------
    design : 2-D array
             2^{10-6} design matrix.
    y : 1-D array
        Contains the response function values at each observation of the
        design matrix centred at centre_point.
    positions : 1-D array
                Positions of centre_point in which the design matrix has been
                centred. If the design matrix is of size (n, m), then
                positions = (1,2,...,m). Otherwise, if the design matrix is
                of size (n, no_vars), where no_vars < m, then
                positions will be of size (num_vars,) with entries chosen
                randomly from set_all_positions.
    func_evals : integer
                 Number of times the response function has been evaluated.
    """
    positions = np.sort(np.random.choice(set_all_positions, no_vars,
                        replace=False))
    assert(np.unique(positions).shape[0] == no_vars)
    if no_vars == 10:
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
        n_temp = 16
    else:
        raise ValueError('Incorrect no_vars choice')
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n_temp,
                                      m, f, func_args, region)
    return design, y, positions, func_evals


def compute_direction_LS(m, centre_point, f, func_args, no_vars, region):
    """
    Assume that the response function is approx linear in the neighbourhood
    of centre_point. Estimate unknown parameters of the linear model by
    using OLSE. Then direction is (theta_1,...,theta_m).

    Parameters
    ----------
    m : integer
        Number of variables.
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

    Returns
    -------
    direction : 1-D array
                Estimate of the direction i.e. (theta_1,...,theta_m),
                where theta_i (i=1,...,m) are coefficients of the linear
                model.
    func_evals_count : integer
                       Number of response function evaluations to compute
                       direction.
    """

    func_evals_count = 0
    set_all_positions = np.arange(m)
    total_checks = int(np.floor(m / no_vars))
    act_design, y, positions, func_evals = (compute_frac_fact
                                            (m, centre_point, no_vars,
                                             f, func_args, region,
                                             set_all_positions))
    func_evals_count += func_evals
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    est = sm.OLS(y, full_act_design)
    results = est.fit()
    index_vars = 1
    while results.f_pvalue >= 0.1:
        if index_vars >= total_checks:
            break
        set_all_positions = np.setdiff1d(set_all_positions, positions)
        act_design, y, positions, func_evals = (compute_frac_fact
                                                (m, centre_point, no_vars,
                                                 f, func_args, region,
                                                 set_all_positions))
        func_evals_count += func_evals
        full_act_design = np.ones((act_design.shape[0],
                                   act_design.shape[1] + 1))
        full_act_design[:, 1:] = act_design
        est = sm.OLS(y, full_act_design)
        results = est.fit()
        index_vars += 1
    if results.f_pvalue < 0.1:
        direction = np.zeros((m,))
        direction[positions] = est_dir.divide_abs_max_value(
                               (results.params)[1:])
        assert(max(abs(direction) == 1))
        return direction, func_evals_count
    else:
        return False, func_evals_count


def calc_first_phase_RSM_LS(centre_point, init_func_val, f, func_args,
                            m, const_back, back_tol, const_forward,
                            forward_tol, step, no_vars, region):
    """
    Compute iteration of local search, where search direction is estimated
    by (theta_1,...,theta_m), where theta_i (i=1,...,m) are coefficients
    of the linear model, and the step size is computed using either forward or
    backward tracking.

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
    flag : boolean
           Will be False if direction cannot be computed. Otherwise, will
           be True if direction can be computed.
    norm_direction_relative_to_m : float
                                   Norm of direction relative to no_vars and m.
    """

    direction, total_func_evals_dir = (compute_direction_LS
                                       (m, centre_point, f,
                                        func_args, no_vars,
                                        region))
    if direction is not False:
        (upd_point,
         f_new, total_func_evals_step) = (est_dir.combine_tracking
                                          (centre_point, init_func_val,
                                           direction, step, const_back,
                                           back_tol, const_forward,
                                           forward_tol, f, func_args))
        return (upd_point, f_new, total_func_evals_step,
                total_func_evals_dir, True,
                (np.linalg.norm(direction) / no_vars) * m)
    else:
        return centre_point, init_func_val, 0, total_func_evals_dir, False, 0


def calc_its_until_sc_LS(centre_point, f, func_args, m, f_no_noise,
                         func_args_no_noise, region,
                         const_back=0.5, back_tol=0.000001,
                         const_forward=2, forward_tol=100000000,
                         tol_evals=10000000):
    """
    Compute iterations of Phase I of response surface methodology
    until some stopping criterion is met.
    The direction is estimated by the least squares coefficients
    of the linear model at each iteration. The step length is
    computed by forward or backward tracking.

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
    region : float
             Region of exploration around the centre point.
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
    tol_evals : int
                Maximum number of function evaluations before stopping.

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
                          at the k-th iteration, subtracted by the response
                          function value with point at the
                          (k+1)-th iteration.
    norm_grad : list
                Norm of direction relative to no_vars and m at each iteration.

    """
    t0 = time.time()
    no_vars = 10
    store_good_dir = 0
    store_good_dir_norm = []
    store_good_dir_func = []
    store_norm_grad = []
    total_func_evals_step = 0
    total_func_evals_dir = 0
    step = 1
    init_func_val = f(centre_point, *func_args)
    (upd_point, f_val,
     func_evals_step,
     func_evals_dir,
     flag, norm_grad) = (calc_first_phase_RSM_LS
                         (centre_point, np.copy(init_func_val), f,
                          func_args, m, const_back, back_tol, const_forward,
                          forward_tol, step, no_vars, region))

    total_func_evals_step += func_evals_step
    total_func_evals_dir += func_evals_dir
    no_iterations = 1

    if (f_no_noise(centre_point, *func_args_no_noise) >
            f_no_noise(upd_point, *func_args_no_noise)):
        store_good_dir += 1
        store_good_dir_norm.append(np.linalg.norm(centre_point - func_args[0])
                                   - np.linalg.norm(upd_point - func_args[0]))
        store_good_dir_func.append(f_no_noise(centre_point,
                                              *func_args_no_noise)
                                   - f_no_noise(upd_point,
                                                *func_args_no_noise))

    while flag:
        centre_point = upd_point
        store_norm_grad.append(norm_grad)
        new_func_val = f_val
        step = 1
        (upd_point, f_val,
         func_evals_step,
         func_evals_dir,
         flag, norm_grad) = (calc_first_phase_RSM_LS
                             (centre_point, np.copy(new_func_val), f,
                              func_args, m, const_back, back_tol,
                              const_forward, forward_tol,
                              step, no_vars, region))
        total_func_evals_step += func_evals_step
        total_func_evals_dir += func_evals_dir
        no_iterations += 1
        if (f_no_noise(centre_point, *func_args_no_noise) >
                f_no_noise(upd_point, *func_args_no_noise)):
            store_good_dir += 1
            store_good_dir_norm.append(np.linalg.norm(centre_point -
                                                      func_args[0])
                                       - np.linalg.norm(upd_point -
                                                        func_args[0]))
            store_good_dir_func.append(f_no_noise(centre_point,
                                                  *func_args_no_noise)
                                       - f_no_noise(upd_point,
                                                    *func_args_no_noise))

        if (total_func_evals_step + total_func_evals_dir) >= tol_evals:
            break

    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations, store_good_dir,
            store_good_dir_norm, store_good_dir_func,
            store_norm_grad)
