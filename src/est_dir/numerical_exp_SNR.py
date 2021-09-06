import numpy as np
import est_dir
import tqdm


def generate_func_params(j, cov, m, lambda_max):
    """
    Generate function parameters for numerical experiments,
    with direction compute by either LS, XY or MP.

    Parameters
    ----------
    j : integer
        Function index used to determine the random seed.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    m : integer
        Number of variables
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    noise_sd : float
               Standard deviation to generate noise within response
               function evaluations.

    Returns
    -------
    centre_point : 1-D array
                   Point used to apply Phase I of RSM.
    minimizer : 1-D array
                Minimizer of the quadratic response function.
    matrix : 2-D array
             Positive definite matrix used to compute the
             quadratic response function values.
    func_args : tuple
                Arguments passed to the function f with noise.
    func_args_no_noise : tuple
                         Arguments passed to the function f without noise.
    """

    seed = j * 50
    np.random.seed(seed)
    centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
    minimizer = np.zeros((m, ))
    matrix = est_dir.quad_func_params(1, lambda_max, m)
    return centre_point, minimizer, matrix


def calc_initial_func_values(m, num_funcs, lambda_max, cov, f_no_noise):
    """
    Store initial function values at each centre_point. Will be used to
    compute the standard deviation of the errors for the noisy function
    values.

    Parameters
    ----------
    m : integer
        Number of variables
    num_funcs : integer
                Number of different function parameters to generate.
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.

    Returns
    -------
    sp_func_vals : 1-D array
                   Stores all initial function values at centre_point for
                   each set of generated function parameters.
    """

    sp_func_vals = np.zeros((num_funcs))
    for j in tqdm.tqdm(range(num_funcs)):
        (centre_point,
         minimizer,
         matrix) = generate_func_params(j, cov, m, lambda_max)
        func_args_no_noise = (minimizer, matrix)
        sp_func_vals[j] = f_no_noise(centre_point, *func_args_no_noise)
    return sp_func_vals


def compute_var_quad_form(snr_list, sp_func_vals, region):
    """
    Compute standard deviation of errors used to compute noisy function
    values.

    Parameters
    ----------
    m : integer
        Number of variables
    snr_list : list
               Predefined list of various values for SNR.
    sp_func_vals : 1-D array
                   Stores all initial function values at centre_point for
                   each set of generated function parameters.
    region : float
             Region of exploration around the centre point.

    Returns
    -------
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
    """

    noise_list = np.zeros((len(snr_list)))
    index = 0
    for snr in snr_list:
        noise_list[index] = np.sqrt(np.var(sp_func_vals * region) / snr)
        index += 1
    return noise_list


def num_exp_SNR_LS(f, f_no_noise, m, num_funcs, lambda_max, cov,
                   noise_list, region, function_type):
    """
    Numerical experiments for LS - with various SNR.

    Parameters
    ----------
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.
    m : integer
        Number of variables
    num_funcs : integer
                Number of different function parameters to generate.
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
    region : float
             Region of exploration around the centre point.
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    Returns
    -------
    sp_norms_LS : 2-D array
                  Distances between starting point and minimizer.
    sp_func_vals_LS : 2-D array
                  Response function values without noise at starting points.
    fp_norms_LS : 2-D array
                  Distances between final point and minimizer.
    fp_func_vals_LS : 2-D array
                      Response function values without noise at final points.
    sp_func_vals_noise_LS : 2-D array
                            Response function values with noise at starting
                            points.
    fp_func_vals_noise_LS : 2-D array
                            Response function values with noise at final
                            points.
    time_taken_LS : 2-D array
                    Time taken to until stopping criterion is met for Phase I
                    of RSM.
    func_evals_step_LS : 2-D array
                         Total number of function evaluations taken to compute
                         step length until some stopping criterion is met for
                         Phase I of RSM.
    func_evals_dir_LS : 2-D array
                        Total number of function evaluations taken to compute
                        direction until some stopping criterion is met for
                        Phase I of RSM.
    no_its_LS : 2-D array
                Total number of iterations of until some stopping
                criterion is met for Phase I of RSM.
    good_dir_no_its_prop_LS : 2-D array
                              Total number of 'good' directions produced.
                              That is, a 'good' direction is when the response
                              function value has improved by moving along the
                              direction.
    good_dir_norm_LS : 2-D array
                        If a 'good' direction is determined,
                        distance of point and minimizer at the k-th
                        iteration subtracted by the distance of point
                        and minimizer at the (k+1)-th iteration is
                        measured. Then an average of all distances is stored.
    good_dir_func_LS : 2-D array
                       If a 'good' direction is determined,
                       compute the response function value with point
                       at the k-th iteration, subtracted by the response
                       function value with point at the
                       (k+1)-th iteration. Then an average of all
                       response function values is stored.
    mean_norm_grad_LS : 2-D array
                        Average norm of direction relative to no_vars and m at
                        each iteration.

    """
    n = 16
    no_vars = 10
    sp_norms_LS = np.zeros((noise_list.shape[0], num_funcs))
    fp_norms_LS = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_noise_LS = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_noise_LS = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_LS = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_LS = np.zeros((noise_list.shape[0], num_funcs))
    time_taken_LS = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_step_LS = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_dir_LS = np.zeros((noise_list.shape[0], num_funcs))
    no_its_LS = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_no_its_prop_LS = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_norm_LS = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_func_LS = np.zeros((noise_list.shape[0], num_funcs))
    mean_norm_grad_LS = np.zeros((noise_list.shape[0], num_funcs))

    for index_noise in range(noise_list.shape[0]):
        noise_sd = noise_list[index_noise]
        for j in tqdm.tqdm(range(num_funcs)):
            (centre_point,
             minimizer,
             matrix) = generate_func_params(j, cov, m, lambda_max)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            sp_norms_LS[index_noise, j] = np.linalg.norm(minimizer -
                                                         centre_point)
            sp_func_vals_LS[index_noise, j] = f_no_noise(centre_point,
                                                         minimizer, matrix)
            seed = j * 50
            np.random.seed(seed + 1)
            (upd_point,
             sp_func_vals_noise_LS[index_noise, j],
             fp_func_vals_noise_LS[index_noise, j],
             time_taken_LS[index_noise, j],
             func_evals_step_LS[index_noise, j],
             func_evals_dir_LS[index_noise, j],
             no_its_LS[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_LS(
                                 centre_point, f, func_args, m,
                                 f_no_noise, func_args_no_noise,
                                 region))
            fp_norms_LS[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_LS[index_noise, j] = f_no_noise(upd_point, minimizer,
                                                         matrix)
            good_dir_no_its_prop_LS[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_LS[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_LS[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_LS[index_noise, j] = np.mean(store_good_dir_func)

    option_t = 'LS'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type),
               sp_norms_LS, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type),
               sp_func_vals_LS, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), fp_norms_LS,
               delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), fp_func_vals_LS,
               delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), func_evals_step_LS,
               delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), func_evals_dir_LS,
               delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), time_taken_LS,
               delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), fp_func_vals_noise_LS,
               delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), good_dir_no_its_prop_LS,
               delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), no_its_LS,
               delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), good_dir_norm_LS,
               delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region,  function_type), good_dir_func_LS,
               delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type), mean_norm_grad_LS,
               delimiter=',')

    return (sp_norms_LS, sp_func_vals_LS,
            fp_norms_LS, fp_func_vals_LS,
            sp_func_vals_noise_LS,
            fp_func_vals_noise_LS,
            time_taken_LS, func_evals_step_LS,
            func_evals_dir_LS, no_its_LS,
            good_dir_no_its_prop_LS,
            good_dir_norm_LS, good_dir_func_LS,
            mean_norm_grad_LS)


def num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs, lambda_max, cov, noise_list,
                   no_vars, region, max_func_evals_list, function_type,
                   store_max_func_evals):
    """
    Numerical experiments for XY - with various SNR.

    Parameters
    ----------
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.
    n : integer
        Number of observations for design matrix.
    m : integer
        Number of variables
    num_funcs : integer
                Number of different function parameters to generate.
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
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
    max_func_evals_list : list
                          Number of function evaluations permitted for each
                          SNR.
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    Returns
    -------
    sp_norms_XY : 2-D array
                  Distances between starting point and minimizer.
    sp_func_vals_XY : 2-D array
                      Response function values without noise at starting
                      points.
    fp_norms_XY : 2-D array
                  Distances between final point and minimizer.
    fp_func_vals_XY : 2-D array
                      Response function values without noise at final points.
    sp_func_vals_noise_XY : 2-D array
                            Response function values with noise at starting
                            points.
    fp_func_vals_noise_XY : 2-D array
                            Response function values with noise at final
                            points.
    time_taken_XY : 2-D array
                    Time taken to until stopping criterion is met for Phase I
                    of RSM.
    func_evals_step_XY : 2-D array
                         Total number of function evaluations taken to compute
                         step length until some stopping criterion is met for
                         Phase I of RSM.
    func_evals_dir_XY : 2-D array
                        Total number of function evaluations taken to compute
                        direction until some stopping criterion is met for
                        Phase I of RSM.
    no_its_XY : 2-D array
                Total number of iterations of until some stopping
                criterion is met for Phase I of RSM.
    good_dir_no_its_prop_XY : 2-D array
                              Total number of 'good' directions produced.
                              That is, a 'good' direction is when the
                              response function value has improved by
                              moving along the direction.
    good_dir_norm_XY : 2-D array
                       If a 'good' direction is determined,
                       distance of point and minimizer at the k-th
                       iteration subtracted by the distance of point
                       and minimizer at the (k+1)-th iteration is
                       measured. Then an average of all distances is stored.
    good_dir_func_XY : 2-D array
                       If a 'good' direction is determined,
                       compute the response function value with point
                       at the k-th iteration, subtracted by the response
                       function value with point at the
                       (k+1)-th iteration. Then an average of all
                       response function values is stored.
    mean_norm_grad_XY : 2-D array
                        Average norm of direction relative to no_vars and m at
                        each iteration.
    """
    sp_norms_XY = np.zeros((noise_list.shape[0], num_funcs))
    fp_norms_XY = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_noise_XY = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_noise_XY = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_XY = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_XY = np.zeros((noise_list.shape[0], num_funcs))
    time_taken_XY = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_step_XY = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_dir_XY = np.zeros((noise_list.shape[0], num_funcs))
    no_its_XY = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_no_its_prop_XY = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_norm_XY = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_func_XY = np.zeros((noise_list.shape[0], num_funcs))
    mean_norm_grad_XY = np.zeros((noise_list.shape[0], num_funcs))

    for index_noise in range(noise_list.shape[0]):
        max_func_evals = max_func_evals_list[index_noise]
        noise_sd = noise_list[index_noise]
        for j in tqdm.tqdm(range(num_funcs)):
            (centre_point,
             minimizer,
             matrix) = generate_func_params(j, cov, m, lambda_max)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            sp_norms_XY[index_noise, j] = np.linalg.norm(minimizer -
                                                         centre_point)
            sp_func_vals_XY[index_noise, j] = f_no_noise(centre_point,
                                                         minimizer, matrix)
            seed = j * 50
            np.random.seed(seed + 1)
            (upd_point,
             sp_func_vals_noise_XY[index_noise, j],
             fp_func_vals_noise_XY[index_noise, j],
             time_taken_XY[index_noise, j],
             func_evals_step_XY[index_noise, j],
             func_evals_dir_XY[index_noise, j],
             no_its_XY[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_XY(
                                 centre_point, f, func_args, n, m, f_no_noise,
                                 func_args_no_noise, no_vars, region,
                                 max_func_evals))
            fp_norms_XY[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_XY[index_noise, j] = f_no_noise(upd_point, minimizer,
                                                         matrix)
            good_dir_no_its_prop_XY[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_XY[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_XY[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_XY[index_noise, j] = np.mean(store_good_dir_func)

    option_t = 'XY'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type,
                store_max_func_evals),
               sp_norms_XY, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type,
                store_max_func_evals),
               sp_func_vals_XY, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               fp_norms_XY,
               delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               fp_func_vals_XY,
               delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               func_evals_step_XY,
               delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               func_evals_dir_XY,
               delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals), time_taken_XY,
               delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s'
               '_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               fp_func_vals_noise_XY,
               delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               good_dir_no_its_prop_XY,
               delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               no_its_XY,
               delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals), good_dir_norm_XY,
               delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals), good_dir_func_XY,
               delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                region, function_type, store_max_func_evals),
               mean_norm_grad_XY,
               delimiter=',')

    return (sp_norms_XY, sp_func_vals_XY,
            fp_norms_XY, fp_func_vals_XY,
            sp_func_vals_noise_XY,
            fp_func_vals_noise_XY,
            time_taken_XY, func_evals_step_XY,
            func_evals_dir_XY, no_its_XY,
            good_dir_no_its_prop_XY,
            good_dir_norm_XY, good_dir_func_XY,
            mean_norm_grad_XY)


def num_exp_SNR_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                   noise_list, no_vars, region, max_func_evals_list,
                   type_inverse, function_type, store_max_func_evals):
    """
    Numerical experiments for MP - with various SNR.

    Parameters
    ----------
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.
    n : integer
        Number of observations for design matrix.
    m : integer
        Number of variables
    num_funcs : integer
                Number of different function parameters to generate.
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
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
    max_func_evals_list : list
                          Number of function evaluations permitted for each
                          SNR.
    type_inverse : string
                   Determine whether to perform a left or right inverse.
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    Returns
    -------
    sp_norms_MP : 2-D array
                  Distances between starting point and minimizer.
    sp_func_vals_MP : 2-D array
                      Response function values without noise at starting
                      points.
    fp_norms_MP : 2-D array
                  Distances between final point and minimizer.
    fp_func_vals_MP : 2-D array
                      Response function values without noise at final points.
    sp_func_vals_noise_MP : 2-D array
                            Response function values with noise at starting
                            points.
    fp_func_vals_noise_MP : 2-D array
                            Response function values with noise at final
                            points.
    time_taken_MP : 2-D array
                    Time taken to until stopping criterion is met for Phase I
                    of RSM.
    func_evals_step_MP : 2-D array
                         Total number of function evaluations taken to compute
                         step length until some stopping criterion is met for
                         Phase I of RSM.
    func_evals_dir_MP : 2-D array
                        Total number of function evaluations taken to compute
                        direction until some stopping criterion is met for
                        Phase I of RSM.
    no_its_MP : 2-D array
                Total number of iterations of until some stopping
                criterion is met for Phase I of RSM.
    good_dir_no_its_prop_MP : 2-D array
                              Total number of 'good' directions produced.
                              That is, a 'good' direction is when the
                              response function value has improved by
                              moving along the direction.
    good_dir_norm_MP : 2-D array
                       If a 'good' direction is determined,
                       distance of point and minimizer at the k-th
                       iteration subtracted by the distance of point
                       and minimizer at the (k+1)-th iteration is
                       measured. Then an average of all distances is stored.
    good_dir_func_MP : 2-D array
                       If a 'good' direction is determined,
                       compute the response function value with point
                       at the k-th iteration, subtracted by the response
                       function value with point at the
                       (k+1)-th iteration. Then an average of all
                       response function values is stored.
    mean_norm_grad_MP : 2-D array
                        Average norm of direction relative to no_vars and m at
                        each iteration.
    """
    sp_norms_MP = np.zeros((noise_list.shape[0], num_funcs))
    fp_norms_MP = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_noise_MP = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_noise_MP = np.zeros((noise_list.shape[0], num_funcs))
    sp_func_vals_MP = np.zeros((noise_list.shape[0], num_funcs))
    fp_func_vals_MP = np.zeros((noise_list.shape[0], num_funcs))
    time_taken_MP = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_step_MP = np.zeros((noise_list.shape[0], num_funcs))
    func_evals_dir_MP = np.zeros((noise_list.shape[0], num_funcs))
    no_its_MP = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_no_its_prop_MP = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_norm_MP = np.zeros((noise_list.shape[0], num_funcs))
    good_dir_func_MP = np.zeros((noise_list.shape[0], num_funcs))
    mean_norm_grad_MP = np.zeros((noise_list.shape[0], num_funcs))

    for index_noise in range(noise_list.shape[0]):
        max_func_evals = max_func_evals_list[index_noise]
        noise_sd = noise_list[index_noise]
        for j in tqdm.tqdm(range(num_funcs)):
            (centre_point,
             minimizer,
             matrix) = generate_func_params(j, cov, m, lambda_max)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            sp_norms_MP[index_noise, j] = np.linalg.norm(minimizer -
                                                         centre_point)
            sp_func_vals_MP[index_noise, j] = f_no_noise(centre_point,
                                                         minimizer, matrix)
            seed = j * 50
            np.random.seed(seed + 1)
            (upd_point,
             sp_func_vals_noise_MP[index_noise, j],
             fp_func_vals_noise_MP[index_noise, j],
             time_taken_MP[index_noise, j],
             func_evals_step_MP[index_noise, j],
             func_evals_dir_MP[index_noise, j],
             no_its_MP[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_MP(
                                 centre_point, f, func_args, n, m, f_no_noise,
                                 func_args_no_noise, no_vars, region,
                                 max_func_evals, type_inverse))
            fp_norms_MP[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_MP[index_noise, j] = f_no_noise(upd_point, minimizer,
                                                         matrix)
            good_dir_no_its_prop_MP[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_MP[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_MP[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_MP[index_noise, j] = np.mean(store_good_dir_func)

    option_t = 'MP'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type,
                store_max_func_evals),
               sp_norms_MP, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, function_type,
                store_max_func_evals),
               sp_func_vals_MP, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               fp_norms_MP,
               delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               fp_func_vals_MP,
               delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s'
               '_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               func_evals_step_MP,
               delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               func_evals_dir_MP,
               delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               time_taken_MP,
               delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s_%s'
               '_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               fp_func_vals_noise_MP,
               delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               good_dir_no_its_prop_MP,
               delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               no_its_MP,
               delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               good_dir_norm_MP,
               delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               good_dir_func_MP,
               delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, no_vars,
                type_inverse, region, function_type, store_max_func_evals),
               mean_norm_grad_MP,
               delimiter=',')

    return (sp_norms_MP, sp_func_vals_MP,
            fp_norms_MP, fp_func_vals_MP,
            sp_func_vals_noise_MP,
            fp_func_vals_noise_MP,
            time_taken_MP, func_evals_step_MP,
            func_evals_dir_MP, no_its_MP,
            good_dir_no_its_prop_MP,
            good_dir_norm_MP, good_dir_func_MP,
            mean_norm_grad_MP)


def quad_LS_XY_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov, noise_list,
                  no_vars, region, function_type, type_inverse,
                  store_max_func_evals):
    """
    Runs all numerical experiments.

    Parameters
    ----------
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    f_no_noise : function
                 response function with no noise.

                `f_no_noise(point, *func_args_no_noise) -> float`

                where point` is a 1-D array with shape(d, ) and
                func_args_no_noise is a tuple of arguments needed to compute
                the response function value.
    n : integer
        Number of observations for design matrix.
    m : integer
        Number of variables
    num_funcs : integer
                Number of different function parameters to generate.
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
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
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    type_inverse : string
                   Determine whether to perform a left or right inverse for
                   PI-MP.
    store_max_func_evals : either a list or None
                           If a list is provided then, PI-MP and PI-XY will use
                           a predefined number of function evaluations to
                           compute step length and direction before stopping.
    """
    if np.all(store_max_func_evals) == None:
        results_LS = num_exp_SNR_LS(f, f_no_noise, m, num_funcs, lambda_max,
                                    cov, noise_list, region, function_type)
        total_func_evals = results_LS[7] + results_LS[8]

        max_func_evals_list = np.zeros(len(noise_list))
        for j in range(len(noise_list)):
            max_func_evals_list[j] = np.mean(total_func_evals[j])

        results_MP = num_exp_SNR_MP(f, f_no_noise, n, m, num_funcs,
                                    lambda_max, cov, noise_list,
                                    no_vars, region,
                                    max_func_evals_list, type_inverse,
                                    function_type, store_max_func_evals)

        results_XY = num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs, lambda_max,
                                    cov, noise_list, no_vars, region,
                                    max_func_evals_list, function_type,
                                    store_max_func_evals)

        assert(np.all(np.round(results_XY[0], 5) ==
                      np.round(results_LS[0], 5)))
        assert(np.all(np.round(results_XY[0], 5) ==
                      np.round(results_MP[0], 5)))
        assert(np.all(np.round(results_XY[1], 5) ==
                      np.round(results_LS[1], 5)))
        assert(np.all(np.round(results_XY[1], 5) ==
                      np.round(results_MP[1], 5)))

    else:
        max_func_evals_list = np.array(store_max_func_evals)
        results_MP = num_exp_SNR_MP(f, f_no_noise, n, m, num_funcs, lambda_max,
                                    cov, noise_list, no_vars, region,
                                    max_func_evals_list, type_inverse,
                                    function_type, store_max_func_evals[0])

        results_XY = num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs, lambda_max,
                                    cov, noise_list, no_vars, region,
                                    max_func_evals_list, function_type,
                                    store_max_func_evals[0])
        assert(np.all(np.round(results_XY[0], 5) ==
                      np.round(results_MP[0], 5)))
        assert(np.all(np.round(results_XY[1], 5) ==
                      np.round(results_MP[1], 5)))

    np.savetxt('max_func_evals_list_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
               (n, m, lambda_max, no_vars,
                region, function_type),
               max_func_evals_list,
               delimiter=',')

    for j in range(len(noise_list)-1):
        for i in range(j, len(noise_list)):
            assert(np.all(np.round(results_XY[1][j], 5) ==
                          np.round(results_XY[1][i], 5)))
            assert(np.all(np.round(results_XY[0][j], 5) ==
                          np.round(results_XY[0][i], 5)))
