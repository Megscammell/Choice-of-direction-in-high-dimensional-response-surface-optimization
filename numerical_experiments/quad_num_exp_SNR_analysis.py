import sys

import numpy as np
import pandas as pd


def mean_good_dir(snr_list, lambda_max_list, good_dir_prop_LS,
                  good_dir_norm_LS, no_its_LS, func_evals_step_LS,
                  func_evals_dir_LS, good_dir_prop_MP,
                  good_dir_norm_MP, no_its_MP, func_evals_step_MP,
                  func_evals_dir_MP, good_dir_prop_XY,
                  good_dir_norm_XY, no_its_XY, func_evals_step_XY,
                  func_evals_dir_XY):
    """
    Generate results focusing on the number of 'good' directions computed
    by various methods.

    Parameters
    ----------
    snr_list : list
               Contains signal to noise ratios.
    lambda_max_list : list
                      Contains a range of largest eigenvalues of the positive
                      definite matrix used to compute the quadratic response
                      function.
    good_dir_no_its_prop_LS : 2-D array
                              Total number of 'good' directions produced by
                              PI_LS. That is, a 'good' direction is when the
                              response function value has improved by
                              moving along the direction.
    good_dir_norm_LS : 2-D array
                       If a 'good' direction is determined for PI_LS,
                       distance of point and minimizer at the k-th
                       iteration subtracted by the distance of point
                       and minimizer at the (k+1)-th iteration is
                       measured. Then an average of all distances is stored.
    no_its_LS : 2-D array
                Total number of iterations of PI_LS until some stopping
                criterion is met for Phase I of RSM.
    func_evals_step_LS : 2-D array
                         Total number of function evaluations taken by PI_LS
                         to compute step length until some stopping criterion
                         is met for Phase I of RSM.
    func_evals_dir_LS : 2-D array
                        Total number of function evaluations taken by PI_LS
                        to compute direction until some stopping criterion
                        is met for Phase I of RSM.
    good_dir_no_its_prop_MP : 2-D array
                              Total number of 'good' directions produced by
                              PI_MPI. That is, a 'good' direction is when the
                              response function value has improved by
                              moving along the direction.
    good_dir_norm_MP : 2-D array
                       If a 'good' direction is determined for PI_MPI,
                       distance of point and minimizer at the k-th
                       iteration subtracted by the distance of point
                       and minimizer at the (k+1)-th iteration is
                       measured. Then an average of all distances is stored.
    no_its_MP : 2-D array
                Total number of iterations of PI_MPI until some stopping
                criterion is met for Phase I of RSM.
    func_evals_step_MP : 2-D array
                         Total number of function evaluations taken by PI_MPI
                         to compute step length until some stopping criterion
                         is met for Phase I of RSM.
    func_evals_dir_MP : 2-D array
                        Total number of function evaluations taken by PI_MPI
                        to compute direction until some stopping criterion
                        is met for Phase I of RSM.
    good_dir_no_its_prop_XY : 2-D array
                              Total number of 'good' directions produced by
                              PI_XY. That is, a 'good' direction is when the
                              response function value has improved by
                              moving along the direction.
    good_dir_norm_XY : 2-D array
                       If a 'good' direction is determined for PI_XY,
                       distance of point and minimizer at the k-th
                       iteration subtracted by the distance of point
                       and minimizer at the (k+1)-th iteration is
                       measured. Then an average of all distances is stored.
    no_its_XY : 2-D array
                Total number of iterations of PI_XY until some stopping
                criterion is met for Phase I of RSM.
    func_evals_step_XY : 2-D array
                         Total number of function evaluations taken by PI_XY
                         to compute step length until some stopping criterion
                         is met for Phase I of RSM.
    func_evals_dir_XY : 2-D array
                        Total number of function evaluations taken by PI_XY
                        to compute direction until some stopping criterion
                        is met for Phase I of RSM.

    Returns
    -------
    arr : 2D array
          Store all relevant results for various lambda_max and snr.
    """

    mean_good_dir_prop = np.zeros((3, len(lambda_max_list), len(snr_list)))
    mean_good_dir_norm = np.zeros((3, len(lambda_max_list), len(snr_list)))
    mean_no_its = np.zeros((3, len(lambda_max_list), len(snr_list)))
    mean_func_evals = np.zeros((3, len(lambda_max_list), len(snr_list)))

    for j in range(len(lambda_max_list)):
        for i in range(len(snr_list)):
            mean_good_dir_prop[0, j, i] = np.mean(good_dir_prop_LS[j, i])
            mean_good_dir_norm[0, j, i] = np.mean(good_dir_norm_LS[j, i])
            mean_no_its[0, j, i] = np.mean(no_its_LS[j, i])
            mean_func_evals[0, j, i] = np.mean(func_evals_step_LS[j, i] +
                                               func_evals_dir_LS[j, i])

            mean_good_dir_prop[1, j, i] = np.mean(good_dir_prop_MP[j, i])
            mean_good_dir_norm[1, j, i] = np.mean(good_dir_norm_MP[j, i])
            mean_no_its[1, j, i] = np.mean(no_its_MP[j, i])
            mean_func_evals[1, j, i] = np.mean(func_evals_step_MP[j, i] +
                                               func_evals_dir_MP[j, i])

            mean_good_dir_prop[2, j, i] = np.mean(good_dir_prop_XY[j, i])
            mean_good_dir_norm[2, j, i] = np.mean(good_dir_norm_XY[j, i])
            mean_no_its[2, j, i] = np.mean(no_its_XY[j, i])
            mean_func_evals[2, j, i] = np.mean(func_evals_step_XY[j, i] +
                                               func_evals_dir_XY[j, i])

    arr1_a = np.concatenate((np.round(mean_good_dir_prop[0], 2),
                             np.round(mean_good_dir_prop[1], 2)), axis=0)
    arr1_b = np.concatenate((arr1_a,
                            np.round(mean_good_dir_prop[2], 2)), axis=0)

    arr2_a = np.concatenate(((np.round(mean_no_its[0], 2),
                            np.round(mean_no_its[1], 2))), axis=0)
    arr2_b = np.concatenate(((arr2_a,
                              np.round(mean_no_its[2], 2))), axis=0)

    arr3_a = np.concatenate(((np.round(mean_good_dir_norm[0], 2),
                            np.round(mean_good_dir_norm[1], 2))), axis=0)
    arr3_b = np.concatenate(((arr3_a,
                            np.round(mean_good_dir_norm[2], 2))), axis=0)

    arr4_a = np.concatenate(((np.round(mean_func_evals[0], 2),
                            np.round(mean_func_evals[1], 2))), axis=0)
    arr4_b = np.concatenate(((arr4_a,
                            np.round(mean_func_evals[2], 2))), axis=0)

    arr_temp1 = np.concatenate((arr1_b, arr2_b), axis=0)
    arr_temp2 = np.concatenate((arr_temp1, arr3_b), axis=0)
    arr = np.concatenate((arr_temp2, arr4_b), axis=0)
    return arr


def noise_list_all_lambda_max(lambda_max_list, n, m, function_type):
    """
    Obtain all standard deviations of errors used to compute noisy
    function values.

    Parameters
    ----------
    lambda_max_list : list
                      Contains a range of largest eigenvalues of the positive
                      definite matrix used to compute the quadratic response
                      function.
    n : integer
        Number of observations for design matrix.
    m : integer
        Number of variables.
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    Returns
    -------
    noise_sd : 2D array
               Store all standard deviations of errors used to compute noisy
               function values.
    """
    noise_sd = np.zeros((len(lambda_max_list), 6))
    index = 0
    for lambda_max in lambda_max_list:
        noise_sd[index] = (np.round(np.genfromtxt(
                           'noise_list_n=%s_m=%s_lambda_max=%s_%s.csv' %
                           (n, m, lambda_max, function_type),
                           delimiter=','), 2))
        index += 1
    return noise_sd


def write_to_latex(arr, title, n, m, function_type, region):
    """
    Write outputs to latex.

    Parameters
    ----------
    arr: array
         Convert arr to a Pandas dataframe and then write to latex.
    title : string
            Name of saved outputs.
    n : integer
        Number of observations for design matrix.
    m : integer
        Number of variables
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    region : float
             Region of exploration around the centre point.

    """
    df = pd.DataFrame(arr)
    df.to_csv(df.to_csv('%s_n=%s_m=%s.csv'
                        % (title, n, m)))
    with open('%s_n=%s_m=%s_%s_%s.tex' %
              (title, n, m, function_type, region), 'w') as tf:
        tf.write(df.to_latex())


if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    region = float(sys.argv[3])
    function_type = str(sys.argv[4])
    func_evals = int(sys.argv[5])

    if func_evals == 0:
        save_outputs = None
    else:
        save_outputs = func_evals
    snr_list = [0.5, 1, 2, 3, 5, 10]
    lambda_max_list = [1, 4, 8]
    no_vars = m
    num_funcs = 100

    fp_norms_LS = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    fp_func_vals_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                num_funcs))
    fp_func_vals_noise_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                      num_funcs))
    func_evals_step_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                   num_funcs))
    func_evals_dir_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                  num_funcs))
    good_dir_prop_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    good_dir_norm_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    no_its_LS = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))

    fp_norms_MP = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    fp_func_vals_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                num_funcs))
    fp_func_vals_noise_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                      num_funcs))
    func_evals_step_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                   num_funcs))
    func_evals_dir_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                  num_funcs))
    good_dir_prop_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    good_dir_norm_MP = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    no_its_MP = np.zeros((len(lambda_max_list), len(snr_list),
                          num_funcs))

    fp_norms_XY = np.zeros((len(lambda_max_list), len(snr_list),
                            num_funcs))
    fp_func_vals_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                num_funcs))
    fp_func_vals_noise_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                      num_funcs))
    func_evals_step_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                   num_funcs))
    func_evals_dir_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                  num_funcs))
    good_dir_prop_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    good_dir_norm_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    no_its_XY = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))

    sp_norms = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    sp_func_vals = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))

    lambda_max_index = 0
    for lambda_max in lambda_max_list:
        fp_norms_LS[lambda_max_index] = (np.genfromtxt(
                                        'fp_norms_LS_n=%s_m=%s_lambda_max=%s'
                                        '_%s_%s_%s.csv' %
                                        (16, m, lambda_max,
                                         10,
                                         region,
                                         function_type),
                                        delimiter=','))

        fp_func_vals_LS[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_LS_n=%s_m=%s_'
                                             'lambda_max=%s_%s_%s_%s.csv' %
                                             (16, m, lambda_max,
                                              10,
                                              region,
                                              function_type),
                                             delimiter=','))

        fp_func_vals_LS[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_LS_n=%s_m=%s_lambda'
                                             '_max=%s_%s_%s_%s.csv' %
                                             (16, m, lambda_max,
                                              10,
                                              region,
                                              function_type),
                                             delimiter=','))

        func_evals_step_LS[lambda_max_index] = (np.genfromtxt(
                                                'func_evals_step_LS_n=%s_m=%s'
                                                '_lambda_max=%s_%s_%s_%s.csv' %
                                                (16, m, lambda_max,
                                                 10,
                                                 region,
                                                 function_type),
                                                delimiter=','))

        func_evals_dir_LS[lambda_max_index] = (np.genfromtxt(
                                              'func_evals_dir_LS_n=%s_m=%s'
                                              '_lambda_max=%s_%s_%s_%s.csv' %
                                              (16, m, lambda_max,
                                               10,
                                               region,
                                               function_type),
                                              delimiter=','))

        good_dir_norm_LS[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_norm_LS_n=%s_m=%s_lam'
                                              'bda_max=%s_%s_%s_%s.csv' %
                                              (16, m, lambda_max,
                                               10,
                                               region,
                                               function_type),
                                              delimiter=','))

        good_dir_prop_LS[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_prop_LS_n=%s_m=%s'
                                              '_lambda_max=%s_%s_%s_%s.csv' %
                                              (16, m, lambda_max,
                                               10,
                                               region,
                                               function_type),
                                              delimiter=','))

        no_its_LS[lambda_max_index] = (np.genfromtxt(
                                       'no_its_LS_n=%s_m=%s_lambda_max=%s'
                                       '_%s_%s_%s.csv' %
                                       (16, m, lambda_max,
                                        10,
                                        region,
                                        function_type),
                                       delimiter=','))

        fp_norms_MP[lambda_max_index] = (np.genfromtxt(
                                         'fp_norms_MP_n=%s_m=%s_lambda_max'
                                         '=%s_%s_%s_%s_%s.csv' %
                                         (n, m, lambda_max,
                                          no_vars,
                                          region,
                                          function_type,
                                          save_outputs),
                                         delimiter=','))

        fp_func_vals_MP[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_MP_n=%s_m=%s_lambda'
                                             '_max=%s_%s_%s_%s_%s.csv' %
                                             (n, m, lambda_max,
                                              no_vars,
                                              region,
                                              function_type,
                                              save_outputs),
                                             delimiter=','))

        func_evals_step_MP[lambda_max_index] = (np.genfromtxt(
                                                'func_evals_step_MP_n=%s_m=%s'
                                                '_lambda_max=%s_%s_%s'
                                                '_%s_%s.csv' %
                                                (n, m, lambda_max,
                                                 no_vars,
                                                 region,
                                                 function_type,
                                                 save_outputs),
                                                delimiter=','))

        func_evals_dir_MP[lambda_max_index] = (np.genfromtxt(
                                               'func_evals_dir_MP_n=%s_m=%s_'
                                               'lambda_max=%s_%s_%s'
                                               '_%s_%s.csv' %
                                               (n, m, lambda_max,
                                                no_vars,
                                                region,
                                                function_type,
                                                save_outputs),
                                               delimiter=','))

        good_dir_norm_MP[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_norm_MP_n=%s_m=%s_lamb'
                                              'da_max=%s_%s_%s_%s_%s.csv' %
                                              (n, m, lambda_max,
                                               no_vars,
                                               region,
                                               function_type,
                                               save_outputs),
                                              delimiter=','))

        good_dir_prop_MP[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_prop_MP_n=%s_m=%s_lamb'
                                              'da_max=%s_%s_%s_%s_%s.csv' %
                                              (n, m, lambda_max,
                                               no_vars,
                                               region,
                                               function_type,
                                               save_outputs),
                                              delimiter=','))

        no_its_MP[lambda_max_index] = (np.genfromtxt(
                                       'no_its_MP_n=%s_m=%s_lambda_max=%s_%s'
                                       '_%s_%s_%s.csv' %
                                       (n, m, lambda_max,
                                        no_vars,
                                        region,
                                        function_type,
                                        save_outputs),
                                       delimiter=','))

        fp_norms_XY[lambda_max_index] = (np.genfromtxt(
                                         'fp_norms_XY_n=%s_m=%s_lambda_max=%s'
                                         '_%s_%s_%s_%s.csv' %
                                         (n, m, lambda_max,
                                          no_vars,
                                          region,
                                          function_type,
                                          save_outputs),
                                         delimiter=','))

        fp_func_vals_XY[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_XY_n=%s_m=%s_lambda'
                                             '_max=%s_%s_%s_%s_%s.csv' %
                                             (n, m, lambda_max,
                                              no_vars,
                                              region,
                                              function_type,
                                              save_outputs),
                                             delimiter=','))

        func_evals_step_XY[lambda_max_index] = (np.genfromtxt(
                                                'func_evals_step_XY_n=%s_m=%s'
                                                '_lambda_max=%s_%s_%s_%s'
                                                '_%s.csv' %
                                                (n, m, lambda_max,
                                                 no_vars,
                                                 region,
                                                 function_type,
                                                 save_outputs),
                                                delimiter=','))

        func_evals_dir_XY[lambda_max_index] = (np.genfromtxt(
                                               'func_evals_dir_XY_n=%s_m=%s'
                                               '_lambda_max=%s_%s_%s_%s'
                                               '_%s.csv' %
                                               (n, m, lambda_max,
                                                no_vars,
                                                region,
                                                function_type,
                                                save_outputs),
                                               delimiter=','))

        good_dir_norm_XY[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_norm_XY_n=%s_m=%s_lamb'
                                              'da_max=%s_%s_%s_%s_%s.csv' %
                                              (n, m, lambda_max,
                                               no_vars,
                                               region,
                                               function_type,
                                               save_outputs),
                                              delimiter=','))

        good_dir_prop_XY[lambda_max_index] = (np.genfromtxt(
                                              'good_dir_prop_XY_n=%s_m=%s_la'
                                              'mbda_max=%s_%s_%s_%s_%s.csv' %
                                              (n, m, lambda_max,
                                               no_vars,
                                               region,
                                               function_type,
                                               save_outputs),
                                              delimiter=','))

        no_its_XY[lambda_max_index] = (np.genfromtxt(
                                       'no_its_XY_n=%s_m=%s_lambda_max=%s_%s'
                                       '_%s_%s_%s.csv' %
                                       (n, m, lambda_max,
                                        no_vars,
                                        region,
                                        function_type,
                                        save_outputs),
                                       delimiter=','))

        sp_norms[lambda_max_index] = (np.genfromtxt(
                                      'sp_norms_LS_n=%s_m=%s_lambda'
                                      '_max=%s_%s.csv' %
                                      (n, m, lambda_max, function_type),
                                      delimiter=','))

        sp_func_vals[lambda_max_index] = (np.genfromtxt(
                                          'sp_func_vals_LS_n=%s_m=%s'
                                          '_lambda_max=%s_%s.csv' %
                                          (n, m, lambda_max, function_type),
                                          delimiter=','))
        lambda_max_index += 1

    arr_mean_dir = mean_good_dir(snr_list, lambda_max_list, good_dir_prop_LS,
                                 good_dir_norm_LS, no_its_LS,
                                 func_evals_step_LS, func_evals_dir_LS,
                                 good_dir_prop_MP, good_dir_norm_MP, no_its_MP,
                                 func_evals_step_MP, func_evals_dir_MP,
                                 good_dir_prop_XY, good_dir_norm_XY, no_its_XY,
                                 func_evals_step_XY, func_evals_dir_XY)
    write_to_latex(arr_mean_dir, 'mean_good_dir',
                   n, m, function_type, region)

    noise_sd = noise_list_all_lambda_max(lambda_max_list, n, m, function_type)
    write_to_latex(noise_sd, 'noise_sd', n, m, function_type, region)
