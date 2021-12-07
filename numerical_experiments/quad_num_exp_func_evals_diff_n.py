import sys

import numpy as np
import pandas as pd


def compute_avg_func_evals(lambda_max_list, n_list, m, no_vars,
                           save_outputs):
    """
    Compute average function evaluations for PI_MPI and PI_XY with
    SNR = (0.5, 2).

    Parameters
    ----------
    lambda_max_list : list
                      Contains a range of largest eigenvalues of the positive
                      definite matrix used to compute the quadratic response
                      function.
    n_list : list
             Contains a range of values for n.
    m : integer
        Number of variables
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
              a tuple of arguments needed to compute the response function
              value.
    save_outputs : integer or None
                   Number of function evaluations if PI_MPI and PI_XY are
                   applied. Otherwise set to None if PI_LS is also applied.
    Returns
    -------
    arr : 2D array
          Contains the average function evaluations computed by PI_MPI and
          PI_XY for various n and lambda_max.

    """
    func_evals_snr_MP = np.zeros((2, len(lambda_max_list), len(n_list)))
    func_evals_snr_XY = np.zeros((2, len(lambda_max_list), len(n_list)))

    for j in range(len(lambda_max_list)):
        for i in range(len(n_list)):
            lambda_max = lambda_max_list[j]
            n = n_list[i]
            func_evals_step_MP = (np.genfromtxt(
                                 'func_evals_step_MP_n=%s_m=%s'
                                 '_lambda_max=%s_%s_%s'
                                 '_%s_%s.csv' %
                                 (n, m, lambda_max,
                                  no_vars,
                                  region,
                                  function_type,
                                  save_outputs),
                                 delimiter=','))

            func_evals_dir_MP = (np.genfromtxt(
                                'func_evals_dir_MP_n=%s_m=%s_'
                                'lambda_max=%s_%s_%s'
                                '_%s_%s.csv' %
                                (n, m, lambda_max,
                                 no_vars,
                                 region,
                                 function_type,
                                 save_outputs),
                                delimiter=','))

            func_evals_step_XY = (np.genfromtxt(
                                'func_evals_step_XY_n=%s_m=%s'
                                '_lambda_max=%s_%s_%s_%s_%s.csv' %
                                (n, m, lambda_max,
                                 no_vars,
                                 region,
                                 function_type,
                                 save_outputs),
                                delimiter=','))

            func_evals_dir_XY = (np.genfromtxt(
                                'func_evals_dir_XY_n=%s_m=%s'
                                '_lambda_max=%s_%s_%s_%s_%s.csv' %
                                (n, m, lambda_max,
                                 no_vars,
                                 region,
                                 function_type,
                                 save_outputs),
                                delimiter=','))

            func_evals_snr_MP[0, j, i] = (np.mean(func_evals_dir_MP[0] +
                                                  func_evals_step_MP[0]))
            func_evals_snr_MP[1, j, i] = (np.mean(func_evals_dir_MP[2] +
                                                  func_evals_step_MP[2]))

            func_evals_snr_XY[0, j, i] = (np.mean(func_evals_dir_XY[0] +
                                                  func_evals_step_XY[0]))
            func_evals_snr_XY[1, j, i] = (np.mean(func_evals_dir_XY[2] +
                                                  func_evals_step_XY[2]))
    arr = np.concatenate((func_evals_snr_MP.reshape(6, 5),
                          func_evals_snr_XY.reshape(6, 5)), axis=0)
    return arr


def write_to_latex(arr, title, m, function_type, region, save_outputs):
    """
    Write outputs to latex.

    Parameters
    ----------
    arr: array
         Convert arr to a Pandas dataframe and then write to latex.
    title : string
            Name of saved outputs.
    m : integer
        Number of variables
    function_type : string
                    Either function_type = 'quad', 'sqr_quad' or
                    'squ_quad'.
    region : float
             Region of exploration around the centre point.
    save_outputs : integer or None
                   Number of function evaluations if PI_MPI and PI_XY are
                   applied. Otherwise set to None if PI_LS is also applied.

    """
    df = pd.DataFrame(arr)
    df.to_csv(df.to_csv('%s_m=%s_%s_%s_%s.csv'
                        % (title, m, function_type, region, save_outputs)))
    with open('%s_m=%s_%s_%s_%s.tex' %
              (title, m, function_type, region, save_outputs), 'w') as tf:
        tf.write(df.to_latex())


if __name__ == "__main__":
    m = int(sys.argv[1])
    region = float(sys.argv[2])
    function_type = str(sys.argv[3])
    func_evals = int(sys.argv[4])
    lambda_max_list = [1, 4, 8]
    n_list = [16, 32, 50, 100, 200]
    no_vars = m
    num_funcs = 100

    if func_evals == 0:
        save_outputs = None
    else:
        save_outputs = func_evals

    arr = compute_avg_func_evals(lambda_max_list, n_list, m, no_vars,
                                 save_outputs)
    write_to_latex(arr, 'func_evals_diff_n', m, function_type,
                   region, save_outputs)
