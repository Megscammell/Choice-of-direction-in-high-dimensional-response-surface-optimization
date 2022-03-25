import numpy as np
import pandas as pd
import sys

import est_dir


def quad_example(n, d, no_vars, region, max_func_evals,
                 seed, lambda_min, lambda_max, mu, sd):
    """
    Example of applying the first phase of RSM for a starting point sampled
    from a Normal distribution with mean zero and identity covariance matrix,
    with a noisy quadratic response function.
    """
    np.random.seed(seed)
    f = est_dir.quad_f_noise
    cov = np.identity(d)
    starting_point = np.random.multivariate_normal(np.zeros((d)), cov)
    minimizer = np.zeros((d,))
    matrix = est_dir.quad_func_params(lambda_min, lambda_max, d)
    func_args = (minimizer, matrix, mu, sd)
    (updated_point,
     sp_func_val,
     fp_func_val,
     full_time,
     total_func_evals_step,
     total_func_evals_dir,
     no_iterations) = (est_dir.rsm_alternative_search_direction
                       (starting_point, f, func_args, n, d,
                        no_vars, region, max_func_evals))

    sp_dist = np.linalg.norm(starting_point - minimizer)
    fp_dist = np.linalg.norm(updated_point - minimizer)
    summary_table = (pd.DataFrame({
                    "Distance between sp and minimizer": [sp_dist],
                    "Distance between fp and minimizer": [fp_dist],
                    "Function value at sp": [sp_func_val],
                    "Function value at fp": [fp_func_val],
                    "Time taken": [full_time],
                    "Step function evals": [total_func_evals_step],
                    "Direction function evals": [total_func_evals_dir],
                    "No iterations": [no_iterations]}))
    summary_table.to_csv('first_phase_RSM_summary_table_d_%s'
                         '_n_%s_quad.csv' % (d, n))


if __name__ == "__main__":
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    no_vars = int(sys.argv[3])
    region = float(sys.argv[4])
    max_func_evals = int(sys.argv[5])
    seed = int(sys.argv[6])
    lambda_min = int(sys.argv[7])
    lambda_max = int(sys.argv[8])
    mu = float(sys.argv[9])
    sd = float(sys.argv[10])
    quad_example(n, d, no_vars, region, max_func_evals,
                 seed, lambda_min, lambda_max, mu, sd)
