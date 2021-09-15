import sys

import numpy as np
import est_dir
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    region = float(sys.argv[4])
    function_type = str(sys.argv[5])
    type_inverse = str(sys.argv[6])
    func_evals = int(sys.argv[7])

    if function_type == 'quad':
        f = est_dir.quad_f_noise
        f_no_noise = est_dir.quad_f
    elif function_type == 'sqr_quad':
        f = est_dir.sqrt_quad_f_noise
        f_no_noise = est_dir.sqrt_quad_f
    elif function_type == 'squ_quad':
        f = est_dir.square_quad_f_noise
        f_no_noise = est_dir.square_quad_f
    else:
        raise ValueError('Incorrect function choice')
    num_funcs = 100
    cov = np.identity(m)
    no_vars = m
    snr_list = [0.5, 1, 2, 3, 5, 10]

    if func_evals == 0:
        store_max_func_evals = None
        save_outputs = None
    else:
        store_max_func_evals = np.repeat(func_evals, len(snr_list))
        save_outputs = store_max_func_evals[0]

    sp_func_vals_init = (est_dir.calc_initial_func_values(
                         m, num_funcs, lambda_max, cov, f_no_noise))

    noise_list = est_dir.compute_var_quad_form(snr_list, sp_func_vals_init,
                                               region)
    np.savetxt('noise_list_n=%s_m=%s_lambda_max=%s_%s.csv' %
               (n, m, lambda_max, function_type), noise_list, delimiter=',')

    est_dir.quad_LS_XY_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                          noise_list, no_vars, region, function_type,
                          type_inverse, store_max_func_evals)
