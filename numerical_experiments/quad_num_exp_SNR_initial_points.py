import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import est_dir
import tqdm


if __name__ == "__main__":
    n =  int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    domain = (0, 5)
    num_funcs = 100
    snr_list = [0.01, 0.1, 0.2, 0.3]
    #Obtain data results
    sp_func_vals = est_dir.calc_initial_func_values(n, m, num_funcs, lambda_max, domain)
    #Save relevant data results
    np.savetxt('sp_func_vals_n=%s_m=%s_lambda_max=%s_initial.csv' %
               (n, m, lambda_max), sp_func_vals, delimiter=',')
    noise_list = est_dir.compute_var_quad_form(n, m, lambda_max, snr_list, sp_func_vals)
    np.savetxt('noise_list_n=%s_m=%s_lambda_max=%s.csv' %
               (n, m, lambda_max), noise_list, delimiter=',')
