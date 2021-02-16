import sys

import numpy as np
import est_dir
import tqdm


if __name__ == "__main__":
    n =  int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    noise_list = np.genfromtxt('noise_list_n=%s_m=%s_lambda_max=%s.csv' %
                                (n, m, lambda_max), delimiter=',')
    if lambda_max == 1:
        if m == 100:
            max_func_evals_list = [10000, 20000, 30000, 40000]
        elif m == 200:
            max_func_evals_list = [10000, 12000, 15000, 20000]
    elif lambda_max == 8:
        if m == 100:
            max_func_evals_list = [40000, 50000, 60000, 80000]
        elif m == 200:
            max_func_evals_list = [20000, 30000, 40000, 50000]
    elif lambda_max == 16:
        if m == 100:
            max_func_evals_list = [30000, 50000, 100000, 120000]
        elif m == 200:
            max_func_evals_list = [30000, 50000, 80000, 100000]
    elif lambda_max == 64:
        if m == 100:
            max_func_evals_list = [20000, 60000, 100000, 120000]
        elif m == 200:
            max_func_evals_list = [50000, 80000, 100000, 140000]
    else:
        raise ValueError('Incorrect lambda_max')
    
    domain = (0, 5)
    num_funcs = 100

    #Obtain data results
    (sp_norms, sp_func_vals,
     fp_norms, fp_func_vals,
     sp_func_vals_noise,
     fp_func_vals_noise,
     time_taken,
     func_evals_step,
     func_evals_dir,
     no_its) = est_dir.num_exp_SNR(n, m, num_funcs, lambda_max, domain, max_func_evals_list,
                             noise_list)
    
    #Save relevant data results
    np.savetxt('sp_norms_n=%s_m=%s_lambda_max=%s.csv' %
               (n, m, lambda_max), sp_norms, delimiter=',')
    np.savetxt('sp_func_vals_n=%s_m=%s_lambda_max=%s.csv' %
               (n, m, lambda_max), sp_func_vals, delimiter=',')
    options = ['LS', 'XY']
    index = 0
    for option in options:
        if option == 'LS':
            assert(index == 0)
        else:
            assert(index == 1)
        np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s.csv' %
                   (option, n, m, lambda_max), fp_norms[index],
                    delimiter=',')

        np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s.csv' %
                   (option, n, m, lambda_max), fp_func_vals[index],
                    delimiter=',')

        np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s.csv' %
                   (option, n, m, lambda_max), func_evals_step[index],
                    delimiter=',')

        np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s.csv' %
                   (option, n, m, lambda_max), func_evals_dir[index],
                    delimiter=',')

        np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s.csv' %
                    (option, n, m, lambda_max), time_taken[index],
                    delimiter=',')
        index += 1

    sp_func_vals_initial = np.genfromtxt('sp_func_vals_n=%s_m=%s_lambda_max=%s_initial.csv' %
                                         (n, m, lambda_max), delimiter=',')
    for j in range(len(noise_list)):
        assert(np.all(np.round(sp_func_vals_initial, 8) == np.round(sp_func_vals[j], 8)))