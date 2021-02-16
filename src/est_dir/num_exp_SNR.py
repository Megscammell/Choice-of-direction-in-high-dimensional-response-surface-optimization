import sys

import numpy as np
import est_dir
import tqdm


def calc_initial_func_values(n, m, num_funcs, lambda_max, domain):
    sp_func_vals = np.zeros((num_funcs))
    for j in tqdm.tqdm(range(num_funcs)):
        seed = j * 50
        #Create function parameters and centre point
        np.random.seed(seed)
        minimizer = np.random.uniform(*domain, (m, ))
        centre_point = np.random.uniform(*domain, (m, ))
        matrix = est_dir.sphere_func_params(1, lambda_max, m)
        sp_func_vals[j] = est_dir.sphere_f(centre_point, minimizer, matrix)
    return sp_func_vals


def compute_var_quad_form(n, m, lambda_max, snr_list, sp_func_vals):
    var_quad_form = np.var(sp_func_vals)
    noise_list = np.zeros((len(snr_list)))
    index = 0
    for snr in snr_list:
        noise_list[index] = np.sqrt(snr * var_quad_form)
        index += 1
    return noise_list


def num_exp_SNR(n, m, num_funcs, lambda_max, domain, max_func_evals_list,
                noise_list):
    """Run numerical experiment with option = 'LS' and option = 'XY'."""
    f = est_dir.sphere_f_noise
    options = ['LS', 'XY']
    sp_norms = np.zeros((len(noise_list), num_funcs))
    fp_norms = np.zeros((2, len(noise_list), num_funcs))
    sp_func_vals_noise = np.zeros((2, len(noise_list), num_funcs))
    fp_func_vals_noise = np.zeros((2, len(noise_list), num_funcs))
    sp_func_vals = np.zeros((len(noise_list), num_funcs))
    fp_func_vals = np.zeros((2, len(noise_list), num_funcs))
    time_taken = np.zeros((2, len(noise_list), num_funcs))
    func_evals_step = np.zeros((2, len(noise_list), num_funcs)) 
    func_evals_dir = np.zeros((2, len(noise_list), num_funcs))
    no_its = np.zeros((2, len(noise_list), num_funcs))
    index_noise = 0
    for noise_sd in noise_list:
        max_func_evals_t = max_func_evals_list[index_noise]
        for j in tqdm.tqdm(range(num_funcs)):
            seed = j * 50
            #Create function parameters and centre point
            np.random.seed(seed)
            minimizer = np.random.uniform(*domain, (m, ))
            centre_point = np.random.uniform(*domain, (m, ))
            matrix = est_dir.sphere_func_params(1, lambda_max, m)
            sp_norms[index_noise, j] = np.linalg.norm(minimizer - centre_point)
            sp_func_vals[index_noise, j] = est_dir.sphere_f(centre_point, minimizer, matrix)
            func_args = (minimizer, matrix, 0, noise_sd)

            index_arr = 0
            for option_t in options:
                if option_t == 'LS':
                    assert(index_arr == 0)
                else:
                    assert(index_arr == 1)
                np.random.seed(seed + 1)
                (upd_point,
                sp_func_vals_noise[index_arr, index_noise, j],
                fp_func_vals_noise[index_arr, index_noise, j],
                time_taken[index_arr, index_noise, j],
                func_evals_step[index_arr, index_noise, j],
                func_evals_dir[index_arr, index_noise, j],
                no_its[index_arr, index_noise, j]) = est_dir.calc_its_until_sc(centre_point, f, func_args, n, m, 
                                                                option=option_t,
                                                                max_func_evals=max_func_evals_t)
                fp_norms[index_arr, index_noise, j] = np.linalg.norm(minimizer - upd_point)
                fp_func_vals[index_arr, index_noise, j] =  est_dir.sphere_f(upd_point, minimizer, matrix)
                index_arr  += 1

        index_noise += 1
  
    return (sp_norms, sp_func_vals, fp_norms, fp_func_vals, sp_func_vals_noise,
            fp_func_vals_noise, time_taken, func_evals_step, func_evals_dir,
            no_its)


