import numpy as np

import est_dir


def test_1():
    m = 100
    n = 16
    lambda_max = 1
    noise_list = np.array([2.37909586,  7.52336169])
    max_func_evals_list = [20000, 30000]
    num_funcs = 5
    domain = (0, 5)
    (sp_norms, sp_func_vals,
     fp_norms, fp_func_vals,
     sp_func_vals_noise,
     fp_func_vals_noise,
     time_taken, func_evals_step,
     func_evals_dir, no_its) = est_dir.num_exp_SNR(n, m, num_funcs, lambda_max,
                                                   domain, max_func_evals_list,
                                                   noise_list)
    #check results have been populated correctly. 
    assert(np.all(sp_norms > 0))
    assert(np.all(sp_func_vals > 0))
    assert(np.all(fp_norms > 0))
    assert(np.all(fp_func_vals > 0))
    assert(np.all(func_evals_step > 0))
    assert(np.all(func_evals_dir > 0))
    assert(np.all(time_taken > 0))
    assert(np.all(no_its > 0))

    assert(np.all(fp_norms[0] != fp_norms[1]))
    assert(np.all(fp_func_vals[0] != fp_func_vals[1]))
    assert(np.all(func_evals_step[0] != func_evals_step[1]))
    assert(np.all(func_evals_dir[0] != func_evals_dir[1]))
    assert(np.all(time_taken [0] != time_taken [1]))

    assert(np.all(sp_norms[0] == sp_norms[1]))
    assert(np.all(sp_func_vals[0] == sp_func_vals[1]))

    #check results are stored correctly for 'LS' and 'XY'. 
    f = est_dir.sphere_f_noise
    test_sp_norms = np.zeros((len(noise_list), num_funcs))
    test_fp_norms = np.zeros((2, len(noise_list), num_funcs))
    test_sp_func_vals_noise = np.zeros((2, len(noise_list), num_funcs))
    test_fp_func_vals_noise = np.zeros((2, len(noise_list), num_funcs))
    test_sp_func_vals = np.zeros((len(noise_list), num_funcs))
    test_fp_func_vals = np.zeros((2, len(noise_list), num_funcs))
    test_time_taken = np.zeros((2, len(noise_list), num_funcs))
    test_func_evals_step = np.zeros((2, len(noise_list), num_funcs)) 
    test_func_evals_dir = np.zeros((2, len(noise_list), num_funcs))
    test_no_its = np.zeros((2, len(noise_list), num_funcs))
    index_noise = 0
    for noise_sd in noise_list:
        max_func_evals_t = max_func_evals_list[index_noise]
        for j in range(num_funcs):
            seed = j * 50
            #Create function parameters and centre point
            np.random.seed(seed)
            minimizer = np.random.uniform(*domain, (m, ))
            centre_point = np.random.uniform(*domain, (m, ))
            matrix = est_dir.sphere_func_params(1, lambda_max, m)
            test_sp_norms[index_noise, j] = np.linalg.norm(minimizer - centre_point)
            test_sp_func_vals[index_noise, j] = est_dir.sphere_f(centre_point, minimizer, matrix)
            func_args = (minimizer, matrix, 0, noise_sd)

            np.random.seed(seed + 1)
            (upd_point_LS,
            test_sp_func_vals_noise[0, index_noise, j],
            test_fp_func_vals_noise[0, index_noise, j],
            test_time_taken[0, index_noise, j],
            test_func_evals_step[0, index_noise, j],
            test_func_evals_dir[0, index_noise, j],
            test_no_its[0, index_noise, j]) = est_dir.calc_its_until_sc(centre_point, f, func_args, n, m, 
                                                                        option='LS',
                                                                        max_func_evals=max_func_evals_t)
            test_fp_norms[0, index_noise, j] = np.linalg.norm(minimizer - upd_point_LS)
            test_fp_func_vals[0, index_noise, j] =  est_dir.sphere_f(upd_point_LS, minimizer, matrix)


            np.random.seed(seed + 1)
            (upd_point_XY,
            test_sp_func_vals_noise[1, index_noise, j],
            test_fp_func_vals_noise[1, index_noise, j],
            test_time_taken[1, index_noise, j],
            test_func_evals_step[1, index_noise, j],
            test_func_evals_dir[1, index_noise, j],
            test_no_its[1, index_noise, j]) = est_dir.calc_its_until_sc(centre_point, f, func_args, n, m, 
                                                                    option='XY',
                                                                    max_func_evals=max_func_evals_t)
            test_fp_norms[1, index_noise, j] = np.linalg.norm(minimizer - upd_point_XY)
            test_fp_func_vals[1, index_noise, j] =  est_dir.sphere_f(upd_point_XY, minimizer, matrix)
        
        index_noise += 1
    

    assert(np.all(test_sp_norms == sp_norms))
    assert(np.all(test_sp_func_vals == sp_func_vals))
    assert(np.all(test_sp_func_vals_noise == sp_func_vals_noise))
    assert(np.all(test_fp_func_vals_noise == fp_func_vals_noise))
    assert(np.all(test_func_evals_step == func_evals_step))
    assert(np.all(test_func_evals_dir == func_evals_dir))
    assert(np.all(test_fp_norms == fp_norms))
    assert(np.all(test_fp_func_vals == fp_func_vals))


def test_2():
    m = 100
    n = 16
    lambda_max = 1
    domain = (0, 5)
    num_funcs = 100
    snr_list = [0.01, 0.1, 0.2, 0.3]
    sp_func_vals = est_dir.calc_initial_func_values(n, m, num_funcs, lambda_max, domain)
    assert(sp_func_vals.shape == (num_funcs, ))
    assert(np.all(sp_func_vals > 0))
    noise_list = est_dir.compute_var_quad_form(n, m, lambda_max, snr_list, sp_func_vals)
    assert(np.all(np.round((noise_list ** 2) / np.var(sp_func_vals), 6) == np.round(snr_list, 6)))