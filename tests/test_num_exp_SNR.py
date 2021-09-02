import numpy as np

import est_dir


def test_1():
    """
    Test for num_exp_SNR_LS().
    """
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    function_type = 'quad'
    m = 100
    lambda_max = 1
    noise_list = np.array([1, 2])
    region = 0.1
    num_funcs = 5
    cov = np.identity(m)
    (sp_norms, sp_func_vals,
     fp_norms, fp_func_vals,
     sp_func_vals_noise,
     fp_func_vals_noise,
     time_taken, func_evals_step,
     func_evals_dir, no_its,
     good_dir_no_its_prop,
     good_dir_norm, good_dir_func,
     mean_norm_grad) = est_dir.num_exp_SNR_LS(f, f_no_noise, m, num_funcs,
                                              lambda_max, cov, noise_list,
                                              region, function_type)

    assert(np.all(sp_norms > 0))
    assert(np.all(sp_func_vals > 0))
    assert(np.all(fp_norms > 0))
    assert(np.all(fp_func_vals > 0))
    assert(np.all(func_evals_step >= 0))
    assert(np.all(func_evals_dir > 0))
    assert(np.all(time_taken > 0))
    assert(np.all(no_its > 0))
    assert(np.where(fp_norms[0] == fp_norms[1])[0].shape[0] != num_funcs)
    assert(np.where(fp_func_vals[0] == fp_func_vals[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_step[0] == func_evals_step[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_dir[0] == func_evals_dir[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_norm[0] == good_dir_norm[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_func[0] == good_dir_func[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(time_taken[0] == time_taken[1])[0].shape[0] !=
           num_funcs)
    assert(np.all(sp_norms[0] == sp_norms[1]))
    assert(np.all(sp_func_vals[0] == sp_func_vals[1]))

    test_sp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_time_taken = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_step = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_dir = np.zeros((noise_list.shape[0], num_funcs))
    test_no_its = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_no_its_prop = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_norm = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_func = np.zeros((noise_list.shape[0], num_funcs))
    test_mean_norm_grad = np.zeros((noise_list.shape[0], num_funcs))
    for index_noise in range(noise_list.shape[0]):
        for j in range(num_funcs):
            noise_sd = noise_list[index_noise]
            seed = j * 50
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.quad_func_params(1, lambda_max, m)
            test_sp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           centre_point)
            test_sp_func_vals[index_noise, j] = est_dir.quad_f(centre_point,
                                                               minimizer,
                                                               matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            np.random.seed(seed + 1)
            (upd_point_LS,
             test_sp_func_vals_noise[index_noise, j],
             test_fp_func_vals_noise[index_noise, j],
             test_time_taken[index_noise, j],
             test_func_evals_step[index_noise, j],
             test_func_evals_dir[index_noise, j],
             test_no_its[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_LS(
                                 centre_point, f, func_args, m,
                                 f_no_noise, func_args_no_noise,
                                 region))
            test_fp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           upd_point_LS)
            test_fp_func_vals[index_noise, j] = est_dir.quad_f(upd_point_LS,
                                                               minimizer,
                                                               matrix)
            test_good_dir_no_its_prop[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                test_mean_norm_grad[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                test_good_dir_norm[index_noise,
                                   j] = np.mean(store_good_dir_norm)
                test_good_dir_func[index_noise,
                                   j] = np.mean(store_good_dir_func)

    assert(np.all(test_sp_norms == sp_norms))
    assert(np.all(test_sp_func_vals == sp_func_vals))
    assert(np.all(test_sp_func_vals_noise == sp_func_vals_noise))
    assert(np.all(test_fp_func_vals_noise == fp_func_vals_noise))
    assert(np.all(test_func_evals_step == func_evals_step))
    assert(np.all(test_func_evals_dir == func_evals_dir))
    assert(np.all(test_fp_norms == fp_norms))
    assert(np.all(test_fp_func_vals == fp_func_vals))
    assert(np.all(test_good_dir_no_its_prop == good_dir_no_its_prop))
    assert(np.all(test_good_dir_norm == good_dir_norm))
    assert(np.all(test_good_dir_func == good_dir_func))
    assert(np.all(test_mean_norm_grad == mean_norm_grad))


def test_2():
    """
    Test for num_exp_SNR_XY().
    """
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    function_type = 'quad'
    m = 100
    n = 16
    lambda_max = 1
    noise_list = np.array([1, 2])
    max_func_evals_list = np.array([1000, 2000])
    no_vars = m
    region = 0.1
    num_funcs = 5
    cov = np.identity(m)
    (sp_norms, sp_func_vals,
     fp_norms, fp_func_vals,
     sp_func_vals_noise,
     fp_func_vals_noise,
     time_taken, func_evals_step,
     func_evals_dir, no_its,
     good_dir_no_its_prop,
     good_dir_norm, good_dir_func,
     mean_norm_grad) = est_dir.num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs,
                                              lambda_max, cov, noise_list,
                                              no_vars, region,
                                              max_func_evals_list,
                                              function_type)

    assert(np.all(sp_norms > 0))
    assert(np.all(sp_func_vals > 0))
    assert(np.all(fp_norms > 0))
    assert(np.all(fp_func_vals > 0))
    assert(np.all(func_evals_step > 0))
    assert(np.all(func_evals_dir > 0))
    assert(np.all(time_taken > 0))
    assert(np.all(no_its > 0))
    assert(np.where(fp_norms[0] == fp_norms[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(fp_func_vals[0] == fp_func_vals[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_step[0] == func_evals_step[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_dir[0] == func_evals_dir[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_norm[0] == good_dir_norm[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_func[0] == good_dir_func[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(time_taken[0] == time_taken[1])[0].shape[0] !=
           num_funcs)
    assert(np.all(sp_norms[0] == sp_norms[1]))
    assert(np.all(sp_func_vals[0] == sp_func_vals[1]))

    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    test_sp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_time_taken = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_step = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_dir = np.zeros((noise_list.shape[0], num_funcs))
    test_no_its = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_no_its_prop = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_norm = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_func = np.zeros((noise_list.shape[0], num_funcs))
    test_mean_norm_grad = np.zeros((noise_list.shape[0], num_funcs))
    for index_noise in range(noise_list.shape[0]):
        max_func_evals = max_func_evals_list[index_noise]
        for j in range(num_funcs):
            noise_sd = noise_list[index_noise]
            seed = j * 50
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.quad_func_params(1, lambda_max, m)
            test_sp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           centre_point)
            test_sp_func_vals[index_noise, j] = est_dir.quad_f(centre_point,
                                                               minimizer,
                                                               matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            np.random.seed(seed + 1)
            (upd_point_XY,
             test_sp_func_vals_noise[index_noise, j],
             test_fp_func_vals_noise[index_noise, j],
             test_time_taken[index_noise, j],
             test_func_evals_step[index_noise, j],
             test_func_evals_dir[index_noise, j],
             test_no_its[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_XY(
                                 centre_point, f, func_args, n, m,
                                 f_no_noise, func_args_no_noise,
                                 no_vars, region, max_func_evals))
            test_fp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           upd_point_XY)
            test_fp_func_vals[index_noise, j] = est_dir.quad_f(upd_point_XY,
                                                               minimizer,
                                                               matrix)
            test_good_dir_no_its_prop[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                test_mean_norm_grad[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                test_good_dir_norm[index_noise,
                                   j] = np.mean(store_good_dir_norm)
                test_good_dir_func[index_noise,
                                   j] = np.mean(store_good_dir_func)

    assert(np.all(test_sp_norms == sp_norms))
    assert(np.all(test_sp_func_vals == sp_func_vals))
    assert(np.all(test_sp_func_vals_noise == sp_func_vals_noise))
    assert(np.all(test_fp_func_vals_noise == fp_func_vals_noise))
    assert(np.all(test_func_evals_step == func_evals_step))
    assert(np.all(test_func_evals_dir == func_evals_dir))
    assert(np.all(test_fp_norms == fp_norms))
    assert(np.all(test_fp_func_vals == fp_func_vals))
    assert(np.all(test_good_dir_no_its_prop == good_dir_no_its_prop))
    assert(np.all(test_good_dir_norm == good_dir_norm))
    assert(np.all(test_good_dir_func == good_dir_func))
    assert(np.all(test_mean_norm_grad == mean_norm_grad))


def test_3():
    """
    Test for num_exp_SNR_MP().
    """
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    function_type = 'quad'
    m = 100
    n = 16
    lambda_max = 1
    noise_list = np.array([1, 2])
    max_func_evals_list = np.array([1000, 2000])
    no_vars = m
    region = 0.1
    num_funcs = 5
    cov = np.identity(m)
    type_inverse = 'right'
    (sp_norms, sp_func_vals,
     fp_norms, fp_func_vals,
     sp_func_vals_noise,
     fp_func_vals_noise,
     time_taken, func_evals_step,
     func_evals_dir, no_its,
     good_dir_no_its_prop,
     good_dir_norm, good_dir_func,
     mean_norm_grad) = est_dir.num_exp_SNR_MP(f, f_no_noise, n, m, num_funcs,
                                              lambda_max, cov, noise_list,
                                              no_vars, region,
                                              max_func_evals_list,
                                              type_inverse, function_type)

    assert(np.all(sp_norms > 0))
    assert(np.all(sp_func_vals > 0))
    assert(np.all(fp_norms > 0))
    assert(np.all(fp_func_vals > 0))
    assert(np.all(func_evals_step > 0))
    assert(np.all(func_evals_dir > 0))
    assert(np.all(time_taken > 0))
    assert(np.all(no_its > 0))
    assert(np.where(fp_norms[0] == fp_norms[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(fp_func_vals[0] == fp_func_vals[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_step[0] == func_evals_step[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(func_evals_dir[0] == func_evals_dir[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_norm[0] == good_dir_norm[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(good_dir_func[0] == good_dir_func[1])[0].shape[0] !=
           num_funcs)
    assert(np.where(time_taken[0] == time_taken[1])[0].shape[0] !=
           num_funcs)
    assert(np.all(sp_norms[0] == sp_norms[1]))
    assert(np.all(sp_func_vals[0] == sp_func_vals[1]))

    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    test_sp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_norms = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals_noise = np.zeros((noise_list.shape[0], num_funcs))
    test_sp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_fp_func_vals = np.zeros((noise_list.shape[0], num_funcs))
    test_time_taken = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_step = np.zeros((noise_list.shape[0], num_funcs))
    test_func_evals_dir = np.zeros((noise_list.shape[0], num_funcs))
    test_no_its = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_no_its_prop = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_norm = np.zeros((noise_list.shape[0], num_funcs))
    test_good_dir_func = np.zeros((noise_list.shape[0], num_funcs))
    test_mean_norm_grad = np.zeros((noise_list.shape[0], num_funcs))
    for index_noise in range(noise_list.shape[0]):
        max_func_evals = max_func_evals_list[index_noise]
        for j in range(num_funcs):
            noise_sd = noise_list[index_noise]
            seed = j * 50
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.quad_func_params(1, lambda_max, m)
            test_sp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           centre_point)
            test_sp_func_vals[index_noise, j] = est_dir.quad_f(centre_point,
                                                               minimizer,
                                                               matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)
            np.random.seed(seed + 1)
            (upd_point_MP,
             test_sp_func_vals_noise[index_noise, j],
             test_fp_func_vals_noise[index_noise, j],
             test_time_taken[index_noise, j],
             test_func_evals_step[index_noise, j],
             test_func_evals_dir[index_noise, j],
             test_no_its[index_noise, j],
             store_good_dir,
             store_good_dir_norm,
             store_good_dir_func,
             store_norm_grad) = (est_dir.calc_its_until_sc_MP(
                                 centre_point, f, func_args, n, m,
                                 f_no_noise, func_args_no_noise,
                                 no_vars, region, max_func_evals,
                                 type_inverse))
            test_fp_norms[index_noise, j] = np.linalg.norm(minimizer -
                                                           upd_point_MP)
            test_fp_func_vals[index_noise, j] = est_dir.quad_f(upd_point_MP,
                                                               minimizer,
                                                               matrix)
            test_good_dir_no_its_prop[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                test_mean_norm_grad[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                test_good_dir_norm[index_noise,
                                   j] = np.mean(store_good_dir_norm)
                test_good_dir_func[index_noise,
                                   j] = np.mean(store_good_dir_func)

    assert(np.all(test_sp_norms == sp_norms))
    assert(np.all(test_sp_func_vals == sp_func_vals))
    assert(np.all(test_sp_func_vals_noise == sp_func_vals_noise))
    assert(np.all(test_fp_func_vals_noise == fp_func_vals_noise))
    assert(np.all(test_func_evals_step == func_evals_step))
    assert(np.all(test_func_evals_dir == func_evals_dir))
    assert(np.all(test_fp_norms == fp_norms))
    assert(np.all(test_fp_func_vals == fp_func_vals))
    assert(np.all(test_good_dir_no_its_prop == good_dir_no_its_prop))
    assert(np.all(test_good_dir_norm == good_dir_norm))
    assert(np.all(test_good_dir_func == good_dir_func))
    assert(np.all(test_mean_norm_grad == mean_norm_grad))


def test_4():
    """
    Test for calc_initial_func_values() and compute_var_quad_form()
    with region = 1.
    """
    f_no_noise = est_dir.quad_f
    m = 100
    lambda_max = 1
    cov = np.identity(m)
    num_funcs = 100
    snr_list = [0.5, 0.75, 1, 2]
    region = 1
    sp_func_vals = (est_dir.calc_initial_func_values(
                    m, num_funcs, lambda_max, cov, f_no_noise))
    assert(sp_func_vals.shape == (num_funcs, ))
    assert(np.all(sp_func_vals > 0))
    noise_list = est_dir.compute_var_quad_form(m, snr_list, sp_func_vals,
                                               region)
    for k in range(noise_list.shape[0]):
        assert(np.all(np.round(np.var(sp_func_vals * region) /
                               (noise_list**2), 6)[k] ==
               np.round(snr_list[k], 6)))


def test_5():
    """
    Test for calc_initial_func_values() and compute_var_quad_form()
    with region = 0.1.
    """
    f_no_noise = est_dir.quad_f
    m = 100
    n = 16
    lambda_max = 1
    cov = np.identity(m)
    num_funcs = 100
    snr_list = [0.5, 0.75, 1, 2]
    region = 0.1
    sp_func_vals = (est_dir.calc_initial_func_values(
                    m, num_funcs, lambda_max, cov, f_no_noise))
    assert(sp_func_vals.shape == (num_funcs, ))
    assert(np.all(sp_func_vals > 0))
    noise_list = est_dir.compute_var_quad_form(m, snr_list, sp_func_vals,
                                               region)
    for k in range(noise_list.shape[0]):
        assert(np.all(np.round(np.var(sp_func_vals * region) /
                               (noise_list**2), 6)[k] ==
               np.round(snr_list[k], 6)))


def test_6():
    """
    Test for quad_LS_XY_MP() with store_max_func_evals=False.
    """
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    function_type = 'quad'
    m = 100
    n = 16
    lambda_max = 1
    num_funcs = 5
    cov = np.identity(m)
    no_vars = m
    snr_list = [0.5, 0.75, 1, 2]
    region = 0.1
    type_inverse = 'left'
    store_max_func_evals = None
    sp_func_vals = (est_dir.calc_initial_func_values(
                    m, num_funcs, lambda_max, cov, f_no_noise))
    noise_list = est_dir.compute_var_quad_form(m, snr_list,
                                               sp_func_vals, region)
    est_dir.quad_LS_XY_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                          noise_list, no_vars, region, function_type,
                          type_inverse, store_max_func_evals)


def test_7():
    """
    Test for quad_LS_XY_MP() with
    store_max_func_evals = [1000, 1000, 2000, 2000].
    """
    f = est_dir.quad_f_noise
    f_no_noise = est_dir.quad_f
    function_type = 'quad'
    m = 100
    n = 16
    lambda_max = 1
    num_funcs = 5
    cov = np.identity(m)
    no_vars = m
    snr_list = [0.5, 0.75, 1, 2]
    region = 0.1
    type_inverse = 'left'
    store_max_func_evals = [1000, 1000, 2000, 2000]
    sp_func_vals = (est_dir.calc_initial_func_values(
                    m, num_funcs, lambda_max, cov, f_no_noise))
    noise_list = est_dir.compute_var_quad_form(m, snr_list,
                                               sp_func_vals, region)
    est_dir.quad_LS_XY_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                          noise_list, no_vars, region, function_type,
                          type_inverse, store_max_func_evals)


def test_8():
    """
    Test for generate_func_params()
    """
    m = 100
    cov = np.identity(m)
    lambda_max = 10
    for j in range(100):
        (centre_point,
         minimizer,
         matrix) = (est_dir.generate_func_params(
                    j, cov, m, lambda_max))
        seed = j * 50
        np.random.seed(seed)
        centre_point_2 = np.random.multivariate_normal(np.zeros((m)), cov)
        minimizer_2 = np.zeros((m, ))
        matrix_2 = est_dir.quad_func_params(1, lambda_max, m)

        assert(np.all(centre_point == centre_point_2))
        assert(np.all(minimizer == minimizer_2))
        assert(np.all(matrix == matrix_2))
