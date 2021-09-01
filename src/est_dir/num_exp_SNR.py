import sys

import numpy as np
import est_dir
import tqdm


# def calc_initial_func_values(n, m, num_funcs, lambda_max, cov):
#     sp_func_vals = np.zeros((num_funcs))
#     for j in tqdm.tqdm(range(num_funcs)):
#         seed = j * 50
#         #Create function parameters and centre point
#         np.random.seed(seed)
#         centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
#         minimizer = np.zeros((m, ))
#         matrix = est_dir.sphere_func_params(1, lambda_max, m)
#         sp_func_vals[j] = est_dir.sphere_f(centre_point, minimizer, matrix)
#     return sp_func_vals


# def compute_var_quad_form(n, m, lambda_max, snr_list, sp_func_vals):
#     var_quad_form = np.var(sp_func_vals)
#     noise_list = np.zeros((len(snr_list)))
#     index = 0
#     for snr in snr_list:
#         noise_list[index] = np.sqrt(var_quad_form / snr)
#         index += 1
#     return noise_list

def calc_initial_func_values(n, m, num_funcs, lambda_max, cov, f_no_noise):
    sp_func_vals = np.zeros((num_funcs))
    for j in tqdm.tqdm(range(num_funcs)):
        seed = j * 50
        #Create function parameters and centre point
        np.random.seed(seed)
        centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
        minimizer = np.zeros((m, ))
        matrix = est_dir.sphere_func_params(1, lambda_max, m)
        sp_func_vals[j] = f_no_noise(centre_point, minimizer, matrix)
    return sp_func_vals


def compute_var_quad_form(n, m, lambda_max, snr_list, sp_func_vals, region):
    noise_list = np.zeros((len(snr_list)))
    index = 0
    for snr in snr_list:
        noise_list[index] = np.sqrt(np.var(sp_func_vals * region) / snr)
        index += 1
    return noise_list


# def compute_var_quad_form(n, m, lambda_max, snr_list, sp_var, region):
#     noise_list = np.zeros((len(snr_list), sp_var.shape[0]))
#     index = 0
#     for snr in snr_list:
#         for j in range(sp_var.shape[0]):
#             noise_list[index, j] = np.sqrt(sp_var[j] / snr) * region
#         index += 1
#     return noise_list


def num_exp_SNR_LS(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                   noise_list, no_vars, region, const, function_type):
    save_choice = 'f'
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
            seed = j * 50
            #Create function parameters and centre point
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.sphere_func_params(1, lambda_max, m)
            sp_norms_LS[index_noise, j] = np.linalg.norm(minimizer - centre_point)
            sp_func_vals_LS[index_noise, j] = f_no_noise(centre_point, minimizer, matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)


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
            store_norm_grad) = est_dir.calc_its_until_sc_LS(centre_point, f, func_args, n, m,
                                                            f_no_noise, func_args_no_noise, 
                                                            no_vars, region, const)
            fp_norms_LS[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_LS[index_noise, j] =  f_no_noise(upd_point, minimizer, matrix)
            good_dir_no_its_prop_LS[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_LS[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_LS[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_LS[index_noise, j] = np.mean(store_good_dir_func)

    #Save relevant data results

    option_t = 'LS'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_norms_LS, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_func_vals_LS, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_norms_LS,
                delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_LS,
                delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_step_LS,
                delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_dir_LS,
                delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), time_taken_LS,
                delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_noise_LS,
                delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_no_its_prop_LS,
                delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), no_its_LS,
                delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_norm_LS,
                delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_func_LS,
                delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), mean_norm_grad_LS,
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


def num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                   noise_list, no_vars,
                   region, max_func_evals_list, const, function_type):
    save_choice = 'r'
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
            seed = j * 50
            #Create function parameters and centre point
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.sphere_func_params(1, lambda_max, m)
            sp_norms_XY[index_noise, j] = np.linalg.norm(minimizer - centre_point)
            sp_func_vals_XY[index_noise, j] = f_no_noise(centre_point, minimizer, matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)


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
            store_norm_grad) = est_dir.calc_its_until_sc_XY(centre_point, f, func_args, n, m,
                                                                f_no_noise, func_args_no_noise, 
                                                                no_vars, region, max_func_evals,
                                                                const)
            fp_norms_XY[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_XY[index_noise, j] =  f_no_noise(upd_point, minimizer, matrix)
            good_dir_no_its_prop_XY[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_XY[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_XY[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_XY[index_noise, j] = np.mean(store_good_dir_func)


    #Save relevant data results

    option_t = 'XY'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_norms_XY, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_func_vals_XY, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_norms_XY,
                delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_XY,
                delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_step_XY,
                delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_dir_XY,
                delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), time_taken_XY,
                delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_noise_XY,
                delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_no_its_prop_XY,
                delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), no_its_XY,
                delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_norm_XY,
                delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_func_XY,
                delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), mean_norm_grad_XY,
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
                   noise_list, no_vars,
                   region, max_func_evals_list, const, function_type):
    save_choice = 'r'
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
            seed = j * 50
            #Create function parameters and centre point
            np.random.seed(seed)
            centre_point = np.random.multivariate_normal(np.zeros((m)), cov)
            minimizer = np.zeros((m, ))
            matrix = est_dir.sphere_func_params(1, lambda_max, m)
            sp_norms_MP[index_noise, j] = np.linalg.norm(minimizer - centre_point)
            sp_func_vals_MP[index_noise, j] = f_no_noise(centre_point, minimizer, matrix)
            func_args = (minimizer, matrix, 0, noise_sd)
            func_args_no_noise = (minimizer, matrix)


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
            store_norm_grad) = est_dir.calc_its_until_sc_MP(centre_point, f, func_args, n, m,
                                                                f_no_noise, func_args_no_noise, 
                                                                no_vars, region, max_func_evals,
                                                                const)
            fp_norms_MP[index_noise, j] = np.linalg.norm(minimizer - upd_point)
            fp_func_vals_MP[index_noise, j] =  f_no_noise(upd_point, minimizer, matrix)
            good_dir_no_its_prop_MP[index_noise, j] = store_good_dir

            if len(store_norm_grad) > 0:
                mean_norm_grad_MP[index_noise, j] = np.mean(store_norm_grad)

            if len(store_good_dir_norm) > 0:
                good_dir_norm_MP[index_noise, j] = np.mean(store_good_dir_norm)
                good_dir_func_MP[index_noise, j] = np.mean(store_good_dir_func)


    #Save relevant data results

    option_t = 'MP'
    np.savetxt('sp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_norms_MP, delimiter=',')
    np.savetxt('sp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
               (option_t, n, m, lambda_max, const, function_type), sp_func_vals_MP, delimiter=',')

    np.savetxt('fp_norms_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_norms_MP,
                delimiter=',')

    np.savetxt('fp_func_vals_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_MP,
                delimiter=',')

    np.savetxt('func_evals_step_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_step_MP,
                delimiter=',')

    np.savetxt('func_evals_dir_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), func_evals_dir_MP,
                delimiter=',')

    np.savetxt('time_taken_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), time_taken_MP,
                delimiter=',')

    np.savetxt('fp_func_vals_noise_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), fp_func_vals_noise_MP,
                delimiter=',')

    np.savetxt('good_dir_prop_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_no_its_prop_MP,
                delimiter=',')

    np.savetxt('no_its_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), no_its_MP,
                delimiter=',')

    np.savetxt('good_dir_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_norm_MP,
                delimiter=',')

    np.savetxt('good_dir_func_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), good_dir_func_MP,
                delimiter=',')
    np.savetxt('mean_grad_norm_%s_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                (option_t, n, m, lambda_max, no_vars,
                save_choice, region, const, function_type), mean_norm_grad_MP,
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


def quad_LS_XY_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                noise_list, no_vars_list, 
                region, const, function_type):

    # results_LS = num_exp_SNR_LS(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
    #                             noise_list, no_vars_list[0],
    #                             region, const, function_type)
    # total_func_evals = results_LS[7] + results_LS[8]
    
    # max_func_evals_list = np.zeros(len(noise_list))
    # for j in range(len(noise_list)):
    #     max_func_evals_list[j] = np.mean(total_func_evals[j]) 

    max_func_evals_list = np.array([ 3100,3100,3100,3100])
    # results_MP = num_exp_SNR_MP(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
    #                             noise_list, no_vars_list[1],
    #                             region, max_func_evals_list, const, function_type) 

    results_XY = num_exp_SNR_XY(f, f_no_noise, n, m, num_funcs, lambda_max, cov,
                                noise_list, no_vars_list[1],
                                region, max_func_evals_list, const, function_type)

    # assert(np.all(np.round(results_XY[0], 5) == np.round(results_LS[0], 5)))
    # assert(np.all(np.round(results_XY[0], 5) == np.round(results_MP[0], 5)))
    # assert(np.all(np.round(results_XY[1], 5) == np.round(results_LS[1], 5)))
    # assert(np.all(np.round(results_XY[1], 5) == np.round(results_MP[1], 5)))
    for j in range(len(noise_list)-1):
        for i in range(j, len(noise_list)):
            assert(np.all(np.round(results_XY[1][j], 5) == np.round(results_XY[1][i], 5)))
            assert(np.all(np.round(results_XY[0][j], 5) == np.round(results_XY[0][i], 5)))
