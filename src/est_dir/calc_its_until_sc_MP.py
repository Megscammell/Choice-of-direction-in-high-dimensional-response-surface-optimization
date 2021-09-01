import numpy as np
import time
from warnings import warn

import est_dir


def compute_direction_MP(n, m, centre_point, f, func_args, no_vars, region, const):
    act_design, y, positions, func_evals = (est_dir.compute_rand_ones
                                            (n, m, centre_point, no_vars,
                                             f, func_args, region))
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    direction = np.zeros((m,))
    est = np.linalg.pinv(full_act_design.T @ full_act_design) @ full_act_design.T @ y
    direction[positions] = est_dir.coeffs_dir(est[1:])
    assert(max(abs(direction) == 1))
    return direction, func_evals


def calc_first_phase_RSM_MP(centre_point, init_func_val, f, func_args,
                            n, m, const_back, back_tol, const_forward,
                            forward_tol, step, no_vars, region, const):
    direction, total_func_evals_dir = (compute_direction_MP
                                       (n, m, centre_point, f,
                                        func_args, no_vars,
                                        region, const))
    (opt_t, upd_point,
     f_new, total_func_evals_step) = (est_dir.combine_tracking
                                      (centre_point, init_func_val,
                                       direction, step, const_back,
                                       back_tol, const_forward,
                                       forward_tol, f, func_args))
    return (opt_t, upd_point, f_new, total_func_evals_step,
            total_func_evals_dir, (np.linalg.norm(direction) / no_vars) * m)


def calc_its_until_sc_MP(centre_point, f, func_args, n, m,
                         f_no_noise, func_args_no_noise, 
                         no_vars, region, max_func_evals,
                         const,
                         const_back=0.5, back_tol=0.000001,
                         const_forward=2, forward_tol=100000000):
    t0 = time.time()
    if (no_vars > m):
        raise ValueError('Incorrect no_vars choice')
    store_good_dir = 0
    store_good_dir_norm = []
    store_good_dir_func = []
    store_norm_grad = []
    total_func_evals_step = 0
    total_func_evals_dir = 0
    step = 1
    init_func_val = f(centre_point, *func_args)
    (opt_t, upd_point, f_val,
     func_evals_step,
     func_evals_dir,
     norm_grad) = (calc_first_phase_RSM_MP
                   (centre_point, np.copy(init_func_val), f,
                    func_args, n, m, const_back,
                    back_tol, const_forward, forward_tol, step,
                    no_vars, region, const))
    total_func_evals_step += func_evals_step
    total_func_evals_dir += func_evals_dir
    no_iterations = 1

    if f_no_noise(centre_point, *func_args_no_noise) > f_no_noise(upd_point, *func_args_no_noise):
        store_good_dir += 1
        store_good_dir_norm.append(np.linalg.norm(centre_point - func_args[0]) - np.linalg.norm(upd_point - func_args[0]))
        store_good_dir_func.append(f_no_noise(centre_point, *func_args_no_noise) - f_no_noise(upd_point, *func_args_no_noise))


    while (total_func_evals_step + total_func_evals_dir + n) < max_func_evals:
        centre_point = upd_point
        store_norm_grad.append(norm_grad)
        new_func_val = f_val
        step = 1
        (opt_t, upd_point, f_val,
         func_evals_step,
         func_evals_dir,
         norm_grad) = (calc_first_phase_RSM_MP
                       (centre_point, np.copy(new_func_val), f, func_args,
                        n, m, const_back, back_tol,
                        const_forward, forward_tol,
                        step, no_vars, region, const))
        total_func_evals_step += func_evals_step
        total_func_evals_dir += func_evals_dir
        no_iterations += 1

        if f_no_noise(centre_point, *func_args_no_noise) > f_no_noise(upd_point, *func_args_no_noise):
                store_good_dir += 1
                store_good_dir_norm.append(np.linalg.norm(centre_point - func_args[0]) - np.linalg.norm(upd_point - func_args[0]))
                store_good_dir_func.append(f_no_noise(centre_point, *func_args_no_noise) - f_no_noise(upd_point, *func_args_no_noise))


    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations, store_good_dir,
            store_good_dir_norm, store_good_dir_func,
            store_norm_grad)