import numpy as np
import time
import statsmodels.api as sm

import est_dir


def compute_frac_fact(n, m, centre_point, no_vars, f, func_args,
                      region, set_all_positions):
    """
    Compute response function values using a 2^{7-4}, 2^{10-6} or 2^{15-11} design
    matrix.
    """
    positions = np.sort(np.random.choice(set_all_positions, no_vars,
                         replace=False))
    assert(np.unique(positions).shape[0] == no_vars)
    if no_vars == 10:
        design = np.array([[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  +1,  +1],
                            [+1,  -1,  -1,  -1,  +1,  -1,  +1,  +1,  -1,  -1],
                            [-1,  +1,  -1,  -1,  +1,  +1,  -1,  +1,  -1,  -1],
                            [+1,  +1,  -1,  -1,  -1,  +1,  +1,  -1,  +1,  +1],
                            [-1,  -1,  +1,  -1,  +1,  +1,  +1,  -1,  -1,  +1],
                            [+1,  -1,  +1,  -1,  -1,  +1,  -1,  +1,  +1,  -1],
                            [-1,  +1,  +1,  -1,  -1,  -1,  +1,  +1,  +1,  -1],
                            [+1,  +1,  +1,  -1,  +1,  -1,  -1,  -1,  -1,  +1],
                            [-1,  -1,  -1,  +1,  -1,  +1,  +1,  +1,  -1,  +1],
                            [+1,  -1,  -1,  +1,  +1,  +1,  -1,  -1,  +1,  -1],
                            [-1,  +1,  -1,  +1,  +1,  -1,  +1,  -1,  +1,  -1],
                            [+1,  +1,  -1,  +1,  -1,  -1,  -1,  +1,  -1,  +1],
                            [-1,  -1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1],
                            [+1,  -1,  +1,  +1,  -1,  -1,  +1,  -1,  -1,  -1],
                            [-1,  +1,  +1,  +1,  -1,  +1,  -1,  -1,  -1,  -1],
                            [+1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1,  +1]])
        n_temp = 16
    else:
        raise ValueError('Incorrect no_vars choice')
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n_temp, m, f,
                                      func_args, region)
    return design, y, positions, func_evals


def compute_direction_LS(n, m, centre_point, f, func_args, no_vars, region, const):
    func_evals_count = 0
    set_all_positions = np.arange(m)
    total_checks = int(np.floor(m / no_vars))
    act_design, y, positions, func_evals = (compute_frac_fact
                                            (n, m, centre_point, no_vars,
                                             f, func_args, region,
                                             set_all_positions))
    func_evals_count += func_evals
    full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
    full_act_design[:, 1:] = act_design
    est = sm.OLS(y, full_act_design)
    results = est.fit()
    index_vars = 1
    while results.f_pvalue >= 0.1:
        if index_vars >= total_checks:
            break
        set_all_positions = np.setdiff1d(set_all_positions, positions)
        act_design, y, positions, func_evals = (compute_frac_fact
                                                (n, m, centre_point, no_vars,
                                                 f, func_args, region,
                                                 set_all_positions))
        func_evals_count += func_evals
        full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
        full_act_design[:, 1:] = act_design
        est = sm.OLS(y, full_act_design)
        results = est.fit()
        index_vars += 1
    if results.f_pvalue < 0.1:
        direction = np.zeros((m,))
        direction[positions] = est_dir.coeffs_dir((results.params)[1:])
        assert(max(abs(direction) == 1))
        return direction, func_evals_count            
    else:
        return False, func_evals_count


def calc_first_phase_RSM_LS(centre_point, init_func_val, f, func_args,
                            n, m, const_back, back_tol,
                            const_forward, forward_tol, step, no_vars,
                            region, const):
    direction, total_func_evals_dir = (compute_direction_LS
                                       (n, m, centre_point, f,
                                        func_args, no_vars,
                                        region, const))
    if direction is not False:
        (opt_t, upd_point,
        f_new, total_func_evals_step) = (est_dir.combine_tracking
                                         (centre_point, init_func_val,
                                          direction, step, const_back,
                                          back_tol, const_forward,
                                          forward_tol, f, func_args))
        return (opt_t, upd_point, f_new, total_func_evals_step,
                 total_func_evals_dir, True, (np.linalg.norm(direction) / no_vars) * m)
    else:
        return 0, centre_point, init_func_val, 0, total_func_evals_dir, False, 0


def calc_its_until_sc_LS(centre_point, f, func_args, n, m,
                         f_no_noise, func_args_no_noise, 
                         no_vars, region, const,
                         const_back=0.5, back_tol=0.000001,
                         const_forward=2, forward_tol=100000000,
                         tol_its=10000000):
    t0 = time.time()
    if (no_vars != 10):
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
     flag, norm_grad) = (calc_first_phase_RSM_LS
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

    while flag:
        centre_point = upd_point
        store_norm_grad.append(norm_grad)
        new_func_val = f_val
        step = 1
        (opt_t, upd_point, f_val,
         func_evals_step,
         func_evals_dir,
         flag, norm_grad) = (calc_first_phase_RSM_LS
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

        if (total_func_evals_step + total_func_evals_dir) >= tol_its:
            break

    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations, store_good_dir,
            store_good_dir_norm, store_good_dir_func,
            store_norm_grad)