import numpy as np
import time
from warnings import warn

import est_dir


def calc_its_until_sc(centre_point, f, func_args, n, m,
                      option, const_back=0.5, back_tol=0.000001,
                      const_forward=2, forward_tol=100000000,
                      max_func_evals=1000):
    t0 = time.time()
    store_centre_points = []
    total_func_evals_step = 0
    total_func_evals_dir = 0
    no_iterations = 0
    step = 1
    store_centre_points.append(centre_point)
    init_func_val = f(centre_point, *func_args)
    (opt_t, upd_point, f_val,
     func_evals_step,
     func_evals_dir,
     flag) = (est_dir.calc_first_phase_RSM
              (store_centre_points, np.copy(init_func_val), f,
               func_args, option, n, m, const_back,
               back_tol, const_forward, forward_tol, step))
    total_func_evals_step += func_evals_step
    total_func_evals_dir += func_evals_dir
    no_iterations += 1
    while flag == True:
        if (total_func_evals_step + total_func_evals_dir) > max_func_evals:
            break
        centre_point = upd_point
        store_centre_points.append(centre_point)
        func_val = f_val
        if opt_t > 0:
            step = opt_t
        else:
            step = 1
        (opt_t, upd_point, f_val,
         func_evals_step,
         func_evals_dir,
         flag) = (est_dir.calc_first_phase_RSM
                  (store_centre_points, func_val, f, func_args,
                   option, n, m, const_back, back_tol,
                   const_forward, forward_tol,
                   step))
        total_func_evals_step += func_evals_step
        total_func_evals_dir += func_evals_dir
        no_iterations += 1
    t1 = time.time()
    full_time = t1-t0
    return (upd_point, init_func_val, f_val, full_time,
            total_func_evals_step, total_func_evals_dir,
            no_iterations)