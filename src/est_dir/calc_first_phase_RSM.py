import numpy as np

import est_dir


def compute_direction(n, m, centre_point, f, func_args, option):
    if option == 'LS':
        choice = 'fractional_fact'
        act_design, y, positions, func_evals = (est_dir.create_design_matrix
                                                (n, m, centre_point,
                                                 choice, f, func_args))
        full_act_design = np.ones((act_design.shape[0], act_design.shape[1] + 1))
        full_act_design[:, 1:] = act_design
        OLS = (np.linalg.inv(full_act_design.T @ full_act_design) @
               full_act_design.T @ y)
        direction = np.zeros((m,))
        direction[positions] = OLS[1:]
        return direction, func_evals
    elif option == 'XY':
        choice = 'random_cols'
        act_design, y, positions, func_evals = (est_dir.create_design_matrix
                                                (n, m, centre_point,
                                                 choice, f, func_args))
        direction = np.zeros((m,))
        direction[positions] = act_design.T @ y
        return direction, func_evals
    else:
        raise ValueError('option is not correct.')


def calc_first_phase_RSM(store_centre_points, init_func_val, f, func_args,
                         option, n, m, const_back, back_tol,
                         const_forward, forward_tol, step):
    if (type(store_centre_points) is not list):
        raise ValueError('store_centre_points is not a list.')
    centre_point = store_centre_points[-1]
    direction, total_func_evals_dir = (compute_direction
                                       (n, m, centre_point, f,
                                        func_args, option))

    (opt_t, upd_point,
     f_new, total_func_evals_step) = (est_dir.combine_tracking
                                        (centre_point, init_func_val,
                                        direction, step, const_back,
                                        back_tol, const_forward,
                                        forward_tol, f, func_args))
    return (opt_t, upd_point, f_new, total_func_evals_step,
            total_func_evals_dir, True)
