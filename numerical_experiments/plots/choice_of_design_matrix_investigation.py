import sys
import matplotlib.pyplot as plt
import numpy as np

import est_dir


def compute_random_design_old(n, m, centre_point, no_vars, f, func_args,
                              region):
    """
    Compute random design matrix centred at centre_point, where entries are
    chosen randomly as +1 or -1, with no condition on number of +1's or -1's
    in each column. Also, compute the response function value at each
    observation of the design matrix.

    Parameters
    ----------
    n : integer
        Number of observations of the design matrix (rows).
    m : integer
        Number of variables of the design matrix (columns).
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
    region : float
             Region of exploration around the centre point.

    Returns
    -------
    design : 2-D array
             Random design matrix.
    y : 1-D array
        Contains the response function values at each observation of the
        design matrix centred at centre_point.
    positions : 1-D array
                Positions of centre_point in which the design matrix has been
                centred. If the design matrix is of size (n, m), then
                positions = (1,2,...,m). Otherwise, if the design matrix is
                of size (n, no_vars), where no_vars < m, then
                positions will be of size (num_vars,) with entries chosen
                randomly from (1,2,...,m).
    func_evals : integer
                 Number of times the repsonse function has been evaluated.

    """
    if (n % 2) != 0:
        raise ValueError('n must be even.')
    if no_vars == m:
        positions = np.arange(m)
    else:
        positions = np.sort(np.random.choice(np.arange(m), no_vars,
                            replace=False))
    assert(np.unique(positions).shape[0] == no_vars)
    design = np.random.choice([-1, 1], size=(n, m))
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n, m, f,
                                      func_args, region)
    return design, y, positions, func_evals


def compute_var_diff_search_direction(noise_list, cov, n, m, lambda_max,
                                      no_vars, f, region):
    """
    Compares the standard deviation of elements of two search directions.
    The first search direction uses a design matrix centred at a point,
    where entries are chosen randomly as +1 or -1, with the condition that
    each column of the design matrix has the same number of +1's or -1's.
    The second search direction uses a design matrix centred at a point, where
    entries are chosen randomly as +1 or -1, with no condition on the
    number of +1's or -1's in each column.

    Parameters
    ----------
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
    cov : 2-D array
          Covariance matrix used to sample centre_point from normal
          distribution.
    n : integer
        Number of observations of the design matrix (rows).
    m : integer
        Number of variables of the design matrix (columns).
    lambda_max : integer
                 Largest eigenvalue of the positive definite matrix
                 used to compute the quadratic response function.
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    region : float
             Region of exploration around the centre point.

    Returns
    -------
    var_dir_elements : 3-D array
                          Returns the variance of two search directions
                          (PI-MY) for 100 functions with different standard
                          deviation of errors. The first search direction
                          uses a design matrix centred at a point, where
                          entries are chosen randomly as +1 or -1, with the
                          condition that each column of the design matrix has
                          the same number of +1's or -1's. The second search
                          direction uses a design matrix centred at a point,
                          where entries are chosen randomly as +1 or -1, with
                          no condition on the number of +1's or -1's in each
                          column.
    var_dir_elements_MP : 3-D array
                          Returns the variance of two search directions
                          (PI-MPI) for 100 functions with different standard
                          deviation of errors. The first search direction
                          uses a design matrix centred at a point, where
                          entries are chosen randomly as +1 or -1, with the
                          condition that each column of the design matrix has
                          the same number of +1's or -1's. The second search
                          direction uses a design matrix centred at a point,
                          where entries are chosen randomly as +1 or -1, with
                          no condition on the number of +1's or -1's in each
                          column.
    range_dir_elements: 3-D array
                        For directions computed by MY, 
                        returns the absolute maximum coefficient of the search
                        direction subtracted by the absolute minimum coefficient
                        of the search direction.
    range_dir_elements_MP: 3-D array                 
                           For directions computed by MP, 
                           returns the absolute maximum coefficient of the
                           search direction subtracted by the absolute minimum
                           coefficient of the search direction.
    """
    var_dir_elements = np.zeros((2, len(noise_list), 100))
    range_dir_elements = np.zeros((2, len(noise_list), 100))
    var_dir_elements_MP = np.zeros((2, len(noise_list), 100))
    range_dir_elements_MP = np.zeros((2, len(noise_list), 100))
    index_noise = 0
    for noise_sd in noise_list:
        for j in range(100):
            (centre_point,
             minimizer,
             matrix) = est_dir.generate_func_params(j, cov, m, lambda_max)
            func_args = (minimizer, matrix, 0, noise_sd)

            np.random.seed(j)
            (design_temp,
             y_temp,
             positions_temp,
             func_evals_temp) = (compute_random_design_old(
                                 n, m, centre_point,
                                 no_vars, f, func_args,
                                 region))
            direction_MY_rand = design_temp.T @ y_temp
            var_dir_elements[0, index_noise, j] = np.var(direction_MY_rand)
            range_dir_elements[0, 
                               index_noise,
                               j] = (np.max(abs(direction_MY_rand)) -
                                     np.min(abs(direction_MY_rand)))

            full_act_design_rand = np.ones((design_temp.shape[0],
                                            design_temp.shape[1] + 1))
            full_act_design_rand[:, 1:] = design_temp
            direction_MP_rand = (np.linalg.pinv(full_act_design_rand)
                                 @ y_temp)
            var_dir_elements_MP[0,
                                index_noise,
                                j] = (np.var(direction_MP_rand[1:]))
            range_dir_elements_MP[0,
                                  index_noise,
                                  j] = (np.max(abs(direction_MP_rand[1:])) -
                                       np.min(abs(direction_MP_rand[1:])))
            np.random.seed(j)
            (act_design,
             y, positions,
             func_evals) = (est_dir.compute_random_design
                            (n, m, centre_point, no_vars,
                             f, func_args, region))
            direction_MY = act_design.T @ y
            var_dir_elements[1, index_noise, j] = np.var(direction_MY)
            range_dir_elements[1, index_noise, j] = (np.max(abs(direction_MY)) -
                                                     np.min(abs(direction_MY)))

            full_act_design = np.ones((act_design.shape[0],
                                    act_design.shape[1] + 1))
            full_act_design[:, 1:] = act_design
            direction_MP = (np.linalg.pinv(full_act_design)
                            @ y)
            var_dir_elements_MP[1,
                                index_noise,
                                j] = (np.var(direction_MP[1:]))
            range_dir_elements_MP[1,
                                  index_noise,
                                  j] = (np.max(abs(direction_MP[1:])) -
                                        np.min(abs(direction_MP[1:])))
        index_noise += 1
    return (var_dir_elements, var_dir_elements_MP,
            range_dir_elements, range_dir_elements_MP)



def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots_ratio_2(arr1, arr2, labels,  ticks, title, m, n,
                            lambda_max, no_vars, region, function_type,
                            type_dir):
    """Create boxplots."""
    plt.figure(figsize=(5, 5))
    plt.yscale("log")
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    set_box_color(bpl, 'green')
    set_box_color(bpr, 'red')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='red', label=labels[1])
    plt.xlabel(r'SNR', size=14)
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_ratio_m_%s_n_%s_lambda_max_%s_%s_%s_%s_%s.png' %
                (title, m, n, lambda_max,
                 no_vars, region, function_type, type_dir))


if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    region = float(sys.argv[4])
    function_type = str(sys.argv[5])

    if function_type == 'quad':
        f = est_dir.quad_f_noise
        f_no_noise = est_dir.quad_f
    elif function_type == 'sqr_quad':
        f = est_dir.sqrt_quad_f_noise
        f_no_noise = est_dir.sqrt_quad_f
    elif function_type == 'squ_quad':
        f = est_dir.square_quad_f_noise
        f_no_noise = est_dir.square_quad_f
    else:
        raise ValueError('Incorrect function choice')
    num_funcs = 100
    cov = np.identity(m)
    no_vars = m
    snr_list = [0.5, 2, 5]
    ticks = snr_list
    labels = ['Random', 'Even']
    sp_func_vals_init = (est_dir.calc_initial_func_values(
                         m, num_funcs, lambda_max, cov, f_no_noise))

    noise_list = est_dir.compute_var_quad_form(snr_list, sp_func_vals_init,
                                               region)

    (var_dir_elements,
     var_dir_elements_MP,
     range_dir_elements,
     range_dir_elements_MP) = (compute_var_diff_search_direction(
                               noise_list, cov, n, m, lambda_max,
                               no_vars, f, region))
    title = 'variance'
    create_boxplots_ratio_2(var_dir_elements[0], var_dir_elements[1], labels,
                            ticks, title, m, n, lambda_max, no_vars, region,
                            function_type, 'MY')
    create_boxplots_ratio_2(var_dir_elements_MP[0], var_dir_elements_MP[1],
                            labels, ticks, title, m, n, lambda_max, no_vars,
                            region, function_type, 'MPI')
    title = 'range'
    create_boxplots_ratio_2(range_dir_elements[0], range_dir_elements[1], labels,
                            ticks, title, m, n, lambda_max, no_vars, region,
                            function_type, 'MY')
    create_boxplots_ratio_2(range_dir_elements_MP[0], range_dir_elements_MP[1],
                            labels, ticks, title, m, n, lambda_max, no_vars,
                            region, function_type, 'MPI')
