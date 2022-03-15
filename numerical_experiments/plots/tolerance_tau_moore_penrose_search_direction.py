import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

import est_dir


def compute_MP_diff_search_direction(noise_list, tol_list, cov, n, m,
                                     lambda_max, no_vars, f, region):
    """
    Compares the variance and range of elements of search directions, where
    different tolerances are used for the inclusion of singular values within
    the Moore-Penrose pseudoinverse.

    Parameters
    ----------
    noise_list : 1-D array
                 Store standard deviation of errors used to compute noisy
                 function values.
    tol_list : 1-D array
               Store list of tolerances used for the inclusion of singular
               values within the Moore-Penrose pseudoinverse.     
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
    var_dir_elements_MP : 3-D array
                          Returns the variance of the search direction
                          for 100 functions with different standard
                          deviation of errors.
    range_dir_elements_MP: 3-D array                 
                           Returns the absolute maximum coefficient of the
                           search direction subtracted by the absolute minimum
                           coefficient of the search direction,
                           for 100 functions with different standard
                           deviation of errors.
    """
    var_dir_elements_MP = np.zeros((len(noise_list), len(tol_list), 100))
    range_dir_elements_MP = np.zeros((len(noise_list),len(tol_list), 100))
    index_noise = 0
    for noise_sd in noise_list:
        index_tol = 0
        for tol in tol_list:
            for j in range(100):
                (centre_point,
                 minimizer,
                 matrix) = est_dir.generate_func_params(j, cov, m, lambda_max)
                func_args = (minimizer, matrix, 0, noise_sd)

                np.random.seed(j)
                (act_design,
                 y, positions,
                 func_evals) = (est_dir.compute_random_design
                                (n, m, centre_point, no_vars,
                                 f, func_args, region))
                full_act_design = np.ones((act_design.shape[0],
                                           act_design.shape[1] + 1))
                full_act_design[:, 1:] = act_design
                direction_MP = (np.linalg.pinv(full_act_design, tol) @ y)
                var_dir_elements_MP[index_noise,
                                    index_tol,
                                    j] = (np.var(direction_MP[1:]))
                range_dir_elements_MP[index_noise,
                                      index_tol,
                                      j] = (np.max(abs(direction_MP[1:])) -
                                            np.min(abs(direction_MP[1:])))
            index_tol += 1
        index_noise += 1
    return (var_dir_elements_MP, range_dir_elements_MP)


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots_ratio_3(arr1, arr2, arr3, labels,  ticks, title, m, n,
                            lambda_max, no_vars, region, function_type):
    """Create boxplots."""
    plt.figure(figsize=(5, 6))
    plt.yscale("log")
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*3.0-0.6)
    bpc = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr1)))*3.0)
    bpr = plt.boxplot(arr3.T,
                      positions=np.array(range(len(arr1)))*3.0+0.6)
    set_box_color(bpl, 'green')
    set_box_color(bpc, 'red')
    set_box_color(bpr, 'blue')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='red', label=labels[1])
    plt.plot([], c='blue', label=labels[2])
    
    green_patch = mpatches.Patch(color=sns.xkcd_rgb["medium green"],
                                 label='SNR=0.5')
    red_patch = mpatches.Patch(color=sns.xkcd_rgb["pale red"],
                               label='SNR=2')
    blue_patch = mpatches.Patch(color=sns.xkcd_rgb["medium blue"],
                                label='SNR=5')
   
    plt.legend(handles=[green_patch, red_patch, blue_patch],
               bbox_to_anchor=[1.02, 1.02], loc='upper right',
               prop={'size': 15})    

    plt.xlabel(r'$\tau$', size=18)
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_diff_tol_MP_m_%s_n_%s_lambda_max_%s_%s_%s_%s.png' %
               (title, m, n, lambda_max,
                no_vars, region, function_type))


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
    tol_list = [0.000000000000001, 0.001, 0.01, 0.15]
    labels = snr_list
    ticks = tol_list
    sp_func_vals_init = (est_dir.calc_initial_func_values(
                        m, num_funcs, lambda_max, cov, f_no_noise))

    noise_list = est_dir.compute_var_quad_form(snr_list, sp_func_vals_init,
                                               region)

    (var_dir_elements_MP,
    range_dir_elements_MP) = (compute_MP_diff_search_direction(
                            noise_list, tol_list, cov, n, m, lambda_max,
                            no_vars, f, region))
    title = 'var'
    create_boxplots_ratio_3(var_dir_elements_MP[0], var_dir_elements_MP[1],
                            var_dir_elements_MP[2], labels,
                            ticks, title, m, n, lambda_max, no_vars, region,
                            function_type)
    title = 'range'
    create_boxplots_ratio_3(range_dir_elements_MP[0], range_dir_elements_MP[1],
                            range_dir_elements_MP[2], labels,
                            ticks, title, m, n, lambda_max, no_vars, region,
                            function_type)
