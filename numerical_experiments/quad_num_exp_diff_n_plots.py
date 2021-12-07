import sys

import numpy as np
import matplotlib.pyplot as plt


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots_ratio_2(arr1, arr2, labels, m,
                            lambda_max, title, ticks,
                            region, function_type, snr_list,
                            snr_pos, func_evals):
    """Create boxplots."""
    plt.figure(figsize=(5, 5))
    plt.ylim(-0.01, 1)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    set_box_color(bpl, 'green')
    set_box_color(bpr, 'purple')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.xlabel(r'$N$', size=14)
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_%s_ratio_m_%s_lambda_max_%s_%s_%s_%s.png' %
                (title, snr_list[snr_pos], m, lambda_max,
                 region, function_type, func_evals))


if __name__ == "__main__":
    m = int(sys.argv[1])
    lambda_max = int(sys.argv[2])
    region = float(sys.argv[3])
    function_type = str(sys.argv[4])
    func_evals = int(sys.argv[5])

    n_list = [16, 32, 50, 100, 200]
    snr_list = [0.5, 1, 2, 3, 5, 10]
    no_vars = m
    num_funcs = 100

    if func_evals == 0:
        save_outputs = None
    else:
        save_outputs = func_evals

    fp_norms_XY = np.zeros((len(snr_list), len(n_list), num_funcs))
    fp_func_vals_XY = np.zeros((len(snr_list), len(n_list), num_funcs))
    func_evals_step_XY = np.zeros((len(snr_list), len(n_list), num_funcs))
    func_evals_dir_XY = np.zeros((len(snr_list), len(n_list), num_funcs))

    fp_norms_MP = np.zeros((len(snr_list), len(n_list), num_funcs))
    fp_func_vals_MP = np.zeros((len(snr_list), len(n_list), num_funcs))
    func_evals_step_MP = np.zeros((len(snr_list), len(n_list), num_funcs))
    func_evals_dir_MP = np.zeros((len(snr_list), len(n_list), num_funcs))

    sp_norms = np.zeros((len(snr_list), len(n_list), num_funcs))
    sp_func_vals = np.zeros((len(snr_list), len(n_list), num_funcs))

    for i in range(len(snr_list)):
        for j in range(len(n_list)):
            n = n_list[j]

            fp_norms_MP[i, j] = (np.genfromtxt(
                                 'fp_norms_MP_n=%s_m=%s_lambda_max=%s_'
                                 '%s_%s_%s_%s.csv' %
                                 (n, m, lambda_max,
                                  no_vars,
                                  region,
                                  function_type,
                                  save_outputs),
                                 delimiter=',')[i])

            fp_func_vals_MP[i, j] = (np.genfromtxt(
                                     'fp_func_vals_MP_n=%s_m=%s_lambda_max'
                                     '=%s_%s_%s_%s_%s.csv' %
                                     (n, m, lambda_max,
                                      no_vars,
                                      region,
                                      function_type,
                                      save_outputs),
                                     delimiter=',')[i])

            func_evals_step_MP[i, j] = (np.genfromtxt(
                                        'func_evals_step_MP_n=%s_m=%s_lambda_'
                                        'max=%s_%s_%s_%s_%s.csv' %
                                        (n, m, lambda_max,
                                         no_vars,
                                         region,
                                         function_type,
                                         save_outputs),
                                        delimiter=',')[i])

            func_evals_dir_MP[i, j] = (np.genfromtxt(
                                       'func_evals_dir_MP_n=%s_m=%s_lambda'
                                       '_max=%s_%s_%s_%s_%s.csv' %
                                       (n, m, lambda_max,
                                        no_vars,
                                        region,
                                        function_type,
                                        save_outputs),
                                       delimiter=',')[i])

            fp_norms_XY[i, j] = (np.genfromtxt(
                                 'fp_norms_XY_n=%s_m=%s_lambda_max=%s'
                                 '_%s_%s_%s_%s.csv' %
                                 (n, m, lambda_max,
                                  no_vars,
                                  region,
                                  function_type,
                                  save_outputs),
                                 delimiter=',')[i])

            fp_func_vals_XY[i, j] = (np.genfromtxt(
                                     'fp_func_vals_XY_n=%s_m=%s_lambda_max'
                                     '=%s_%s_%s_%s_%s.csv' %
                                     (n, m, lambda_max,
                                      no_vars,
                                      region,
                                      function_type,
                                      save_outputs),
                                     delimiter=',')[i])

            func_evals_step_XY[i, j] = (np.genfromtxt(
                                        'func_evals_step_XY_n=%s_m=%s_lambda'
                                        '_max=%s_%s_%s_%s_%s.csv' %
                                        (n, m, lambda_max,
                                         no_vars,
                                         region,
                                         function_type,
                                         save_outputs),
                                        delimiter=',')[i])

            func_evals_dir_XY[i, j] = (np.genfromtxt(
                                       'func_evals_dir_XY_n=%s_m=%s_lambda'
                                       '_max=%s_%s_%s_%s_%s.csv' %
                                       (n, m, lambda_max,
                                        no_vars,
                                        region,
                                        function_type,
                                        save_outputs),
                                       delimiter=',')[i])

            sp_norms[i, j] = (np.genfromtxt(
                              'sp_norms_XY_n=%s_m=%s_lambda_max'
                              '=%s_%s_%s.csv' %
                              (n, m, lambda_max, function_type, save_outputs),
                              delimiter=',')[i])

            sp_func_vals[i, j] = (np.genfromtxt(
                                  'sp_func_vals_XY_n=%s_m=%s_lambda_max'
                                  '=%s_%s_%s.csv' %
                                  (n, m, lambda_max, function_type,
                                   save_outputs),
                                  delimiter=',')[i])

    labels = [[r'$||x^{(1)} - x^{*}||$',
               r'$||x_{LS}^{(K)} - x^{*}||$',
               r'$||x_{MP}^{(K)} - x^{*}||$',
               r'$||x_{MY}^{(K)} - x^{*}||$'],
              [r'$\eta(x^{(1)})$',
               r'$\eta(x_{LS}^{(K)})$',
               r'$\eta(x_{MP}^{(K)})$',
               r'$\eta(x_{MY}^{(K)})$'],
              [r'$\frac{||x_{LS}^{(K)} - x^{*}||}{||x^{(1)} - x^{*}||}$',
               r'$\frac{||x_{MP}^{(K)} - x^{*}||}{||x^{(1)} - x^{*}||}$',
               r'$\frac{||x_{MY}^{(K)} - x^{*}||}{||x^{(1)} - x^{*}||}$'],
              [r'$\frac{\eta(x_{LS}^{(K)})}{\eta(x^{(1)})}$',
               r'$\frac{\eta(x_{MP}^{(K)})}{\eta(x^{(1)})}$',
               r'$\frac{\eta(x_{MY}^{(K)})}{\eta(x^{(1)})}$'],
              ['MP', 'MY']]

    for i in range(len(snr_list)):

        create_boxplots_ratio_2(fp_norms_MP[i]/sp_norms[i],
                                fp_norms_XY[i]/sp_norms[i], labels[2][1:],
                                m, lambda_max, 'norms_MP_XY', n_list,
                                region, function_type, snr_list, i, save_outputs)

        create_boxplots_ratio_2(fp_func_vals_MP[i]/sp_func_vals[i],
                                fp_func_vals_XY[i]/sp_func_vals[i],
                                labels[3][1:], m, lambda_max,
                                'func_vals_MP_XY', n_list, region,
                                function_type, snr_list, i, save_outputs)
