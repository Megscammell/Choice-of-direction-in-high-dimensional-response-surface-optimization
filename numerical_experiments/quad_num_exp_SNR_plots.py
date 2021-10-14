import sys

import numpy as np
import est_dir
import matplotlib.pyplot as plt
import seaborn as sns


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots_ratio_2(arr1, arr2, labels, m, n, lambda_max, title, ticks,
                            no_vars, range_1, range_2, region, function_type,
                            func_evals):
    """Create boxplots."""
    plt.figure(figsize=(5, 5))
    plt.ylim(range_1, range_2)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    set_box_color(bpl, 'navy')
    set_box_color(bpr, 'purple')
    plt.plot([], c='navy', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.xlabel(r'SNR', size=14)
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_ratio_m_%s_n_%s_lambda_max_%s_%s_%s_%s_%s_%s.png' %
                (title, m, n, lambda_max,
                 no_vars, no_vars,
                 region, function_type, func_evals))


def create_boxplots_ratio_3(arr1, arr2, arr3, labels, m, n, lambda_max, title,
                            ticks, no_vars, range_1, range_2, region,
                            function_type, func_evals):
    """Create boxplots."""
    plt.figure(figsize=(5, 5))
    plt.ylim(range_1, range_2)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*3.0-0.6)
    bpc = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr1)))*3.0)
    bpr = plt.boxplot(arr3.T,
                      positions=np.array(range(len(arr2)))*3.0+0.6)
    set_box_color(bpl, 'green')
    set_box_color(bpc, 'navy')
    set_box_color(bpr, 'purple')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='navy', label=labels[1])
    plt.plot([], c='purple', label=labels[2])
    plt.xlabel(r'SNR', size=14)
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_ratio_m_%s_n_%s_lambda_max_%s_%s_%s_%s_%s.png' %
                (title, m, n, lambda_max,
                 no_vars, region, function_type, func_evals))


def create_scatter_plot(arr1, arr2, arr1_title, arr2_title, labels_legend,
                        title, m, n, lambda_max, no_vars, max_num,
                        region, function_type, func_evals):
    """Create scatter plots."""
    plt.figure(figsize=(7, 5))
    plt.ylim(-0.1, max_num)
    plt.xlim(-0.1, max_num)
    plt.xlabel(arr1_title, size=14)
    plt.ylabel(arr2_title, size=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    color_list = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium blue"],
                  sns.xkcd_rgb["medium purple"], sns.xkcd_rgb["medium green"],
                  sns.xkcd_rgb["pale orange"], sns.xkcd_rgb["pale pink"]]
    for j in range(arr1.shape[0]):
        plt.scatter(arr1[j], arr2[j], marker='*', color=color_list[j])
        plt.plot([], c=color_list[j], label=labels_legend[j])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 14})
    plt.plot([0, max_num], [0, max_num], color='black')
    plt.tight_layout()
    plt.savefig('%s_scatter_m_%s_n_%s_lambda_max_%s_%s_%s_%s_%s.png' %
                (title, m, n, lambda_max,
                 no_vars, region, function_type, func_evals))


if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    region = float(sys.argv[4])
    function_type = str(sys.argv[5])
    type_inverse = str(sys.argv[6])
    func_evals = int(sys.argv[7])

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
    snr_list = [0.5, 1, 2, 3, 5, 10]

    if func_evals == 0:
        store_max_func_evals = None
        save_outputs = None
    else:
        store_max_func_evals = np.repeat(func_evals, len(snr_list))
        save_outputs = store_max_func_evals[0]

    if store_max_func_evals is None:
        fp_norms_LS = (np.genfromtxt(
                       'fp_norms_LS_n=%s_m=%s_lambda_max=%s_%s_%s_%s.csv' %
                       (16, m, lambda_max,
                        10,
                        region,
                        function_type),
                       delimiter=','))

        fp_func_vals_LS = (np.genfromtxt(
                           'fp_func_vals_LS_n=%s_m=%s_lambda_max=%s'
                           '_%s_%s_%s.csv' %
                           (16, m, lambda_max,
                            10,
                            region,
                            function_type),
                           delimiter=','))

    fp_norms_MP = (np.genfromtxt(
                   'fp_norms_MP_n=%s_m=%s_lambda_max=%s_%s_%s_%s_%s_%s.csv' %
                   (n, m, lambda_max,
                    no_vars,
                    type_inverse,
                    region,
                    function_type,
                    save_outputs),
                   delimiter=','))

    fp_func_vals_MP = (np.genfromtxt(
                       'fp_func_vals_MP_n=%s_m=%s_lambda_max=%s_%s'
                       '_%s_%s_%s_%s.csv' %
                       (n, m, lambda_max,
                        no_vars,
                        type_inverse,
                        region,
                        function_type,
                        save_outputs),
                       delimiter=','))

    fp_norms_XY = (np.genfromtxt(
                   'fp_norms_XY_n=%s_m=%s_lambda_max=%s_%s'
                   '_%s_%s_%s.csv' %
                   (n, m, lambda_max,
                    no_vars,
                    region,
                    function_type,
                    save_outputs),
                   delimiter=','))

    fp_func_vals_XY = (np.genfromtxt(
                       'fp_func_vals_XY_n=%s_m=%s_lambda_max=%s_%s'
                       '_%s_%s_%s.csv' %
                       (n, m, lambda_max,
                        no_vars,
                        region,
                        function_type,
                        save_outputs),
                       delimiter=','))

    sp_norms = np.genfromtxt('sp_norms_XY_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
                             (n, m, lambda_max, function_type, save_outputs),
                             delimiter=',')

    sp_func_vals = (np.genfromtxt(
                    'sp_func_vals_XY_n=%s_m=%s_lambda_max=%s_%s_%s.csv' %
                    (n, m, lambda_max, function_type, save_outputs),
                    delimiter=','))

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

    labels_legend = []
    for j in range(len(snr_list)):
        labels_legend.append(r'SNR = %s' % snr_list[j])

    if store_max_func_evals is None:
        range_2_norm_LS_XY = max(np.max(fp_norms_LS/sp_norms),
                                 np.max(fp_norms_XY/sp_norms))
        range_2_func_LS_XY = max(np.max(fp_func_vals_LS/sp_func_vals),
                                 np.max(fp_func_vals_XY/sp_func_vals))

    range_2_norm_MP_XY = max(np.max(fp_norms_MP/sp_norms),
                             np.max(fp_norms_XY/sp_norms))
    range_2_func_MP_XY = max(np.max(fp_norms_MP/sp_norms),
                             np.max(fp_func_vals_XY/sp_func_vals))

    create_boxplots_ratio_2(fp_norms_MP/sp_norms,
                            fp_norms_XY/sp_norms,
                            labels[2][1:], m, n, lambda_max, 'norms_MP_XY',
                            snr_list, no_vars,  -0.01, range_2_norm_MP_XY,
                            region, function_type, save_outputs)

    create_boxplots_ratio_2(fp_func_vals_MP/sp_func_vals,
                            fp_func_vals_XY/sp_func_vals,
                            labels[3][1:], m, n, lambda_max, 'func_vals_MP_XY',
                            snr_list, no_vars, -0.01, range_2_func_MP_XY,
                            region, function_type, save_outputs)
    if store_max_func_evals is None:
        create_boxplots_ratio_2(fp_norms_LS/sp_norms,
                                fp_norms_XY/sp_norms,
                                [labels[2][0], labels[2][2]], m, n, lambda_max,
                                'norms_LS_XY', snr_list, no_vars, -0.01,
                                range_2_norm_LS_XY, region, function_type,
                                save_outputs)

        create_boxplots_ratio_2(fp_func_vals_LS/sp_func_vals,
                                fp_func_vals_XY/sp_func_vals,
                                [labels[3][0], labels[3][2]], m, n, lambda_max,
                                'func_vals_LS_XY', snr_list, no_vars, -0.01,
                                range_2_func_LS_XY, region, function_type,
                                save_outputs)

    max_num_norm = max(np.max(fp_norms_MP), np.max(fp_norms_XY))
    create_scatter_plot(fp_norms_MP, fp_norms_XY, labels[0][2], labels[0][3],
                        labels_legend, 'norms_MP_XY', m, n, lambda_max,
                        no_vars, max_num_norm, region,
                        function_type, save_outputs)

    max_num_func = max(np.max(fp_func_vals_MP), np.max(fp_func_vals_XY))
    create_scatter_plot(fp_func_vals_MP, fp_func_vals_XY, labels[1][2],
                        labels[1][3], labels_legend, 'func_vals_MP_XY', m, n,
                        lambda_max, no_vars, max_num_func, region,
                        function_type, save_outputs)

    if store_max_func_evals is None:
        max_num_norm = max(np.max(fp_norms_LS), np.max(fp_norms_XY))
        create_scatter_plot(fp_norms_LS, fp_norms_XY, labels[0][1],
                            labels[0][3], labels_legend, 'norms_LS_XY', m, n,
                            lambda_max, no_vars, max_num_norm, region,
                            function_type, save_outputs)

        max_num_func = max(np.max(fp_func_vals_LS), np.max(fp_func_vals_XY))
        create_scatter_plot(fp_func_vals_LS, fp_func_vals_XY, labels[1][1],
                            labels[1][3], labels_legend, 'func_vals_LS_XY', m,
                            n, lambda_max, no_vars, max_num_func, region,
                            function_type, save_outputs)

        max_num_norm = max(np.max(fp_norms_LS), np.max(fp_norms_MP))
        create_scatter_plot(fp_norms_LS, fp_norms_MP, labels[0][1],
                            labels[0][2], labels_legend, 'norms_LS_MP', m, n,
                            lambda_max, no_vars, max_num_norm, region,
                            function_type, save_outputs)

        max_num_func = max(np.max(fp_func_vals_LS), np.max(fp_func_vals_MP))
        create_scatter_plot(fp_func_vals_LS, fp_func_vals_MP, labels[1][1],
                            labels[1][2], labels_legend, 'func_vals_LS_MP', m,
                            n, lambda_max, no_vars, max_num_func, region,
                            function_type, save_outputs)
