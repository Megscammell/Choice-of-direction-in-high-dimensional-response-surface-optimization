import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots(arr1, arr2, arr3, labels, m, n,
                    lambda_max, title, ticks):
    """Generate boxplots for data."""
    plt.figure(figsize=(8, 5))
    if title == 'func_vals':
        plt.yscale("log")
        plt.ylim(0.01, np.max(arr1) + 1000)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*3.0-0.7)
    bpc = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*3.0+0)
    bpr = plt.boxplot(arr3.T,
                      positions=np.array(range(len(arr3)))*3.0+0.7)
    set_box_color(bpl, 'green')
    set_box_color(bpc, 'purple')
    set_box_color(bpr, 'navy')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.plot([], c='navy', label=labels[2])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 12})
    plt.xlabel(r'SNR', size=14)
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_m_%s_n_%s_lambda_max_%s_diff_noise.pdf' %
                 (title, m, n, lambda_max))

                 
def create_boxplots_funv_evals(arr1, arr2, m, n, lambda_max, option,
                               ticks, max_func_evals):
    plt.figure(figsize=(8, 5))
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    plt.plot([], c='#D7191C', label=r'$\gamma_k$')
    plt.plot([], c='#2C7BB6', label=r'$s_k$')
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 14})
    plt.xlabel(r'SNR', size=14)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.ylim(0, max_func_evals)
    plt.tight_layout()
    plt.savefig('func_evals_m_%s_n_%s_lambda_max_%s_diff_noise_%s.pdf' %
                (m, n, lambda_max, option))


def create_boxplots_ratio(arr1, arr2, labels, m, n,
                          lambda_max, title, ticks):
    plt.figure(figsize=(8, 5))
    plt.ylim(-0.01,1)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    set_box_color(bpl, 'green')
    set_box_color(bpr, 'navy')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='navy', label=labels[1])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 17})
    plt.xlabel(r'SNR', size=14)
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('%s_ratio_m_%s_n_%s_lambda_max_%s_diff_noise.pdf' %
               (title, m, n, lambda_max))


def create_scatter_plot(arr1, arr2, arr1_title, arr2_title, labels_legend,
                        title, m, n, lambda_max):
    plt.figure(figsize=(8, 5))
    max_num = max(np.max(arr1), np.max(arr2)) + 1
    plt.xlim(-0.01, max_num)
    plt.ylim(-0.01, max_num)
    plt.xlabel(arr1_title, size=14)
    plt.ylabel(arr2_title, size=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    color_list = ['red', 'blue', 'purple', 'green']
    for j in range(arr1.shape[0]):
        plt.scatter(arr1[j], arr2[j], marker='*', color=color_list[j])
        plt.plot([], c=color_list[j], label=labels_legend[j])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 14})
    plt.plot([0, max_num],[0, max_num], color='black')
    plt.tight_layout() 
    plt.savefig('%s_scatter_m_%s_n_%s_lambda_max_%s_diff_noise.pdf' %
                (title, m, n, lambda_max))


if __name__ == "__main__":
    n =  int(sys.argv[1])
    m = int(sys.argv[2])
    lambda_max = int(sys.argv[3])
    if lambda_max == 1:
        if m == 100:
            max_func_evals = 40000
        elif m == 200:
            max_func_evals = 20000
    elif lambda_max == 8:
        if m == 100:
            max_func_evals = 80000
        elif m == 200:
            max_func_evals = 50000
    elif lambda_max == 16:
        if m == 100:
            max_func_evals = 120000
        elif m == 200:
            max_func_evals = 100000
    elif lambda_max == 64:
        if m == 100:
            max_func_evals = 120000
        elif m == 200:
            max_func_evals = 140000
    else:
        raise ValueError('Incorrect lambda_max')
    domain = (0, 5)
    num_funcs = 100
    snr_list = [0.01, 0.1, 0.2, 0.3]
    
    fp_norms_LS = np.genfromtxt('fp_norms_LS_n=%s_m=%s_lambda_max=%s.csv' %
                                (n, m, lambda_max),
                                 delimiter=',')

    fp_func_vals_LS = np.genfromtxt('fp_func_vals_LS_n=%s_m=%s_lambda_max=%s.csv' %
                                (n, m, lambda_max),
                                 delimiter=',')

    func_evals_step_LS = np.genfromtxt('func_evals_step_LS_n=%s_m=%s_lambda_max=%s.csv' %
                                    (n, m, lambda_max),
                                     delimiter=',')

    func_evals_dir_LS = np.genfromtxt('func_evals_dir_LS_n=%s_m=%s_lambda_max=%s.csv' %
                                    (n, m, lambda_max),
                                     delimiter=',')

    fp_norms_XY = np.genfromtxt('fp_norms_XY_n=%s_m=%s_lambda_max=%s.csv' %
                                (n, m, lambda_max),
                                 delimiter=',')

    fp_func_vals_XY = np.genfromtxt('fp_func_vals_XY_n=%s_m=%s_lambda_max=%s.csv' %
                                (n, m, lambda_max),
                                 delimiter=',')
    
    func_evals_step_XY = np.genfromtxt('func_evals_step_XY_n=%s_m=%s_lambda_max=%s.csv' %
                                    (n, m, lambda_max),
                                     delimiter=',')

    func_evals_dir_XY = np.genfromtxt('func_evals_dir_XY_n=%s_m=%s_lambda_max=%s.csv' %
                                        (n, m, lambda_max),
                                        delimiter=',')
    
    sp_norms = np.genfromtxt('sp_norms_n=%s_m=%s_lambda_max=%s.csv' %
                             (n, m, lambda_max), delimiter=',')

    sp_func_vals = np.genfromtxt('sp_func_vals_n=%s_m=%s_lambda_max=%s.csv' %
                                 (n, m, lambda_max), delimiter=',')
    
    #Generate boxplots
    labels = [[r'$||x_1 - x_{*}||$',
                r'$||x_{K}^{(LS)} - x_{*}||$',
                r'$||x_{K}^{(MY)} - x_{*}||$'],
                [r'$f(x_1)$',
                r'$f(x_{K}^{(LS)})$',
                r'$f(x_{K}^{(MY)})$'],
                [r'$\frac{||x_{K}^{(LS)} - x_{*}||}{||x_1 - x_{*}||}$',
                r'$\frac{||x_{K}^{(MY)} - x_{*}||}{||x_1 - x_{*}||}$'],
                [r'$\frac{f(x_{K}^{(LS)})}{f(x_1)}$',
                r'$\frac{f(x_{K}^{(MY)})}{f(x_1)}$']]
    
    create_boxplots(sp_norms, fp_norms_LS, fp_norms_XY, labels[0], m, n,
                   lambda_max, 'norms', snr_list)

    create_boxplots(sp_func_vals, fp_func_vals_LS, fp_func_vals_XY, labels[1], m, n,
                     lambda_max, 'func_vals', snr_list)

    create_boxplots_funv_evals(func_evals_step_LS, func_evals_dir_LS, m, n,
                               lambda_max, 'LS', snr_list, max_func_evals)

    create_boxplots_funv_evals(func_evals_step_XY, func_evals_dir_XY, m, n,
                                lambda_max, 'XY', snr_list, max_func_evals)

    create_boxplots_ratio(fp_norms_LS/sp_norms, fp_norms_XY/sp_norms, labels[2],
                           m, n, lambda_max, 'norms', snr_list)

    create_boxplots_ratio(fp_func_vals_LS/sp_func_vals, fp_func_vals_XY/sp_func_vals,
                          labels[3], m, n, lambda_max, 'func_vals', snr_list)
    labels_legend = []
    for j in range(len(snr_list)):
        labels_legend.append(r'SNR = %s' % snr_list[j])

    create_scatter_plot(fp_norms_LS, fp_norms_XY, labels[0][1], labels[0][2],
                        labels_legend, 'norms', m, n, lambda_max)
    
    create_scatter_plot(fp_func_vals_LS, fp_func_vals_XY, labels[1][1],
                        labels[1][2], labels_legend, 'func_vals', m, n,
                        lambda_max)