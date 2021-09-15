import sys

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


def bar_charts(arr1, arr2, arr3, arr4, arr5, arr6, title_pdf, n, m,
               function_type, region):
    plt.figure(figsize=(7, 5))
    ticks = ['1', '4', '8']
    plt.xlabel(r'$\lambda_{max}$', fontsize=15)
    X_num = np.arange(len(ticks)) * 2
    plt.bar(X_num - 0.8, arr1, 0.3, color=sns.xkcd_rgb["pale red"])
    plt.bar(X_num - 0.48, arr2, 0.3, color=sns.xkcd_rgb["medium blue"])
    plt.bar(X_num - 0.16, arr3, 0.3, color=sns.xkcd_rgb["medium purple"])
    plt.bar(X_num + 0.16, arr4, 0.3, color=sns.xkcd_rgb["medium green"])
    plt.bar(X_num + 0.48, arr5, 0.3, color=sns.xkcd_rgb["pale orange"])
    plt.bar(X_num + 0.8, arr6, 0.3, color=sns.xkcd_rgb["pale pink"])
    plt.xticks(X_num, ticks, fontsize=15)
    plt.yticks(fontsize=15)
    red_patch = mpatches.Patch(color=sns.xkcd_rgb["pale red"],
                               label='SNR=0.5')
    blue_patch = mpatches.Patch(color=sns.xkcd_rgb["medium blue"],
                                label='SNR=1')
    purple_patch = mpatches.Patch(color=sns.xkcd_rgb["medium purple"],
                                  label='SNR=2')
    green_patch = mpatches.Patch(color=sns.xkcd_rgb["medium green"],
                                 label='SNR=3')
    orange_patch = mpatches.Patch(color=sns.xkcd_rgb["pale orange"],
                                 label='SNR=5')
    pink_patch = mpatches.Patch(color=sns.xkcd_rgb["pale pink"],
                                 label='SNR=10')
    plt.legend(handles=[red_patch, blue_patch, purple_patch, green_patch,
                        orange_patch, pink_patch],
                bbox_to_anchor=[1.34, 1.03], loc='upper right',
                prop={'size': 15})
    plt.ylim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('%s_n_%s_m_%s_%s_%s.png' %
                (title_pdf, n, m, region, function_type),
                bbox_inches="tight")


def compute_no_times_XY_gr_LS(lambda_max_list, snr_list, num_funcs,
                              fp_norms_LS, fp_norms_XY, fp_func_vals_LS,
                              fp_func_vals_XY):
    XY_gr_LS_norms = np.zeros((len(lambda_max_list), len(snr_list)))
    XY_gr_LS_func = np.zeros((len(lambda_max_list), len(snr_list)))
    for j in range(len(lambda_max_list)):
        for i in range(len(snr_list)):
            XY_gr_LS_norms[j, i] = (np.where(fp_norms_XY[j, i] >
                                             fp_norms_LS[j, i])[0].shape[0] /
                                    fp_norms_XY[j, i].shape[0])
            XY_gr_LS_func[j, i] = (np.where(fp_func_vals_XY[j, i] >
                                            fp_func_vals_LS[j, i])[0].shape[0]
                                   / fp_func_vals_XY[j, i].shape[0])
    return XY_gr_LS_norms, XY_gr_LS_func


if __name__ == "__main__":
    function_type = str(sys.argv[1])
    save_outputs = None
    n = 16
    m = 10
    region = 0.1
    snr_list = [0.5, 1, 2, 3, 5, 10]
    lambda_max_list = [1, 4, 8]
    no_vars = m
    num_funcs = 100

    fp_norms_LS = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    fp_func_vals_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                num_funcs))
    fp_func_vals_noise_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                      num_funcs))
    func_evals_step_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                   num_funcs))
    func_evals_dir_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                  num_funcs))
    good_dir_prop_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    good_dir_norm_LS = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    no_its_LS = np.zeros((len(lambda_max_list), len(snr_list),
                          num_funcs))

    fp_norms_XY = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    fp_func_vals_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                num_funcs))
    fp_func_vals_noise_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                      num_funcs))
    func_evals_step_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                   num_funcs))
    func_evals_dir_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                  num_funcs))
    good_dir_prop_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    good_dir_norm_XY = np.zeros((len(lambda_max_list), len(snr_list),
                                 num_funcs))
    no_its_XY = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))

    sp_norms = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))
    sp_func_vals = np.zeros((len(lambda_max_list), len(snr_list), num_funcs))

    lambda_max_index = 0
    for lambda_max in lambda_max_list:
        fp_norms_LS[lambda_max_index] = (np.genfromtxt(
                                         'fp_norms_LS_n=%s_m=%s_lambda_max=%s'
                                         '_%s_%s_%s.csv' %
                                         (n, m, lambda_max,
                                          no_vars,
                                          region,
                                          function_type),
                                         delimiter=','))

        fp_func_vals_LS[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_LS_n=%s_m=%s_lambda'
                                             '_max=%s_%s_%s_%s.csv' %
                                             (n, m, lambda_max,
                                              no_vars,
                                              region,
                                              function_type),
                                             delimiter=','))

        fp_norms_XY[lambda_max_index] = (np.genfromtxt(
                                         'fp_norms_XY_n=%s_m=%s_lambda_max'
                                         '=%s_%s_%s_%s_%s.csv' %
                                         (n, m, lambda_max,
                                          no_vars,
                                          region,
                                          function_type,
                                          save_outputs),
                                         delimiter=','))

        fp_func_vals_XY[lambda_max_index] = (np.genfromtxt(
                                             'fp_func_vals_XY_n=%s_m=%s_lambda'
                                             '_max=%s_%s_%s_%s_%s.csv' %
                                             (n, m, lambda_max,
                                              no_vars,
                                              region,
                                              function_type,
                                              save_outputs),
                                             delimiter=','))

        sp_norms[lambda_max_index] = (np.genfromtxt(
                                      'sp_norms_LS_n=%s_m=%s_lambda_max=%s'
                                      '_%s.csv' %
                                      (n, m, lambda_max, function_type),
                                      delimiter=','))

        sp_func_vals[lambda_max_index] = (np.genfromtxt(
                                          'sp_func_vals_LS_n=%s_m=%s_lambda'
                                          '_max=%s_%s.csv' %
                                          (n, m, lambda_max,
                                           function_type),
                                          delimiter=','))
        lambda_max_index += 1

    (XY_gr_LS_norms,
     XY_gr_LS_func) = compute_no_times_XY_gr_LS(lambda_max_list, snr_list,
                                                num_funcs, fp_norms_LS,
                                                fp_norms_XY, fp_func_vals_LS,
                                                fp_func_vals_XY)
    bar_charts(XY_gr_LS_norms[:, 0], XY_gr_LS_norms[:, 1],
               XY_gr_LS_norms[:, 2], XY_gr_LS_norms[:, 3],
               XY_gr_LS_norms[:, 4], XY_gr_LS_norms[:, 5],
               'norms', n, m, function_type, region)

    bar_charts(XY_gr_LS_func[:, 0], XY_gr_LS_func[:, 1],
               XY_gr_LS_func[:, 2], XY_gr_LS_func[:, 3],
               XY_gr_LS_func[:, 4], XY_gr_LS_func[:, 5],
               'func', n, m, function_type, region)
