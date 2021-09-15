import numpy as np
import est_dir
import matplotlib.pyplot as plt


def check_tolerance(s, const):
    tol = np.max(s) * const
    pos = np.where(abs(s) > tol)[0]
    count = s.shape[0] - np.where(abs(s) > tol)[0].shape[0]
    return s[pos], count


def create_boxplots_MP(min_eig, const_list):
    plt.figure(figsize=(5, 5))
    for j in range(len(min_eig)):
        plt.clf()
        plt.figure(figsize=(5, 5))
        plt.ylim(0, np.max(min_eig))
        bp = plt.boxplot(min_eig[j].T)
        plt.setp(bp['boxes'], color='blue')
        plt.setp(bp['whiskers'], color='blue')
        plt.setp(bp['caps'], color='blue')
        plt.setp(bp['medians'], color='red')
        plt.xlabel(r'$N$', size=14)
        plt.xticks(np.arange(1, 4), ['50', '100', '200'], size=15)
        plt.savefig('Moore_Penrose_experiment_tol_%s.png' %
                    (const_list[j]))


def mean_no_eigs_removed(count_eigs_removed, const_list, n_list):
    store_means = np.zeros((len(const_list), len(n_list)))

    for j in range(len(const_list)):
        for i in range(len(n_list)):
            if i == 0:
                assert(np.all(count_eigs_removed[j, i] == 51))
            elif i == 2:
                assert(np.all(count_eigs_removed[j, i] == 0))
            store_means[j, i] = np.mean(count_eigs_removed[j, i])
    np.savetxt('store_mean_number_eigenvalues_removed_MP_tol.csv', store_means, delimiter=',')


if __name__ == "__main__":
    seed = 100
    m = 100
    n_list = [50, 100, 200]
    const_list = [0.000000000000001, 0.001, 0.01, 0.025]
    min_eig = np.zeros((len(const_list), len(n_list), 100))
    count_eigs_removed = np.zeros((len(const_list),len(n_list), 100))
    no_vars = m
    region = 0.1
    const = 1
    cov = np.identity((m))
    lambda_max = 1
    noise_sd = 1
    f = est_dir.quad_f_noise
    index_const = 0
    for const in const_list:
        index = 0
        for n in n_list:
            np.random.seed(seed)
            for j in range(100):
                centre_point = np.random.multivariate_normal(np.zeros((m)),
                                                             cov)
                minimizer = np.zeros((m, ))
                matrix = est_dir.quad_func_params(1, lambda_max, m)
                func_args = (minimizer, matrix, 0, noise_sd)
                (act_design, y,
                 positions, func_evals) = (est_dir.compute_random_design
                                           (n, m, centre_point, no_vars,
                                            f, func_args, region))
                full_act_design = np.ones((act_design.shape[0],
                                           act_design.shape[1] + 1))
                full_act_design[:, 1:] = act_design
                u, s, vh = np.linalg.svd(full_act_design.T @ full_act_design)
                new_s, count = check_tolerance(s, const)
                min_eig[index_const, index, j] = np.min(new_s)
                count_eigs_removed[index_const, index, j] = count
            index += 1
        index_const += 1

    create_boxplots_MP(min_eig, const_list)
    mean_no_eigs_removed(count_eigs_removed, const_list, n_list)
