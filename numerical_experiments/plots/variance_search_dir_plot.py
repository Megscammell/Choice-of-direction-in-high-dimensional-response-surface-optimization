import matplotlib.pyplot as plt
import numpy as np

import est_dir

def compute_trajectory(f, func_args, points, title):
    """
    Plot step lengths along the search direction.

    Parameters
    -----------
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
    points : 2D array
             Points along the search direction corresponding to different step
             sizes.
    """
    plt.clf()
    test_num = 100
    bounds = (-3, 3)
    test_num = 100
    x = np.linspace(*bounds, test_num)
    y = np.linspace(*bounds, test_num)
    Z = np.zeros((test_num, test_num))
    X, Y = np.meshgrid(x, y)
    for i in range(test_num):
        for j in range(test_num):
            x1_var = X[i, j]
            x2_var = Y[i, j]
            Z[i, j] = f(np.array([x1_var, x2_var]).reshape(2, ), *func_args)

    plt.scatter(points[:, 0], points[:, 1], color='black')
    plt.plot(points[:, 0], points[:, 1], color='black')
    plt.scatter(points[0,0], points[0,1], color='red', s=100, marker='o')
    plt.scatter(points[1,0], points[1,1], color='blue', s=100, marker='o')
    plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.25)
    plt.savefig('illustration_variance_of_coeffs_search_%s.png' % (title))


if __name__ == "__main__":
    m = 2
    f = est_dir.quad_f_noise
    minimizer = np.zeros((m))
    matrix = np.array([[1, 0],
                    [0, 1]])
    mu = 0
    sd = 0.25
    func_args = (minimizer, matrix, mu, sd)
    step = 1
    const_back = 0.5
    const_forward = 2
    forward_tol = 10000000
    back_tol = 0.0000000001

    centre_point = np.array([-1.8, -1.8])

    np.random.seed(40)
    direction = np.array([-0.8, -0.7])
    init_func_val = f(centre_point, *func_args)
    (upd_point,
    func_val,
    total_func_evals) = (est_dir.combine_tracking
                        (centre_point, init_func_val,
                        direction, step, const_back, 
                        back_tol, const_forward, 
                        forward_tol, f, func_args))

    compute_trajectory(f, func_args, np.array([centre_point, upd_point]),
                       'same')

    np.random.seed(40)
    direction = np.array([-80, -0.7])
    init_func_val = f(centre_point, *func_args)
    (upd_point,
    func_val,
    total_func_evals) = (est_dir.combine_tracking
                        (centre_point, init_func_val,
                        direction, step, const_back, 
                        back_tol, const_forward, 
                        forward_tol, f, func_args))
    compute_trajectory(f, func_args, np.array([centre_point, upd_point]),
                       'large')
