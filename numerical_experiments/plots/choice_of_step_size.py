import matplotlib.pyplot as plt
import numpy as np

import est_dir


def combine_tracking(centre_point, f_old, direction, t, const_back, back_tol,
                     const_forward, forward_tol, f, func_args):
    """
    Apply either forward tracking if f(centre_point, *func_args) >
    f(centre_point - t * direction, *func_args) or apply backward tracking
    if f(centre_point, *func_args) <= f(centre_point - t * direction,
    *func_args).
    After applying forward or backward tracking, a quadratic model is fitted
    with the best observed response function values and corresponding step
    sizes. The minimizer of the quadratic model is computed, and the chosen
    step size to be applied for local search is either the minimizer of the
    fitted quadratic model, or the step size corresponding to the smallest
    response function value from applying forward or backward tracking.

    Parameters
    -----------
    centre_point : 1-D array
                   Apply local search to centre_point.
    f_old : float
            Function value at f(centre_point, *func_args).
    direction : 1-D array
                Search direction used for local search.
    t : float
        Initial guess of step size.
    const_back : float
                 If backward tracking is required, the initial guess of the
                 step size will be multiplied by const_back at each iteration
                 of backward tracking. That is,
                 t <- t * const_back
                 It should be noted that const_back < 1.
    back_tol : float
               It must be ensured that the step size computed by backward
               tracking is not smaller than back_tol. If this is the case,
               iterations of backward tracking are terminated. Typically,
               back_tol is a very small number.
    const_forward : float
                    The initial guess of the
                    step size will be multiplied by const_forward at each
                    iteration of forward tracking. That is,
                    t <- t * const_back
                    It should be noted that const_forward > 1.
    forward_tol : float
                  It must be ensured that the step size computed by forward
                  tracking is not greater than forward_tol. If this is the
                  case, iterations of forward tracking are terminated.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    opt_t : float
            Step size corresponding to the lowest response function value
            observed.
    upd_point : 1-D array
                Apply iteration of local search with opt_t and direction.
    func_val : float
               Function value at upd_point. That is, f(upd-point,
               *func_args).
    total_func_evals : integer
                       Total number of function evalautions used to compute
                       step size.
    track : 2-D array
            Contains step sizes in the first column and corresponding noisy
            function values in the second column.
    """
    f_new = f(np.copy(centre_point) - t * direction, *func_args)
    total_func_evals = 1
    if f_old <= f_new:
        track_method = 'Backward'
        track, func_evals = (est_dir.backward_tracking(
                             centre_point, t, f_old, f_new,
                             direction, const_back, back_tol,
                             f, func_args))
        total_func_evals += func_evals
        if len(track) == 2:
            return centre_point, f_old, total_func_evals
        else:
            upd_point, func_val = (est_dir.check_func_val_coeffs
                                   (track, track_method,
                                    centre_point, direction, f,
                                    func_args))
            total_func_evals += 1
            return upd_point, func_val, total_func_evals, track

    elif f_old > f_new:
        track_method = 'Forward'
        track, func_evals, flag = (est_dir.forward_tracking
                                   (centre_point, t, f_old, f_new, direction,
                                    const_forward, forward_tol, f, func_args))
        total_func_evals += func_evals
        if flag == False:
            t = track[-1][0]
            f_new = track[-1][1]
            return (np.copy(centre_point) - t * direction,
                    f_new, total_func_evals)
        else:
            upd_point, func_val = (est_dir.check_func_val_coeffs
                                   (track, track_method,
                                    centre_point, direction, f,
                                    func_args))
            total_func_evals += 1
            return upd_point, func_val, total_func_evals, track


def compute_trajectory(f, func_args, points, upd_point):
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
    upd_point : 1D array
                Resulting point from finding suitable step size.
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

    plt.annotate(r'$\gamma = 0$', (0, 0), (-2.2, -2.25), size=15)
    plt.annotate(r'$\gamma = 1$', (0, 0), (-1.3, -1.6), size=15)
    plt.annotate(r'$\gamma = 2$', (0, 0), (-0.45, -0.9), size=15)
    plt.annotate(r'$\gamma = 4$', (0, 0), (1.1, 0.5), size=15)
    plt.annotate(r'$\gamma^*$', (0, 0), (0, 0.3), size=25)
    plt.scatter(points[:, 0], points[:, 1], color='black')
    plt.plot(points[:, 0], points[:, 1], color='black')
    plt.scatter(upd_point[0], upd_point[1], color='blue', s=100, marker='o')
    plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.25)
    plt.savefig('step_length_ex_contour.png')


def min_f_noise(step, x, direction, f, minimizer, matrix, mu, sd):
    """
    Minimize a noisy function with respect to step size.

    Parameters
    -----------
    step : float
           Step size
    x : 1-D array
        Compute step length for x in order to compute,
        x <- x - step * direction.
    direction : 1-D array
                Search direction
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    minimizer : 1-D array
                Minimizer of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    mu : float
         Mean of the normal distribution, used to sample noise.
    sd : float
         Standard deviation of the normal distribution, used to sample noise.

    Returns
    --------
    func_val : float
               Noisy function value at corresponding step size.

    """
    return f(x - step * direction, minimizer, matrix, mu, sd)


def plot_parabolic_interpolation(track):
    """
    Plot parabolic interpolation with three best points in order
    to find an optimal step size.

    Parameters
    -----------
    track : 2-D array
            Array containing the step sizes attempted along with the
            corresponding response function value.
    """
    plt.clf()
    temp_track = np.array([[track[0, 0], track[0, 1]],
                           [track[2, 0], track[2, 1]],
                           [track[3, 0], track[3, 1]]])
    coeffs = np.polyfit(temp_track[:, 0], temp_track[:, 1], deg=2)
    minimum = -coeffs[1] / (2 * coeffs[0])
    p2 = np.poly1d(coeffs)
    xp = np.linspace(temp_track[0, 0] - 1, temp_track[2, 0] + 1, 100)
    plt.plot(xp, p2(xp), color='black')
    y_noise = np.zeros((100))
    y_noise = np.zeros((100))
    min_f_args_noise = centre_point, direction, f, minimizer, matrix, mu, sd
    for j in range(100):
        y_noise[j] = min_f_noise(xp[j], *min_f_args_noise)
    plt.plot(xp, y_noise, color='green')
    plt.scatter(minimum, p2(minimum), s=100, color='blue', marker='o')
    plt.scatter(temp_track[0, 0], p2(temp_track[0, 0]), s=50, color='black')
    plt.scatter(temp_track[1, 0], p2(temp_track[1, 0]), s=50, color='black')
    plt.scatter(temp_track[2, 0], p2(temp_track[2, 0]), s=50, color='black')
    plt.xlim(-1, 5)
    plt.xlabel(r'$\gamma$', size='20')
    plt.ylabel(r'$y(\gamma)$', size='20')
    plt.savefig('step_length_parabolic_interpolation_plot.png')


if __name__ == "__main__":
    m = 2
    f = est_dir.quad_f_noise
    minimizer = np.zeros((m))
    matrix = np.array([[1, 0],
                       [0, 1]])
    mu = 0
    sd = 0.25
    func_args = (minimizer, matrix, mu, sd)

    centre_point = np.array([-1.8, -1.8])
    direction = np.array([-0.8, -0.7])

    np.random.seed(40)
    init_func_val = f(centre_point, *func_args)
    step = 1
    const_back = 0.5
    const_forward = 2
    forward_tol = 10000000
    back_tol = 0.0000000001

    (upd_point,
     f_new,
     total_func_evals_step,
     track) = (combine_tracking
               (centre_point, init_func_val,
                direction, step, const_back,
                back_tol, const_forward,
                forward_tol, f, func_args))
    points = np.zeros((len(track), 2))
    for j in range(len(track)):
        points[j] = centre_point - track[j, 0] * direction

    compute_trajectory(f, func_args, points, upd_point)

    plot_parabolic_interpolation(track)
