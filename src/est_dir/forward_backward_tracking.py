import numpy as np


def compute_forward(t, const_forward, forward_tol, track, centre_point,
                    direction, f, func_args):
    """
    Repeatedly multiply step size t by const_forward (where const_forward > 1)
    until either forward_tol is met or until the function value cannot be
    improved any further.

    Parameters
    ----------
    t : float
        Initial guess of step size.
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
    track : 2-D array
            Array containing the step sizes attempted along with the
            corresponding repsonse function value.
    centre_point : 1-D array
                   Apply local search to centre_point.
    direction : 1-D array
                Search direction used for local search.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    track : 2-D array
            Updated array containing the step sizes attempted along with the
            corresponding response function value.
    count_func_evals : integer
                       Total number of response function evaluations.
    flag : boolean
           If forward_tol has been met, flag=True. Otherwise, if forward_tol
           has not been met, flag=False.
    """

    count_func_evals = 0
    while track[-2][1] > track[-1][1]:
        t = t * const_forward
        if t > forward_tol:
            return track, count_func_evals, False
        track = np.vstack((track,
                           np.array([t, f(np.copy(centre_point) -
                                          t * direction, *func_args)])))
        count_func_evals += 1
    return track, count_func_evals, True


def forward_tracking(centre_point, t, f_old, f_new, direction, const_forward,
                     forward_tol, f, func_args):
    """
    First part of forward_tracking() obtains a step size from
    compute_forward(). Second part of forward_tracking() checks whether flag
    is False. That is, if the forward_tol is met within compute_forward().
    If flag is False, outputs are returned. Otherwise, if flag is True, it is
    checked whether the response fuction can be improved further by applying
    the two-in-a-row rule. If the response function can be improved, the last
    entry in track is replaced (i.e. the step size and corresponding larger
    response function value than previous iteration)
    and replace with [t, f(centre_point - t * direction, *func_args)]. Then
    compute_forward() is applied again.
    If the response function cannot be improved with the two-in-a-row rule,
    outputs are returned.

    Parameters
    ----------
    centre_point : 1-D array
                   Apply local search to centre_point.
    t : float
        Initial guess of step size.
    f_old : float
            Function value at f(centre_point, *func_args).
    f_new : float
            Function value at f(centre_point - t * direction, *func_args).
    direction : 1-D array
                Search direction used for local search.
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
    track : 2-D array
            Updated array containing the step sizes attempted along with the
            corresponding response function value.
    count_func_evals : integer
                       Total number of response function evaluations.
    flag : boolean
           If forward_tol has been met, flag=True. Otherwise, if forward_tol
           has not been met, flag=False.
    """

    assert(const_forward > 1)
    track = np.array([[0, f_old], [t, f_new]])
    t = t * const_forward
    track = np.vstack((track,
                       np.array([t, f(np.copy(centre_point) -
                                      t * direction, *func_args)])))
    total_func_evals = 1
    track, count_func_evals, flag = (compute_forward
                                     (t, const_forward, forward_tol,
                                      track, centre_point, direction,
                                      f, func_args))
    total_func_evals += count_func_evals
    if flag == False:
        return track, total_func_evals, flag
    while flag:
        t = np.copy(track[-1][0]) * const_forward
        f_new = f(np.copy(centre_point) - t * direction, *func_args)
        total_func_evals += 1
        if f_new < track[-2][1]:
            track[-1] = np.array([t, f_new])
            track, count_func_evals, flag = compute_forward(t, const_forward,
                                                            forward_tol, track,
                                                            centre_point,
                                                            direction,
                                                            f, func_args)
            total_func_evals += count_func_evals
        else:
            return track, total_func_evals, flag
    return track, total_func_evals, flag


def compute_backward(t, const_back, back_tol, track, centre_point, direction,
                     f, func_args):
    """
    Decreases step size by a multiple of const_back (less than one) until
    either back_tol is met or until the response function cannot be improved
    any further.

    Parameters
    ----------
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
    track : 2-D array
            Array containing the step sizes attempted along with the
            corresponding repsonse function value.
    centre_point : 1-D array
                   Apply local search to centre_point.
    direction : 1-D array
                Search direction used for local search.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    track : 2-D array
            Updated array containing the step sizes attempted along with the
            corresponding response function value.
    count_func_evals : integer
                       Total number of response function evaluations.
    flag : boolean
           If back_tol has been met, flag=True. Otherwise, if back_tol
           has not been met, flag=False.
    """
    count_func_evals = 0
    while track[-2][1] > track[-1][1]:
        t = t * const_back
        if t < back_tol:
            return track, count_func_evals, False
        track = np.vstack((track,
                           np.array([t, f(np.copy(centre_point) -
                                          t * direction, *func_args)])))
        count_func_evals += 1
    return track, count_func_evals, True


def backward_tracking(centre_point, t, f_old, f_new, direction, const_back,
                      back_tol, f, func_args):
    """
    Decreases step size until the response function value at some step size
    t is less than the response function value at the centre_point. The step
    size is decreased in order to find the best response function value
    possible. The two-in-a-row rule is used as the stopping criteria for the
    step size.

    Parameters
    ----------
    centre_point : 1-D array
                   Apply local search to centre_point.
    t : float
        Initial guess of step size.
    f_old : float
            Function value at f(centre_point, *func_args).
    f_new : float
            Function value at f(centre_point - t * direction, *func_args).
    direction : 1-D array
                Search direction used for local search.
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
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    track : 2-D array
            Updated array containing the step sizes attempted along with the
            corresponding response function value.
    count_func_evals : integer
                       Total number of response function evaluations.
    """
    assert(const_back < 1)
    total_func_evals = 0
    track = np.array([[0, f_old], [t, f_new]])
    temp_track = np.copy(track)
    while track[0][1] <= track[-1][1]:
        t = t * const_back
        if t < back_tol:
            return temp_track, total_func_evals
        else:
            track = np.vstack((track,
                               np.array([t, f(np.copy(centre_point) -
                                              t * direction, *func_args)])))
            total_func_evals += 1

    track, count_func_evals, flag = compute_backward(t, const_back, back_tol,
                                                     track, centre_point,
                                                     direction, f, func_args)
    total_func_evals += count_func_evals
    if flag == False:
        return track, total_func_evals
    while flag:
        t = np.copy(track[-1][0]) * const_back
        f_new = f(np.copy(centre_point) - t * direction, *func_args)
        total_func_evals += 1
        if f_new < track[-2][1]:
            track[-1] = np.array([t, f_new])
            (track,
             count_func_evals,
             flag) = compute_backward(t, const_back, back_tol, track,
                                      centre_point, direction, f, func_args)
            total_func_evals += count_func_evals
        else:
            return track, total_func_evals
    return track, total_func_evals


def compute_coeffs(track_y, track_t):
    """
    Minimizes the fitted quadratic model of the observed step sizes
    and corresponding response function values.

    Parameters:
    -----------
    track_y : 1-D array
              Array containing response function values at each step size.
    track_t : 1-D array
              Array containing the tested step sizes.

    Returns:
    --------
    coeffs : float
             The point in which the quadratic model is minimized.
    """

    design_matrix_step = np.vstack((np.repeat(track_y[0], len(track_y)),
                                    np.array(track_t),
                                    np.array(track_t) ** 2)).T
    coeffs = (np.linalg.inv(design_matrix_step.T @ design_matrix_step) @
              design_matrix_step.T @ track_y)
    assert((-coeffs[1]/(2 * coeffs[2]) >= 0))
    return -coeffs[1]/(2 * coeffs[2])


def arrange_track_y_t(track, track_method):
    """
    Dependent on whether forward or backward tracking has been applied,
    select three step sizes where the plot of the response function values
    against the three step sizes is a curve.

    Parameters
    ------------
    track : 2-D array
            Array containing the step sizes attempted along with the
            corresponding response function value.
    track_method : string
                   Either track_method = 'Backward' if backward tracking has
                   been applied, or track_method = 'Forward' is forward
                   tracking has been applied.
    Returns
    --------
    track_y : 1-D array
              Array containing the smallest response function values.
    track_t : 1-D array
              Array containing the step sizes corresponding to the smallest
              response function values.
    """

    track_y = track[:, 1]
    track_t = track[:, 0]
    if track_method == 'Backward':
        min_pos = np.argmin(track_y)
        prev_pos = min_pos - 1
        track_y = np.array([track_y[0], track_y[min_pos], track_y[prev_pos]])
        track_t = np.array([track_t[0], track_t[min_pos], track_t[prev_pos]])
        assert(track_t[0] < track_t[1] < track_t[2])
        assert(track_y[0] > track_y[1])
        assert(track_y[2] > track_y[1])
        return track_y, track_t
    else:
        min_pos = np.argmin(track_y)
        next_pos = (min_pos + 1)
        track_y = np.array([track_y[0], track_y[min_pos], track_y[next_pos]])
        track_t = np.array([track_t[0], track_t[min_pos], track_t[next_pos]])
        assert(track_t[0] < track_t[1] < track_t[2])
        assert(track_y[0] > track_y[1])
        assert(track_y[2] > track_y[1])
        return track_y, track_t


def check_func_val_coeffs(track, track_method, centre_point, direction, f,
                          func_args):
    """
    Determine whether the response function value is minimized when using a
    step size from within track or from compute_coeffs().

    Parameters
    ----------
    track : 2-D array
            Array containing the step sizes attempted along with the
            corresponding response function value.
    track_method : string
                   Either track_method = 'Backward' if backward tracking has
                   been applied, or track_method = 'Forward' is forward
                   tracking has been applied.
    centre_point : 1-D array
                   Apply local search to centre_point.
    direction : 1-D array
                Search direction used for local search.
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    --------
    opt_t : float
            Step size corresponding to the lowest response function value
            observed.
    upd_point : 1-D array
                Apply iteration of local search with opt_t and direction.
    func_val : float
               Function value at upd_point. That is, f(upd_point,
               *func_args).
    """
    track_y, track_t = arrange_track_y_t(track, track_method)
    opt_t = compute_coeffs(track_y, track_t)
    upd_point = np.copy(centre_point) - opt_t * direction
    func_val = f(upd_point, *func_args)
    if func_val > track_y[1]:
        opt_t = track_t[1]
        upd_point = np.copy(centre_point) - opt_t * direction
        func_val = track_y[1]
        return upd_point, func_val
    else:
        return upd_point, func_val


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


    """
    f_new = f(np.copy(centre_point) - t * direction, *func_args)
    total_func_evals = 1
    if f_old <= f_new:
        track_method = 'Backward'
        track, func_evals = backward_tracking(centre_point, t, f_old, f_new,
                                              direction, const_back, back_tol,
                                              f, func_args)
        total_func_evals += func_evals
        if len(track) == 2:
            return centre_point, f_old, total_func_evals
        else:
            upd_point, func_val = (check_func_val_coeffs
                                   (track, track_method,
                                    centre_point, direction, f,
                                    func_args))
            total_func_evals += 1
            return upd_point, func_val, total_func_evals

    elif f_old > f_new:
        track_method = 'Forward'
        track, func_evals, flag = (forward_tracking
                                   (centre_point, t, f_old, f_new, direction,
                                    const_forward, forward_tol, f, func_args))
        total_func_evals += func_evals
        if flag == False:
            t = track[-1][0]
            f_new = track[-1][1]
            return (np.copy(centre_point) - t * direction,
                    f_new, total_func_evals)
        else:
            upd_point, func_val = (check_func_val_coeffs
                                   (track, track_method,
                                    centre_point, direction, f,
                                    func_args))
            total_func_evals += 1
            return upd_point, func_val, total_func_evals
