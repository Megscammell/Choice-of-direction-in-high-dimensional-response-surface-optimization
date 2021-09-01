import numpy as np


def compute_forward(t, const_forward, forward_tol, track, centre_point, beta, 
                    f, func_args):
    """
    Increases step size by a multiple of const_forward (greater than one) until
    either forward_tol is met or until the response function cannot be improved
    any further.
    """
    count_func_evals = 0
    while track[-2][1] > track[-1][1]:
        t = t * const_forward
        if t > forward_tol:
            return track, count_func_evals, False
        track = np.vstack((track, np.array([t, f(np.copy(centre_point) - t * beta, *func_args)])))
        count_func_evals += 1
    return track, count_func_evals, True


def forward_tracking(centre_point, t, f_old, f_new, beta, const_forward,
                     forward_tol, f, func_args):
    """
    First part of forward_tracking() obtains a step size from compute_forward().
    Second part of forward_tracking() checks whether flag is False. That is, if
    the forward_tol is met in compute_forward(). If flag is False, outputs are 
    returned from forward_tracking(). Otherwise, if flag is True, it is checked
    whether the response fuction can be improved further by applying the two-in-a-row rule.
    If the response function can be improved, we replace the last entry in track which
    did not improve the response function value and replace with [t, f(centre_point
    - t * beta, *func_args)] and pass to compute_forward(). Otherwise if the response
    function cannot be improved with the two-in-a-row rule, outputs from forward_tracking()
    are returned.
    """
    assert(const_forward > 1)
    track = np.array([[0, f_old], [t, f_new]])    
    t = t * const_forward
    track = np.vstack((track, np.array([t, f(np.copy(centre_point) - t * beta, *func_args)])))
    total_func_evals = 1
    track, count_func_evals, flag = (compute_forward
                                     (t, const_forward, forward_tol,   
                                      track, centre_point, beta, f, func_args))
    total_func_evals += count_func_evals
    if flag == False:
        return track, total_func_evals, flag
    while flag:
        t = np.copy(track[-1][0]) * const_forward
        f_new = f(np.copy(centre_point) - t * beta, *func_args)
        total_func_evals += 1
        if f_new < track[-2][1]:
            track[-1] = np.array([t, f_new])
            track, count_func_evals, flag = compute_forward(t, const_forward, 
                                                            forward_tol, track,
                                                            centre_point, beta,
                                                            f, func_args)
            total_func_evals += count_func_evals
        else:
            return track, total_func_evals, flag
    return track, total_func_evals, flag


def compute_backward(t, const_back, back_tol, track, centre_point, beta, 
                     f, func_args):
    """
    Decreases step size by a multiple of const_back (less than one) until
    either back_tol is met or until the response function cannot be improved
    any further.
    """
    count_func_evals = 0
    while track[-2][1] > track[-1][1]:
        t = t * const_back
        if t < back_tol:
            return track, count_func_evals, False 
        track = np.vstack((track, np.array([t, f(np.copy(centre_point) - t * beta, *func_args)])))
        count_func_evals  += 1
    return track, count_func_evals, True


def backward_tracking(centre_point, t, f_old, f_new, beta, const_back, 
                      back_tol, f, func_args):
    """
    Decreases step size until the response function value at some step size t is less than
    the response function value at the centre_point. The step size is decreased in order to 
    find the best response function value possible. The two-in-a-row rule is used as the 
    stopping criteria for the step size.
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
            track = np.vstack((track, np.array([t, f(np.copy(centre_point) - t * beta, *func_args)])))
            total_func_evals += 1
    
    track, count_func_evals, flag = compute_backward(t, const_back, back_tol, track, centre_point, beta, 
                                                     f, func_args)
    total_func_evals += count_func_evals
    if flag == False:
        return track, total_func_evals
    while flag:
        t = np.copy(track[-1][0]) * const_back
        f_new = f(np.copy(centre_point) - t * beta, *func_args)
        total_func_evals += 1
        if f_new < track[-2][1]:
            track[-1] = np.array([t, f_new])
            track, count_func_evals, flag = compute_backward(t, const_back, back_tol, track, centre_point, beta, 
                                                             f, func_args)
            total_func_evals += count_func_evals             
        else:
            return track, total_func_evals
    return track, total_func_evals


def compute_coeffs(track_y, track_t):
    """
    Sets up design marix and performs least squares to obtain step size.
    """
    design_matrix_step = np.vstack((np.repeat(track_y[0], len(track_y))
                                    ,np.array(track_t),
                                    np.array(track_t) ** 2)).T
    coeffs = (np.linalg.inv(design_matrix_step.T @ design_matrix_step) @ 
              design_matrix_step.T @ track_y)   
    assert((-coeffs[1]/(2 * coeffs[2]) >= 0))
    return -coeffs[1]/(2 * coeffs[2])
    # coeffs = np.polyfit(track_t, track_y, 2)
    # assert((-coeffs[1]/(2 * coeffs[0]) >= 0))
    # return -coeffs[1]/(2 * coeffs[0])


def arrange_track_y_t(track, track_method):
    """
    Dependent on track_method, select three step sizes where the plot of the response function values against the
    three step sizes is a curve. Use selected step sizes and corresponding response function values to construct
    the design matrix in compute_coeffs().
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


def check_func_val_coeffs(track, track_method, centre_point, beta, f, 
                          func_args):
    """
    Compute the step size opt_t using compute_coeffs() and check that
    f(centrer_point - opt_t *beta, *func_args) is less than the smallest
    response function value found so far. If this is not the case, the
    step size corresponding the the best found response function value is
    returned. 
    """    
    track_y, track_t = arrange_track_y_t(track, track_method)
    opt_t = compute_coeffs(track_y, track_t)
    upd_point = np.copy(centre_point) - opt_t * beta
    func_val = f(upd_point, *func_args)
    if func_val > track_y[1]:
        opt_t = track_t[1]
        upd_point = np.copy(centre_point) - opt_t * beta
        func_val = track_y[1]
        return opt_t, upd_point, func_val
    else:
        return opt_t, upd_point, func_val


def combine_tracking(centre_point, f_old, beta, t, const_back, back_tol,
                     const_forward, forward_tol, f, func_args):
    """
    Compare f_new and f_old to determine whether backward or forward tracking
    is required. For backward tracking, if back_tol is met then a step size of
    0 is returned. Otherwise, check_func_val_coeffs() is called and outputs
    are returned. For forward tracking, if forward_tol is met, the step size 
    corresponding to the best response function value is returned. Otherwise, 
    check_func_val_coeffs() is called and outputs are returned.
    """    
    f_new = f(np.copy(centre_point) - t * beta, *func_args)
    total_func_evals = 1
    if f_old <= f_new:
        track_method = 'Backward'
        track, func_evals = backward_tracking(centre_point, t, f_old, f_new, 
                                              beta, const_back, back_tol, f, 
                                              func_args)
        total_func_evals += func_evals
        if len(track) == 2:
            return 0, centre_point, f_old, total_func_evals
        else:
            opt_t, upd_point, func_val = (check_func_val_coeffs
                                          (track, track_method, 
                                           centre_point, beta, f, 
                                           func_args))
            total_func_evals += 1                                          
            return opt_t, upd_point, func_val, total_func_evals

    elif f_old > f_new:
        track_method = 'Forward'
        track, func_evals, flag = (forward_tracking
                                   (centre_point, t, f_old, f_new, beta, 
                                    const_forward, forward_tol, f, func_args))
        total_func_evals += func_evals
        if flag == False:
            t = track[-1][0]
            f_new = track[-1][1]
            return t, np.copy(centre_point) - t * beta, f_new, total_func_evals
        else:
            opt_t, upd_point, func_val = (check_func_val_coeffs
                                          (track, track_method, 
                                           centre_point, beta, f, 
                                           func_args))
            total_func_evals += 1                                           
            return opt_t, upd_point, func_val, total_func_evals
