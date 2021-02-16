import numpy as np

import est_dir


def test_1():
    """
    Test for compute_forward - check that when flag=True, track is updated.
    """
    np.random.seed(90)
    n = 16
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 20, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 0.001
    forward_tol = 1000000
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                func_args, option)
    assert(func_evals == 16)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    track = np.array([[0, f_old], [step, f_new]])
    track, count_func_evals, flag = (est_dir.compute_forward
                                     (step, const_forward, forward_tol, track,
                                      centre_point, beta,
                                      f, func_args))
    assert(f_old > f_new)
    assert(count_func_evals == len(track) - 2)
    assert(flag==True)
    assert(track[0][0] == 0)
    for j in range(1, len(track)):
        assert(track[j][0] == step)
        step = step * const_forward
        if j < len(track) - 1:
            assert(track[j][1] < track[j-1][1])
        else:
            assert(track[j][1] > track[j-1][1])


def test_2():
    """
    Test for compute_forward - check that when flag=False, original track is returned.
    """
    np.random.seed(90)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([20, 20])
    matrix = np.identity(m)
    func_args = (minimizer, matrix, 0, 0.0000001)
    step = 1
    forward_tol = 100000
    beta = np.array([0.0001, 0.0001])
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    track = np.array([[0, f_old], [step, f_new]])
    test_track, count_func_evals, flag = (est_dir.compute_forward
                                          (step, const_forward, forward_tol, track, 
                                           centre_point, beta, f, func_args))
    assert(f_old > f_new)
    assert(np.all(test_track == track))
    assert(flag == False)
    assert(count_func_evals > 0)


def test_3():
    """Test for forward_tracking - flag=True and f_new >= track[-2][1]"""
    np.random.seed(90)
    n = 16
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 20, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 0.05
    forward_tol = 1000000
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                 func_args, option)
    assert(func_evals == 16)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    assert(f_old > f_new)
    track, total_func_evals, flag = (est_dir.forward_tracking
                                    (centre_point, step, f_old, f_new, beta, 
                                        const_forward, forward_tol, f, func_args))
    assert(len(track) - 1 ==  total_func_evals)
    assert(np.round(track[0][0], 3) == np.round(0, 3))
    assert(np.round(track[1][0], 3) == np.round(step, 3))
    assert(flag == True)
    for j in range(2, len(track)):
        step = step * 2
        assert(np.round(track[j][0], 3) == step)
        if j ==  (len(track) - 1):
            assert(track[j][1] > track[j-1][1])
        else:
            assert(track[j-1][1] > track[j][1])


def test_4():
    """Test for forward_tracking"""
    np.random.seed(90)
    n = 20
    m = 10
    f = est_dir.sphere_f_noise
    option = 'XY'
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 20, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 0.1
    forward_tol = 1000000
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                 func_args, option)
    assert(func_evals == n)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    track, total_func_evals, flag = (est_dir.forward_tracking
                            (centre_point, step, f_old, f_new, beta,
                                const_forward, forward_tol, f, func_args))
    assert(np.round(track[0][0], 3) == np.round(0, 3))
    assert(np.round(track[1][0], 3) == np.round(step, 3))
    assert(flag == True)
    assert(total_func_evals > 0)
    for j in range(2, len(track)):
        step = step * 2
        assert(np.round(track[j][0], 3) == step)
        if j ==  (len(track) - 1):
            assert(track[j][1] > track[j-1][1])


def test_5():
    """
    Test for forward_tracking - forward_tol not met and f_new < track[-2][1].
    """
    np.random.seed(25)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([25,25])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 10)
    t = 0.005
    forward_tol = 10000
    beta = np.array([1, 1])
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - t * beta, *func_args)
    assert(f_old > f_new)
    track, total_func_evals, flag = (est_dir.forward_tracking
                                    (centre_point, t, f_old, f_new, beta,
                                    const_forward, forward_tol, f, func_args))
    assert(np.round(track[0][0], 3) == np.round(0, 3))
    assert(np.round(track[1][0], 3) == np.round(t, 3))
    assert(len(track)-1 == total_func_evals)
    assert(flag == True)
    for j in range(1, len(track)):
        if j ==  (len(track) - 1):
            assert(track[j][1] > track[j-1][1]) 
        else:
            assert(track[j-1][1] > track[j][1]) 


def test_6():
    """Test for forward_tracking - forward_tol met"""
    np.random.seed(90)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([20, 20])
    matrix = np.identity(m)
    func_args = (minimizer, matrix, 0, 0.0000001)
    step = 0.5
    forward_tol = 1.5
    beta = np.array([0.0001, 0.0001])
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    track, total_func_evals, flag = (est_dir.forward_tracking
                                     (centre_point, step, f_old, f_new, beta,
                                      const_forward, forward_tol, f, func_args))
    assert(flag == False)
    assert(track[2][1] < track[1][1] < track[0][1])
    assert(total_func_evals == 1)        


def test_7():
    """Test for backward_tracking - back_tol is met"""
    np.random.seed(32964)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    minimizer = np.ones((m,))
    centre_point = np.array([25,25])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    t = 1
    back_tol = 1
    beta = np.array([200,200])
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - t * beta, *func_args)
    assert(f_old< f_new)
    track, count_func_evals = (est_dir.backward_tracking
                               (centre_point, t, f_old, f_new, beta,
                                const_back, back_tol, f, func_args))
    assert(track.shape == (2, m))
    assert(track[0][0] == 0)
    assert(track[1][0] == t)
    assert(track[1][0] < track[1][1])
    assert(count_func_evals == 0)


def test_8():
    """Test for backward_tracking - back tol is not met"""
    np.random.seed(32964)
    n = 6
    m = 2
    f = est_dir.sphere_f_noise
    option = 'XY'
    const_back = 0.5
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0,10,(m,))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    t = 1
    back_tol = 0.000000001
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                 func_args, option)
    assert(func_evals == n)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - t * beta, *func_args)
    track, total_func_evals = (est_dir.backward_tracking
                               (centre_point, t, f_old, f_new, beta, 
                                const_back, back_tol, f, func_args))
    assert(np.round(track[0][0], 3) == np.round(0, 3))
    assert(np.round(track[1][0], 3) == np.round(t, 3))
    assert(len(track) - 2) == total_func_evals
    for j in range(1, len(track)):
        assert(np.round(track[j][0], 4) == np.round(t, 4))
        if j ==  (len(track) - 1):
            assert(track[j][1] < track[j-1][1])
            assert(track[j][1] < track[0][1])
        else:
            assert(track[0][1] < track[j][1])
        t = t / 2


def test_9():
    """Test for compute_coeffs"""
    track_y = np.array([100, 200, 50])
    track_t = np.array([0, 1, 0.5])
    design_matrix_step = np.vstack((np.repeat(track_y[0], len(track_t)),
                                    np.array(track_t),
                                    np.array(track_t) ** 2)).T
    assert(np.all(design_matrix_step[0, :] == np.array([100, 0, 0])))
    assert(np.all(design_matrix_step[1, :] == np.array([100, 1, 1])))
    assert(np.all(design_matrix_step[2, :] == np.array([100, 0.5, 0.25])))
    OLS = (np.linalg.inv(design_matrix_step.T @ design_matrix_step) @ 
        design_matrix_step.T @ track_y)
    check = -OLS[1] / (2 * OLS[2])
    opt_t = est_dir.compute_coeffs(track_y, track_t)
    assert(np.all(np.round(check, 5) == np.round(opt_t, 5)))


def test_10():
    """
    Test for combine_tracking - check that correct step size is returned when forward_tol is met.
    """
    np.random.seed(90)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([20, 20])
    matrix = np.identity(m)
    func_args = (minimizer, matrix, 0, 0.0000001)
    step = 1
    forward_tol = 100000
    back_tol = 0.0000001
    beta = np.array([0.0001, 0.0001])
    f_old = f(np.copy(centre_point), *func_args)
    opt_t, upd_point, func_val, total_func_evals = (est_dir.combine_tracking
                                                    (centre_point, f_old, 
                                                    beta, step, const_back, 
                                                    back_tol, const_forward, 
                                                    forward_tol, f, func_args))
    assert(opt_t == 2)
    assert(upd_point.shape == (m, ))
    assert(type(total_func_evals) is int)
    assert(func_val < f_old)


def test_11():
    """
    Test for combine_tracking - check that correct step size is returned, when forward_tol is not met.
    """
    np.random.seed(3291)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([25,25])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 0.005
    forward_tol = 10000
    back_tol=0.0000001
    beta = np.array([1, 1])
    f_old = f(np.copy(centre_point), *func_args)
    opt_t, upd_point, func_val, total_func_evals = (est_dir.combine_tracking
                                                    (centre_point, f_old, 
                                                    beta, step, const_back, 
                                                    back_tol, const_forward, 
                                                    forward_tol, f, func_args))    
    assert(opt_t > 0)
    assert(upd_point.shape == (m, ))
    assert(type(total_func_evals) is int)
    assert(func_val < f_old)


def test_12():
    """
    Test for combine_tracking - check that correct step size is returned, when back_tol is met.
    """
    np.random.seed(32964)
    m = 2
    f = est_dir.sphere_f_noise
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.array([25,25])
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 1
    back_tol = 1
    forward_tol = 100000
    beta = np.array([200,200])
    f_old = f(np.copy(centre_point), *func_args)
    opt_t, upd_point, func_val, total_func_evals = (est_dir.combine_tracking
                                                    (centre_point, f_old, 
                                                    beta, step, const_back, 
                                                    back_tol, const_forward, 
                                                    forward_tol, f, func_args))    
    assert(opt_t == 0)
    assert(upd_point.shape == (m, ))
    assert(type(total_func_evals) is int)
    assert(func_val == f_old)


def test_13():
    """
    Test for combine_tracking - check that correct step size is returned, when back_tol is not met.
    """
    np.random.seed(32964)
    n = 6
    m = 2
    f = est_dir.sphere_f_noise
    option = 'XY'
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0,10,(m,))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 1
    forward_tol = 1000000
    back_tol = 0.000000001
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                 func_args, option)
    assert(func_evals == n)
    f_old = f(np.copy(centre_point), *func_args)
    opt_t, upd_point, func_val, total_func_evals = (est_dir.combine_tracking
                                                    (centre_point, f_old, 
                                                    beta, step, const_back, 
                                                    back_tol, const_forward, 
                                                    forward_tol, f, func_args))    
    assert(opt_t > 0)
    assert(upd_point.shape == (m, ))
    assert(type(total_func_evals) is int)
    assert(func_val < f_old)


def test_14():
    """Test for arrange_track_y_t"""  
    track = np.array([[0, 100],
                  [1, 80],
                  [2, 160],
                  [4, 40],
                  [8, 20],
                  [16, 90]])
    track_method = 'Forward'
    track_y, track_t = est_dir.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([40, 20, 90])))
    assert(np.all(track_t == np.array([4, 8, 16])))


def test_15():
    """Test for arrange_track_y_t"""  
    track = np.array([[0, 100],
                      [1, 80],
                      [2, 70],
                      [4, 90]])
    track_method = 'Forward'
    track_y, track_t = est_dir.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([80, 70, 90])))
    assert(np.all(track_t == np.array([1, 2, 4])))


def test_16():
    """Test for arrange_track_y_t"""  
    track = np.array([[0, 100],
                      [1, 120],
                      [0.5, 110],
                      [0.25, 90]])
    track_method = 'Backward'
    track_y, track_t = est_dir.arrange_track_y_t(track, track_method)
    assert(np.all(track_y == np.array([100, 90, 110])))
    assert(np.all(track_t == np.array([0, 0.25, 0.5])))


def test_17():
    """Test for check_func_val_coeffs when func_val > track_y[1]."""  
    np.random.seed(90)
    n = 20
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 20, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 60)
    step = 0.01
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                 func_args, option)
    assert(func_evals == 16)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    assert(f_old > f_new)
    track = np.array([[   0       , 100],
                      [   1      , 160],
                      [   2      , 40],
                      [   4      , 90]])
    track_method = 'Forward'
    opt_t, upd_point, func_val = (est_dir.check_func_val_coeffs
                                  (track, track_method, centre_point, beta, f, 
                                   func_args))
    assert(opt_t == 2)
    assert(upd_point.shape == (m, ))
    assert(func_val == 40)


def test_18():
    """Test for check_func_val_coeffs when func_val <= track_y[1]."""
    np.random.seed(91)
    n = 20
    m = 10
    f = est_dir.sphere_f_noise
    option = 'LS'
    const_back = 0.5
    const_forward = (1 / const_back)
    minimizer = np.ones((m,))
    centre_point = np.random.uniform(0, 20, (m, ))
    matrix = est_dir.sphere_func_params(1, 10, m)
    func_args = (minimizer, matrix, 0, 5)
    step = 0.01
    forward_tol = 1000000
    beta, func_evals = est_dir.compute_direction(n, m, centre_point, f,
                                                func_args, option)
    assert(func_evals == 16)
    f_old = f(np.copy(centre_point), *func_args)
    f_new = f(np.copy(centre_point) - step * beta, *func_args)
    assert(f_old > f_new)
    track, total_func_evals, flag = (est_dir.forward_tracking
                                    (centre_point, step, f_old, f_new, beta, 
                                    const_forward, forward_tol, f, func_args))
    assert(flag == True)
    assert(total_func_evals > 0)
    track_method = 'Forward'
    opt_t, upd_point, func_val = (est_dir.check_func_val_coeffs
                                (track, track_method, centre_point, beta, f, 
                                func_args))
    assert(opt_t > 0)
    assert(upd_point.shape == (m, ))
    assert(np.all(func_val <= track[:, 1]))
