import numpy as np
from hypothesis import given, settings, strategies as st

import est_dir


# def test_1():
#     x = np.array([2, 2])
#     t = np.array([7.5, 9])
#     A = np.array([[1, 0],
#                   [0, 4]])
#     func_val = est_dir.sqrt_sphere_f(x, t, A)
#     assert(func_val == 33.5)


# @settings(max_examples=5, deadline=None)
# @given(st.integers(2, 500))
# def test_2(m):
#     x = np.random.uniform(0, 10, (m, ))
#     t = np.random.uniform(0, 10, (m, ))
#     min_eig = 1
#     max_eig = 10
#     A = est_dir.sphere_func_params(min_eig, max_eig, m)
#     mu = 0
#     sd = 1
#     func_val = est_dir.sqrt_sphere_f_noise(x, t, A, mu, sd)
#     assert(isinstance(func_val, float))

def test_1():
    x = np.array([2, 2])
    t = np.array([7.5, 9])
    A = np.array([[1, 0],
                  [0, 4]])
    func_val = est_dir.sqrt_sphere_f(x, t, A)
    assert(np.round(func_val, 4) == 15.0416)


@settings(max_examples=5, deadline=None)
@given(st.integers(2, 500))
def test_2(m):
    x = np.random.uniform(0, 10, (m, ))
    t = np.random.uniform(0, 10, (m, ))
    min_eig = 1
    max_eig = 10
    A = est_dir.sphere_func_params(min_eig, max_eig, m)
    mu = 0
    sd = 1
    func_val = est_dir.sqrt_sphere_f_noise(x, t, A, mu, sd)
    assert(isinstance(func_val, float))