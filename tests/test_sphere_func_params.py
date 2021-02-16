import numpy as np
from hypothesis import given, settings, strategies as st

import est_dir


@settings(max_examples=5, deadline=None)
@given(st.integers(2, 100), st.integers(1, 10), st.integers(16, 100))
def test_1(m, min_eig, max_eig):
    mat = est_dir.sphere_func_params(min_eig, max_eig, m)
    assert(mat.shape == (m, m))
    for j in range(m):
        for i in range(m):
            if i != j:
                assert(np.round(mat[j, i], 4) == np.round(mat[j, i], 4))


@settings(max_examples=5, deadline=None)
@given(st.integers(2, 500))
def test_2(m, ):
    min_eig = 1
    max_eig = 1
    A = est_dir.sphere_func_params(min_eig, max_eig, m)
    assert(np.all(np.round(A, 3) == np.identity(m)))
