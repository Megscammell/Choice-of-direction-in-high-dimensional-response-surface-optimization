import numpy as np
from scipy.stats import ortho_group


def quad_func_params(lambda_1, lambda_2, m):
    """
    Create function arguments for the quadratic function.

    Parameters
    ----------
    lambda_1 : integer
               Smallest eigenvalue of diagonal matrix.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.
    m : integer
        Number of variables.
    Returns
    ----------
    matrix_test : 2-D array
                  Positive definite matrix.
    """

    if lambda_1 != lambda_2:
        diag_vals = np.zeros(m)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 0.1,
                                          lambda_2 - 0.1, (m - 2))
        A = np.diag(diag_vals)
        rotation = ortho_group.rvs(dim=m)
        matrix = rotation.T @ A @ rotation
        return matrix
    else:
        matrix = np.identity(m)
        return matrix
