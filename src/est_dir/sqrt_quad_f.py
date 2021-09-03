import numpy as np


def sqrt_quad_f(x, a, matrix):
    """
    Compute the square root of the quadratic function.

    Parameters
    ----------
    x : 1-D array
        Point in which the square root of the quadratic is to be evaluated.
    minimizer : 1-D array
                Minimizer of the square root of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    Returns
    -------
    func_val : float
               Square root of the quadratic function at x.
    """

    func_val = (np.sqrt((x - a).T @ matrix @ (x - a)))
    return func_val


def sqrt_quad_f_noise(x, a, matrix, mu, sd):
    """
    Compute square root of the quadratic function with noise.

    Parameters
    ----------
    x : 1-D array
        Point in which the square root of the quadratic is to be
        evaluated.
    minimizer : 1-D array
                Minimizer of the square root of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    mu : float
         Mean of the normal distribution, used to sample noise.
    sd : float
         Standard deviation of the normal distribution, used to sample noise.
    Returns
    -------
    noisy_func_val : float
                     Square root of the quadratic function at x with noise.
    """

    noisy_func_val = (sqrt_quad_f(x, a, matrix) + np.random.normal(mu, sd))
    return noisy_func_val
