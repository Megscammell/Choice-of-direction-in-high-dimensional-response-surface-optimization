import numpy as np


def square_quad_f(x, a, matrix):
    """
    Compute the square of the quadratic function.

    Parameters
    ----------
    x : 1-D array
        Point in which the square of the quadratic is to be evaluated.
    minimizer : 1-D array
                Minimizer of the square of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    Returns
    -------
    func_val : float
               Square of the quadratic function at x.
    """

    func_val = ((x - a).T @ matrix @ (x - a)) ** 2
    return func_val


def square_quad_f_noise(x, a, matrix, mu, sd):
    """
    Compute square of the quadratic function with noise.

    Parameters
    ----------
    x : 1-D array
        Point in which the square of the quadratic is to be
        evaluated.
    minimizer : 1-D array
                Minimizer of the square of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    mu : float
         Mean of the normal distribution, used to sample noise.
    sd : float
         Standard deviation of the normal distribution, used to sample noise.
    Returns
    -------
    noisy_func_val : float
                     Square of the quadratic function at x with noise.
    """

    noisy_func_val = (square_quad_f(x, a, matrix) + np.random.normal(mu, sd))
    return noisy_func_val
