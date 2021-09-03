import numpy as np


def quad_f(x, minimizer, matrix):
    """
    Compute quadratic function evaluation.

    Parameters
    ----------
    x : 1-D array
        Point in which the quadratic is to be evaluated.
    minimizer : 1-D array
                Minimizer of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    Returns
    -------
    func_val : float
               Quadratic function value at x.
    """

    func_val = ((x - minimizer).T @ matrix @ (x - minimizer))
    return func_val


def quad_f_noise(x, minimizer, matrix, mu, sd):
    """
    Compute the quadratic function with noise.

    Parameters
    ----------
    x : 1-D array
        Point in which the quadratic function is to be evaluated.
    minimizer : 1-D array
                Minimizer of the quadratic function.
    matrix : 2-D array
             Positive definite matrix.
    mu : float
         Mean of the normal distribution, used to sample noise.
    sd : float
         Standard deviation of the normal distribution, used to sample noise.
    Returns
    -------
    noisy_func_val : float
                     Quadratic function value at x with noise.
    """

    noisy_func_val = (quad_f(x, minimizer, matrix) + np.random.normal(mu, sd))
    return noisy_func_val
