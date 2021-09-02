import numpy as np


def compute_y(centre_point, design, positions, n, m, f, func_args, region):
    """
    Compute response function value at each observation of the design matrix
    centred at centre_point.

    Parameters
    ----------
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    design : 2-D array
             The design matrix can either be a fractional factorial design or
             contain entries chosen randomly as +1 or -1, with the condition
             that each column of the design matrix has the same number of
             +1's or -1's.
    positions : 1-D array
                If the number of columns of the design matrix is less than m,
                a subset of variables of centre_point can be chosen to centre
                the design matrix. Therefore, the positions of the variables
                of centre_point are provided.
    n : integer
        Number of observations of the design matrix (rows).
    m : integer
        Number of variables of the design matrix (columns).
    f : function
        response function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the response function value.
    func_args : tuple
                Arguments passed to the function f.
    region : float
             Region of exploration around the centre point.
    Returns
    -------
    y : 1-D array
        Contains the response function values at each observation of the
        design matrix centred at centre_point.
    func_evals : integer
                 Number of times the repsonse function has been evaluated.
    """

    func_evals = 0
    y = np.zeros((n, ))
    for j in range(n):
        adj_point = np.zeros((m, ))
        adj_point[positions] = region * design[j, :]
        y[j] = f(adj_point + centre_point, *func_args)
        func_evals += 1
    return y, func_evals
