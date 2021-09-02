import numpy as np
import est_dir


def compute_shuffle_cols(temp):
    """
    Shuffle entries in each column.

    Parameters
    ----------
    temp : 2-D array
           Initial design matrix

    Returns
    -------
    random_design_matrix : 2-D array
                           Design matrix with all entries in each column
                           randomly shuffled.
    """

    random_design_matrix = np.take_along_axis(
                           temp,
                           np.random.rand(*temp.shape).argsort(axis=0),
                           axis=0)
    return random_design_matrix


def compute_random_design(n, m, centre_point, no_vars, f, func_args,
                          region):
    """
    Compute random design matrix centred at centre_point, where entries are
    chosen randomly as +1 or -1, with the condition that each column of the
    design matrix has the same number of +1's or -1's. Also, compute the
    response function value at each observation of the design matrix.

    Parameters
    ----------
    n : integer
        Number of observations of the design matrix (rows).
    m : integer
        Number of variables of the design matrix (columns).
    centre_point : 1-D array with shape (m,)
                   Centre point of design.
    no_vars : integer
              If no_vars < m, the size of the resulting
              design matrix is (n, no_vars). Since the centre_point is of size
              (m,), a random subset of variables will need to be chosen
              to evaluate the design matrix centred at centre_point. The
              parameter no_vars will be used to generate a random subset of
              positions, which correspond to the variable indices of
              centre_point in which to centre the design matrix.
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
    design : 2-D array
             Random design matrix with same number of +1's and -1's in each
             column.
    y : 1-D array
        Contains the response function values at each observation of the
        design matrix centred at centre_point.
    positions : 1-D array
                Positions of centre_point in which the design matrix has been
                centred. If the design matrix is of size (n, m), then
                positions = (1,2,...,m). Otherwise, if the design matrix is
                of size (n, no_vars), where no_vars < m, then
                positions will be of size (num_vars,) with entries chosen
                randomly from (1,2,...,m).
    func_evals : integer
                 Number of times the repsonse function has been evaluated.

    """
    if (n % 2) != 0:
        raise ValueError('n must be even.')
    if no_vars == m:
        positions = np.arange(m)
    else:
        positions = np.sort(np.random.choice(np.arange(m), no_vars,
                            replace=False))
    assert(np.unique(positions).shape[0] == no_vars)

    arr = np.repeat(1, n)
    arr[:int(n/2)] = -1
    temp = np.tile(arr, (no_vars, 1)).T
    design = compute_shuffle_cols(temp)
    y, func_evals = est_dir.compute_y(centre_point, design, positions, n, m, f,
                                      func_args, region)
    return design, y, positions, func_evals
