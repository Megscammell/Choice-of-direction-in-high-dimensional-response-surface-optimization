import numpy as np
import math
from scipy.stats import ortho_group



def sphere_func_params(lambda_1, lambda_2, d):
    if lambda_1 != lambda_2:
        diag_vals = np.zeros(d)
        diag_vals[:2] = np.array([lambda_1, lambda_2])
        diag_vals[2:] = np.random.uniform(lambda_1 + 0.1,
                                        lambda_2 - 0.1, (d - 2))
        A = np.diag(diag_vals)
        rotation = ortho_group.rvs(dim=d)
        return rotation.T @ A @ rotation
    else:
        return np.identity(d)