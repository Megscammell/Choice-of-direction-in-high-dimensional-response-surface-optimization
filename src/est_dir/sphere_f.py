import numpy as np


def sphere_f(x, a, matrix):
    return ((x - a).T @ matrix @ (x - a))


def sphere_f_noise(x, a, matrix, mu, sd):
    return (sphere_f(x, a, matrix) + np.random.normal
            (mu, sd))
