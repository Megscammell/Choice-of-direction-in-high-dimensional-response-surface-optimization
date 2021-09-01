import numpy as np


def square_sphere_f(x, a, matrix):
    return ((x - a).T @ matrix @ (x - a))**2


def square_sphere_f_noise(x, a, matrix, mu, sd):
    return (square_sphere_f(x, a, matrix) + np.random.normal
            (mu, sd))