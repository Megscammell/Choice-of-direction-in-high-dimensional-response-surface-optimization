import numpy as np


# def sqrt_sphere_f(x, a, matrix):
#     return (np.sqrt(abs(x - a)).T @ matrix @ np.sqrt(abs(x - a)))


# def sqrt_sphere_f_noise(x, a, matrix, mu, sd):
#     return (sqrt_sphere_f(x, a, matrix) + np.random.normal
#             (mu, sd))


def sqrt_sphere_f(x, a, matrix):
    return (np.sqrt((x - a).T @ matrix @ (x - a)))


def sqrt_sphere_f_noise(x, a, matrix, mu, sd):
    return (sqrt_sphere_f(x, a, matrix) + np.random.normal
            (mu, sd))
