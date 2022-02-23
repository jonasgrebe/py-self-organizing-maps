import numpy as np


def get_neighborhood_function(neighborhood_function: str):
    neighborhood_function = neighborhood_function.lower()

    if neighborhood_function in ['gaussian']:
        return gaussian_neighborhood_function

    if neighborhood_function in ['bubble', 'step']:
        return bubble_neighborhood_function

    if neighborhood_function in ['triangle']:
        return triangle_neighborhood_function


def gaussian_neighborhood_function(d, radius):
    return np.exp(- d ** 2 / (2 * radius ** 2)).reshape(-1, 1)


def bubble_neighborhood_function(d, radius):
    return np.where(d <= radius, 1, 0).reshape(-1, 1)


def triangle_neighborhood_function(d, radius):
    return np.where(d <= radius, 1 - d / radius, 0).reshape(-1, 1)
