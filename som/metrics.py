import numpy as np


def get_metric(metric: str):
    metric = metric.lower()

    if metric in ['l1', 'manhattan']:
        return manhattan_distance

    if metric in ['l2', 'euclidian']:
        return euclidian_distance


def manhattan_distance(x, y):
    return np.abs(x - y).sum(axis=-1)


def euclidian_distance(x, y):
    return np.sqrt(np.square(x - y).sum(axis=-1))
