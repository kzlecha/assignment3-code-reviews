from numpy import ndarray, power, sqrt, sum as np_sum

def euclidean_distance(x: ndarray, y: ndarray) -> ndarray:
    """
    find the euclidean distance between data and centroid

    Assumption: data observations and centroid must be same dimension
    """
    return sqrt(np_sum(power(x - y, 2), axis=1))
