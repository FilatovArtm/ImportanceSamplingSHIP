import numpy as np

def sample(func, size=1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    result = np.zeros(size)
    for i in range(size):
        result[i] = func()

    return result
