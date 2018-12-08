import numpy as np


def makeFinger(v=None, *args, **kwargs):
    h1 = np.histogram(v, np.array(np.arange(np.min(v), np.max(v) + 2)))  # Random returns only 3 values

    f = np.histogram(h1[0], np.array(np.arange(0, np.max(h1[0]) + 2)))

    f = f[0][1:]

    f = np.asanyarray(f).reshape(-1, 1)

    return f
