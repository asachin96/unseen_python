import numpy as np
import matcompat
import matplotlib.pyplot as plt

def makeFinger(v):

    # Local Variables: h1, v, f
    # Function calls: hist, max, makeFinger, min
    #% Input:  vector of integers, v
    #% Output: vector of fingerprints, f where f(i) = |{j: |{k:v(k)=j}|=i }|
    #%         i.e. f(i) is the number of elements that occur exactly i times 
    #%         in the vector v
    h1 = plt.hist(v, np.arange(max(v), (max(v))+1))
    f = plt.hist(h1, np.arange(0., (max(h1))+1))
    f = f[1:]
    f = f.flatten(1)
    return [f]