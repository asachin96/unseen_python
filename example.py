
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#% Generate a sample of size 10,000 from the uniform distribution of support 100,000
n = 100000.
k = 10000.
samp = randi(n, k, 1.)
#% Compute corresponding 'fingerprint'
f = makeFinger(samp)
#% Estimate histogram of distribution from which sample was drawn
[h, x] = unseen(f)
#%output entropy of the true distribution, Unif[n]
trueEntropy = np.log(n)
#%output entropy of the empirical distribution of the sample
empiricalEntropy = np.dot(-f.conj().T, (matdiv(np.arange(1., (matcompat.max(matcompat.size(f)))+1), k)*np.log(matdiv(np.arange(1., (matcompat.max(matcompat.size(f)))+1), k))).conj().T)
#%output entropy of the recovered histogram, [h,x]
estimatedEntropy = np.dot(-h, (x*np.log(x)).conj().T)
#%output entropy using entropy_estC.m (should be almost the same as above):
estimatedEntropy2 = entropy_estC(f)