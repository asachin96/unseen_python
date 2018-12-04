import numpy as np
import matcompat
import random

# if available import pylab (from matlibplot)
from unseen import unseen
from makeFinger import makeFinger

# % Generate a sample of size 10,000 from the uniform distribution of support 100,000
n = 100000
k = 30000
# samp = randi(n, k, 1.)
samp = [random.uniform(0,n) for i in range(k)]
print(samp)
# % Compute corresponding 'fingerprint'
f = makeFinger(samp)
# % Estimate histogram of distribution from which sample was drawn
[h, x] = unseen(f)
# %output entropy of the true distribution, Unif[n]
trueEntropy = np.log(n)
# %output entropy of the empirical distribution of the sample
arr = []

for y in range(0,len(f)):
    arr.append((y+1)/float(k))

arr = np.dot(arr, np.log(arr))
arr = arr.T
empiricalEntropy = np.dot(-f.T, arr)



# %output entropy of the recovered histogram, [h,x]

estimatedEntropy = np.matmul(-h,np.multiply(x, np.log(x)).T)
# %output entropy using entropy_estC.m (should be almost the same as above):
estimatedEntropy2 = entropy_estC(f)
