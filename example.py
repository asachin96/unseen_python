import numpy as np
from math import log

import random

from unseen import unseen
from makeFinger import makeFinger
from entropy_estC import entropy_estC

n = 100000
k = 30000
samp = [random.uniform(0,n) for i in range(k)]
f = makeFinger(samp)
[h, x] = unseen(f)
trueEntropy = np.log(n)

arr = []

for y in range(1,len(f)+1):
    arr.append((y+1)/float(k))

for i in range(len(arr)):
    arr[i] = arr[i]*log(arr[i])

arr = np.array(arr).T
empiricalEntropy = np.dot(-f.T, arr)[0]

estimatedEntropy = np.matmul(-h,np.multiply(x, np.log(x)).T)
estimatedEntropy2 = entropy_estC(f)

print("TrueEntrop :" + str(trueEntropy), "EmpericalEntrop :" + str(empiricalEntropy), "EstimatedEntrop1 :" + str(estimatedEntropy), "EmpericalEntrop2 :" + str(estimatedEntropy2))
