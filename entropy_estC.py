
import numpy as np
import scipy
import matcompat
from scipy.optimize import linprog
from scipy.stats import poisson
from smop.libsmop import *
import __builtin__

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def entropy_estC(f):


    # Input: fingerprint f, where f(i) represents number of elements that
    # appear i times in a sample.  Thus sum_i i*f(i) = sample size.
    # File makeFinger.m transforms a sample into the associated fingerprint.

    # Output: approximation of 'histogram' of true distribution.  Specifically,
    # histx(i) represents the number of domain elements that occur with
    # probability x(i).   Thus sum_i x(i)*histx(i) = 1, as distributions have
    # total probability mass 1.

    # An approximation of the entropy of the true distribution can be computed
    # as:    Entropy = (-1)*sum(histx.*x.*log(x))
    f = np.asanyarray(f).reshape(-1,1).T
    # unseen.m:14
    #k = dot(f, (arange(1, size(f, 2))).T) #HARDCODED THIS SHIREEN
    k=30000
    # unseen.m:15

    ####### algorithm parameters ###########
    gridFactor = 1.05
    # unseen.m:19

    # setting this smaller may slightly increase accuracy, at the cost of speed
    alpha = 0.5
    # unseen.m:21

    # 0.5 worked well in all examples we tried, though the results were nearly indistinguishable
    # for any alpha between 0.25 and 1.  Decreasing alpha increases the chances of overfitting.
    xLPmin = 1 / float(dot(k, __builtin__.max(10, k)))

    # unseen.m:24
    min_i = min(find(f > 0))
    # unseen.m:25
    if min_i > 1:
        xLPmin = min_i / k
    # unseen.m:27

    # a more aggressive bound like 1/k^1.5 would make the LP slightly faster,
    # though at the cost of accuracy
    maxLPIters = 1000
    # unseen.m:31

    #######################################

    # Split the fingerprint into the 'dense' portion for which we
    # solve an LP to yield the corresponding histogram, and 'sparse'
    # portion for which we simply use the empirical histogram
    x = 0
    # unseen.m:38
    histx = 0
    # unseen.m:39
    fLP = zeros(1, max(size(f)))
    # unseen.m:40
    #check back later
    # for i in np.arange(1, max(size(f))).reshape(-1):
    #     if f[0][i] > 0:
    #         wind = concat([max(1, i - ceil(sqrt(i))), __builtin__.min(i + ceil(sqrt(i)), size(f))])
    #         # unseen.m:43
    #         sum = 0
    #         for j in np.arange(wind[0], wind[1]):
    #             sum = sum + f[0][int(j)]
    #         if sum < sqrt(i):
    #             x = concat([x, i / k])
    #             # unseen.m:45
    #             histx = concat([histx, f[0][i]])
    #             # unseen.m:46
    #             fLP[i] = 0
    #         # unseen.m:47
    #         else:
    #             fLP[i] = f[0][i]



    fLP = f.copy() # BADA COMMENTs
    # unseen.m:49

    # If no LP portion, return the empirical histogram
    fmax = max(find(fLP > 0))
    # unseen.m:55
    if min(size(fmax)) == 0:
        x = x(np.arange(1, end()))
        # unseen.m:57
        histx = histx(np.arange(1, end()))
        # ent = np.dot(-histx, (x * np.log(x)).conj().T) + matdiv(np.sum(histx), 2. * k)
        #Compute ent from .m
        # unseen.m:58
        return histx, x
    # Set up the first LP

    LPmass = 1 - dot(x, np.transpose(histx))
    # unseen.m:63
    fLP = np.append(fLP,(zeros(1,int(ceil(sqrt(fmax)))))) #to change
    # fLP=concat(l)
    # unseen.m:65
    szLPf = max(size(fLP))
    # unseen.m:66
    #xLPmax = fmax / k
    xLPmax = fmax / float(k)
    # unseen.m:68


    xLP = dot(xLPmin, gridFactor ** (arange(0, ceil(log(xLPmax / xLPmin) / log(gridFactor)))))
    szLPx = max(size(xLP))

    i = szLPx + dot(2, szLPf)
    i = np.array(i)

    objf = [0 for _ in range(i[0][0])]

    t = fLP.copy()
    for j in range(len(fLP)):
        t[j] = 1./sqrt(t[j]+1)
    m=0
    for j in range(szLPx, len(objf), 2):
        objf[j] = t[m]
        m+=1
    m = 0
    for j in range(szLPx + 1, len(objf), 2):
        objf[j] = t[m]
        m += 1

    m = dot(2, szLPf)
    n = szLPx + dot(2, szLPf)
    n = np.array(n)
    A = [[0.0 for i in range(n[0][0])] for j in range(m)]
    b = [[0.0 for i in range(1)] for j in range(dot(2, szLPf))]

    # unseen.m:77
    for i in range(szLPf):
        t = dot(k, xLP)
        t = np.array(t)[0]
        t = poisson.pmf(i+1, t)
        A[2 * i][:szLPx] = t[:]
        A[2 * i + 1][:szLPx] = -t[:]
        t = szLPx + 2*i
        t = np.asarray(t)[0][0]
        A[2*i][t] = - 1
        t = szLPx + 2 * i + 1
        t = np.asarray(t)[0][0]
        A[2*i+1][t] = - 1
        b[2*i][0] = fLP[i]
        b[2*i+1][0] = -fLP[i]

    # unseen.m:84
    n = szLPx + dot(2, szLPf)
    n = np.array(n)[0][0]
    Aeq = [0 for _ in range(n)]
    # unseen.m:87
    t = xLP.copy()
    t = np.asarray(t)[0]
    Aeq[:szLPx] = t[:]
    # unseen.m:88
    beq = LPmass
    # unseen.m:89
    options = {'maxiter': maxLPIters, 'disp': False}
    # unseen.m:92
    for j in range(len(t)):
        for x in range(len(A)):
            A[x][j] = A[x][j] / t[j]
        # unseen.m:94
        Aeq[j] = Aeq[j] / t[j]
    # unseen.m:95
    n = szLPx + dot(2, szLPf)
    n = np.array(n)[0][0]
    lb =  0.0
    ub =  float("Inf")
    objf=np.array([objf]).T
    A = np.array(A)
    b = np.array(b)
    cobj = objf.flatten()
    x=0

    res = linprog(c=cobj, A_ub=A, b_ub=b, A_eq=np.array([Aeq]), b_eq=np.array([beq]), bounds=(lb, ub), options=options, method = 'interior-point')
    sol2 = res['x']

    if not res['success']:
        print(res['message'])

    for j in range(szLPx):
        sol2[j] = sol2[j]/xLP[j]
    #%removing the scaling
    #%append LP solution to empirical portion of histogram
    if max(matcompat.size(find((x > 0.)))) == 0.:
        t = np.asarray(xLP)[0]
        for j in range(len(t)):
            t[j] = t[j]*log(t[j])
        ent = np.dot(-sol2[0:szLPx].T, t.T)

    return ent
