# Generated with SMOP  0.41
from operator import not_

from numpy import Inf

from smop.libsmop import *
# unseen.m
from scipy.stats import poisson
from scipy.optimize import linprog
import __builtin__


@function
def unseen(f=None, *args, **kwargs):
    varargin = unseen.varargin
    nargin = unseen.nargin

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
    for i in np.arange(1, max(size(f))).reshape(-1):
        if f[0][i] > 0:
            wind = concat([max(1, i - ceil(sqrt(i))), __builtin__.min(i + ceil(sqrt(i)), max(size(f)))])
            # unseen.m:43
            sum = 0
            for j in np.arange(wind[0], wind[1]):
                sum = sum + f[0][int(j)]
            if sum < sqrt(i):
                x = concat([x, i / k])
                # unseen.m:45
                histx = concat([histx, f[0][i]])
                # unseen.m:46
                fLP[i] = 0
            # unseen.m:47
            else:
                fLP[i] = f[0][i]



    fLP = f # BADA COMMENTs
    # unseen.m:49

    # If no LP portion, return the empirical histogram
    fmax = max(find(fLP > 0))
    # unseen.m:55
    if min(size(fmax)) == 0:
        x = x(np.arange(2, end()))
        # unseen.m:57
        histx = histx(np.arange(2, end()))
        # unseen.m:58
        return histx, x
    # Set up the first LP
    LPmass = 1 - dot(x, np.transpose(histx))
    # unseen.m:63
    print LPmass
    print zeros(1,int(ceil(sqrt(fmax))))
    fLP = np.append(fLP,(zeros(1,int(ceil(sqrt(fmax)))))) #to change
    # fLP=concat(l)
    # unseen.m:65
    szLPf = max(size(fLP))
    # unseen.m:66
    #xLPmax = fmax / k
    xLPmax = fmax / float(k)
    # unseen.m:68


    xLP = dot(xLPmin, gridFactor ** (arange(0, ceil(log(xLPmax / xLPmin) / log(gridFactor)))))
    # unseen.m:69
    szLPx = max(size(xLP))
    # unseen.m:70
    i = szLPx + dot(2, szLPf)
    i = np.array(i)
    objf = zeros(i[0][0] , 1)
    # unseen.m:72

    objf[arange(szLPx + 1, end(), 2)] = 1.0 / (sqrt(fLP + 1))
    # unseen.m:73

    objf[arange(szLPx + 2, end(), 2)] = 1.0 / (sqrt(fLP + 1))
    # unseen.m:74

    A = zeros(dot(2, szLPf), szLPx + dot(2, szLPf))
    # unseen.m:76
    b = zeros(dot(2, szLPf), 1)
    # unseen.m:77
    for i in arange(1, szLPf).reshape(-1):
        A[dot(2, i) - 1, arange(1, szLPx)] = poisson.pmf(i, dot(k, xLP))
        # unseen.m:79
        A[dot(2, i), arange(1, szLPx)] = dot((- 1), A(dot(2, i) - 1, arange(1, szLPx)))
        # unseen.m:80
        A[dot(2, i) - 1, szLPx + dot(2, i) - 1] = - 1
        # unseen.m:81
        A[dot(2, i), szLPx + dot(2, i)] = - 1
        # unseen.m:82
        b[dot(2, i) - 1] = fLP(i)
        # unseen.m:83
        b[dot(2, i)] = - fLP(i)
    # unseen.m:84

    Aeq = zeros(1, szLPx + dot(2, szLPf))
    # unseen.m:87
    Aeq[arange(1, szLPx)] = xLP
    # unseen.m:88
    beq = copy(LPmass)
    # unseen.m:89
    options = {'maxiter': maxLPIters, 'disp': False}
    # unseen.m:92
    for i in arange(1, szLPx).reshape(-1):
        A[arange(), i] = A(arange(), i) / xLP(i)
        # unseen.m:94
        Aeq[i] = Aeq(i) / xLP(i)
    # unseen.m:95

    res = linprog(objf, A, b, Aeq, beq, zeros(szLPx + dot(2, szLPf), 1),
                  dot(Inf, ones(szLPx + dot(2, szLPf), 1)), [], options, nargout=4)
    # unseen.m:97
    if res['exitflag'] == 0:
        'maximum number of iterations reached--try increasing maxLPIters'

    if res['exitflag'] < 0:
        'LP1 solution was not found, still solving LP2 anyway...'
        return res['exitflag']

    # Solve the 2nd LP, which minimizes support size subject to incurring at most
    # alpha worse objective function value (of the objective function in the
    # previous LP).
    objf2 = dot(0, objf)
    # unseen.m:109
    objf2[arange(1, szLPx)] = 1
    # unseen.m:110
    A2 = concat([[A], [objf.T]])
    # unseen.m:111

    b2 = concat([[b], [res['fun'] + alpha]])
    # unseen.m:112

    for i in arange(1, szLPx).reshape(-1):
        objf2[i] = objf2(i) / xLP(i)
    # unseen.m:114

    # sol2,fval2,exitflag2,output=linprog(objf2,A2,b2,Aeq,beq,[ (zeros(szLPx + dot(2,szLPf),1)),(dot(float('Inf'),ones(szLPx + dot(2,szLPf),1))) ],options)
    res = linprog(objf2, A2, b2, Aeq, beq, [(zeros(szLPx + dot(2, szLPf), 1)), (
        dot(float('Inf'), ones(szLPx + dot(2, szLPf), 1)))], options)
    exitflag2 = res['status']
    fval2 = res['fun']
    sol2 = res['x']

    # unseen.m:116
    if not_(exitflag2 == 1):
        'LP2 solution was not found'
        exitflag2

    # append LP solution to empirical portion of histogram
    sol2[arange(1, szLPx)] = sol2(arange(1, szLPx)) / xLP.T
    # unseen.m:125

    x = concat([x, xLP])
    # unseen.m:126
    histx = concat([histx, sol2.T])
    # unseen.m:127
    x, ind = sort(x)
    # unseen.m:128
    histx = histx(ind)
    # unseen.m:129
    ind = find(histx > 0)
    # unseen.m:130
    x = x(ind)
    # unseen.m:131
    histx = histx(ind)
# unseen.m:132
