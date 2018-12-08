from operator import not_


from smop.libsmop import *
from scipy.stats import poisson
from scipy.optimize import linprog
import numpy as np
import __builtin__


def unseen(f=None, *args, **kwargs):

    f = np.asanyarray(f).reshape(-1,1).T
    k=30000

    gridFactor = 1.05

    xLPmin = 1 / float(np.dot(k, __builtin__.max(10, k)))

    min_i = min(find(f > 0))
    if min_i > 1:
        xLPmin = min_i / k

    maxLPIters = 1000
    x = 0
    histx = 0

    fLP = f.copy()

    fmax = max(find(fLP > 0))
    if min(size(fmax)) == 0:
        x = x(np.arange(1, end()))
        histx = histx(np.arange(1, end()))
        return histx, x

    LPmass = 1 - dot(x, np.transpose(histx))
    fLP = np.append(fLP,(zeros(1,int(ceil(sqrt(fmax)))))) #to change
    szLPf = max(size(fLP))
    xLPmax = fmax / float(k)

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
    b = [[0.0 for i in range(1)] for j in range(np.dot(2, szLPf))]

    for i in range(szLPf):
        t = np.dot(k, xLP)
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

    n = szLPx + dot(2, szLPf)
    n = np.array(n)[0][0]
    Aeq = [0 for _ in range(n)]
    t = xLP.copy()
    t = np.asarray(t)[0]
    Aeq[:szLPx] = t[:]
    beq = LPmass
    options = {'maxiter': maxLPIters, 'disp': False}
    for j in range(len(t)):
        for x in range(len(A)):
            A[x][j] = A[x][j] / t[j]
        Aeq[j] = Aeq[j] / t[j]

    lb =  0.0
    ub =  float("Inf")
    objf=np.array([objf]).T
    A = np.array(A)
    b = np.array(b)
    cobj = objf.flatten()

    res = linprog(c=cobj, A_ub=A, b_ub=b, A_eq=np.array([Aeq]), b_eq=np.array([beq]), bounds=(lb, ub), options=options, method = 'interior-point')

    if not res['success']:
        print(res['message'])

    objf2 = dot(0, objf)
    objf2[range(0, szLPx)] = 1

    fval = res['fun']
    for i in range(0, szLPx):
        objf2[i] = objf2[i] / t[i] #t is XLP

    exitflag2 = res['status']
    fval2 = res['fun']
    sol2 = res['x']

    if exitflag2 == 1:
        print ('LP2 solution was not found')
        return [0,0]

    tTranspose = t.T
    for i in range (0 , int(szLPx)):
        sol2[i] = sol2[i] / tTranspose[i]
    x=0
    x = np.append(x, t)
    histx = np.append(histx, sol2.T)
    ind = np.argsort(x)

    x = sort(x)

    histx = np.array(histx)[ind]
    h = []
    for val in histx:
        if val != 0:
            h.append(val)
    h = np.array(h)


    x = np.array(x)
    y=[]
    for val in x:
        if val != 0:
            y.append(val)
    y = np.array(y)

    return [h,y]
