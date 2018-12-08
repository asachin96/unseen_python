import matcompat
from scipy.optimize import linprog
from scipy.stats import poisson
import numpy as np;
from smop.libsmop import *
import __builtin__


def entropy_estC(f):

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

    fLP = f.copy() # BADA COMMENTs

    fmax = max(find(fLP > 0))
    if min(size(fmax)) == 0:
        x = x(np.arange(1, end()))
        histx = histx(np.arange(1, end()))
        return histx, x

    LPmass = 1 - np.dot(x, np.transpose(histx))
    fLP = np.append(fLP,(zeros(1,int(ceil(sqrt(fmax))))))
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

    m = np.dot(2, szLPf)
    n = szLPx + np.dot(2, szLPf)
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
    x=0

    res = linprog(c=cobj, A_ub=A, b_ub=b, A_eq=np.array([Aeq]), b_eq=np.array([beq]), bounds=(lb, ub), options=options, method = 'interior-point')
    sol2 = res['x']

    if not res['success']:
        print(res['message'])

    for j in range(szLPx):
        sol2[j] = sol2[j]/xLP[j]
    if max(matcompat.size(find((x > 0.)))) == 0.:
        t = np.asarray(xLP)[0]
        for j in range(len(t)):
            t[j] = t[j]*log(t[j])
        ent = np.dot(-sol2[0:szLPx].T, t.T)

    return ent
