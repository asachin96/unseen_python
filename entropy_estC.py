
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def entropy_estC(f):

    # Local Variables: xLPmax, A, sol, xLPmin, b2, objf, ind, sol2, maxLPIters, fLP, gridFactor, fval, A2, ent, min_i, xLP, szLPx, fmax, exitflag, objf2, output, alpha, LPmass, szLPf, histx, b, f, i, exitflag2, k, Aeq, fval2, x, beq, options, wind
    # Function calls: poisspdf, sort, linprog, log, min, max, sum, entropy_estC, sqrt, ceil, ones, zeros, optimset, not, Inf, find, size
    #% Gregory Valiant and Paul Valiant
    #% Code implementing an adaptation of the entropy estimation procedure from
    #% 'Estimating the Unseen: improved estimators for entropy and other
    #% properties'
    #%
    #% Input: fingerprint f, where f(i) represents number of elements that
    #% appear i times in a sample.  Thus sum_i i*f(i) = sample size.
    #% File makeFinger.m transforms a sample into the associated fingerprint.  
    #%
    #% Output: approximation of 'histogram' of true distribution.  Specifically,
    #% histx(i) represents the number of domain elements that occur with
    #% probability x(i).   Thus sum_i x(i)*histx(i) = 1, as distributions have
    #% total probability mass 1.   
    #%
    #% 
    f = f.flatten(0).conj()
    k = np.dot(f, np.arange(1., (matcompat.size(f, 2.))+1).conj().T)
    #%total sample size
    #%%%%%%% algorithm parameters %%%%%%%%%%%
    gridFactor = 1.05
    #% the grid of probabilities will be geometric, with this ratio.
    #% setting this smaller may slightly increase accuracy, at the cost of speed 
    alpha = .5
    #%the allowable discrepancy between the returned solution and the "best" (overfit).
    #% 0.5 worked well in all examples we tried, though the results were nearly indistinguishable 
    #% for any alpha between 0.25 and 1.  Decreasing alpha increases the chances of overfitting. 
    xLPmin = 1./np.dot(k, matcompat.max(10., k))
    min_i = matcompat.max(nonzero((f > 0.)))
    if min_i > 1.:
        xLPmin = matdiv(min_i, k)
    
    
    #% minimum allowable probability. 
    #% a more aggressive bound like 1/k^1.5 would make the LP slightly faster,
    #% though at the cost of accuracy
    maxLPIters = 1000.
    #% the 'MaxIter' parameter for Matlab's 'linprog' LP solver.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Split the fingerprint into the 'dense' portion for which we 
    #% solve an LP to yield the corresponding histogram, and 'sparse' 
    #% portion for which we simply use the empirical histogram
    x = 0.
    histx = 0.
    fLP = np.zeros(1., matcompat.max(matcompat.size(f)))
    for i in np.arange(1., (matcompat.max(matcompat.size(f)))+1):
        if f[int(i)-1] > 0.:
            wind = np.array(np.hstack((matcompat.max(1., (i-np.ceil(np.sqrt(i)))), matcompat.max((i+np.ceil(np.sqrt(i))), matcompat.max(matcompat.size(f))))))
            if np.sum(f[int(wind[0])-1:wind[1]])<np.sqrt(i):
                #% 2*sqrt(i)
                x = np.array(np.hstack((x, matdiv(i, k))))
                histx = np.array(np.hstack((histx, f[int(i)-1])))
                fLP[int(i)-1] = 0.
            else:
                fLP[int(i)-1] = f[int(i)-1]
                
            
        
        
        
    #% If no LP portion, return the empirical histogram and corrected empirical
    #% entropy
    fmax = matcompat.max(nonzero((fLP > 0.)))
    if matcompat.max(matcompat.size(fmax)) == 0.:
        x = x[1:]
        histx = histx[1:]
        ent = np.dot(-histx, (x*np.log(x)).conj().T)+matdiv(np.sum(histx), 2.*k)
        return []
    
    
    #% Set up the first LP
    LPmass = 1.-np.dot(x, histx.conj().T)
    #%amount of probability mass in the LP region
    fLP = np.array(np.hstack((fLP[0:fmax], np.zeros(1., np.ceil(np.sqrt(fmax))))))
    szLPf = matcompat.max(matcompat.size(fLP))
    xLPmax = matdiv(fmax, k)
    xLP = np.dot(xLPmin, gridFactor**np.arange(0., (np.ceil(matdiv(np.log(matdiv(xLPmax, xLPmin)), np.log(gridFactor))))+1))
    szLPx = matcompat.max(matcompat.size(xLP))
    objf = np.zeros((szLPx+2.*szLPf), 1.)
    objf[int(szLPx+1.)-1::2.] = matdiv(1., np.sqrt((fLP+1.)))
    #% discrepancy in ith fingerprint expectation
    objf[int(szLPx+2.)-1::2.] = matdiv(1., np.sqrt((fLP+1.)))
    #% weighted by 1/sqrt(f(i) + 1)
    A = np.zeros((2.*szLPf), (szLPx+2.*szLPf))
    b = np.zeros((2.*szLPf), 1.)
    for i in np.arange(1., (szLPf)+1):
        A[int((2.*i-1.))-1,0:szLPx] = poisspdf(i, np.dot(k, xLP))
        A[int((2.*i))-1,0:szLPx] = np.dot(-1., A[int((2.*i-1.))-1,0:szLPx])
        A[int((2.*i-1.))-1,int((szLPx+2.*i-1.))-1] = -1.
        A[int((2.*i))-1,int((szLPx+2.*i))-1] = -1.
        b[int((2.*i-1.))-1] = fLP[int(i)-1]
        b[int((2.*i))-1] = -fLP[int(i)-1]
        
    Aeq = np.zeros(1., (szLPx+2.*szLPf))
    Aeq[0:szLPx] = xLP
    beq = LPmass
    options = optimset('MaxIter', maxLPIters, 'Display', 'off')
    for i in np.arange(1., (szLPx)+1):
        A[:,int(i)-1] = matdiv(A[:,int(i)-1], xLP[int(i)-1])
        #%rescaling for better conditioning
        Aeq[int(i)-1] = matdiv(Aeq[int(i)-1], xLP[int(i)-1])
        
    [sol, fval, exitflag, output] = linprog(objf, A, b, Aeq, beq, np.zeros((szLPx+2.*szLPf), 1.), np.dot(Inf, np.ones((szLPx+2.*szLPf), 1.)), np.array([]), options)
    if exitflag == 0.:
        'maximum number of iterations reached--try increasing maxLPIters'
    
    
    if exitflag<0.:
        'LP1 solution was not found, still solving LP2 anyway...'
        exitflag
    
    
    #% Solve the 2nd LP, which minimizes support size subject to incurring at most
    #% alpha worse objective function value (of the objective function in the 
    #% previous LP). 
    if min_i<2.:
        objf2 = 0.*objf
        objf2[0:szLPx] = 1.
        A2 = np.array(np.vstack((np.hstack((A)), np.hstack((objf.conj().T)))))
        #% ensure at most alpha worse obj value
        b2 = np.array(np.vstack((np.hstack((b)), np.hstack((fval+alpha)))))
        #% than solution of previous LP
        for i in np.arange(1., (szLPx)+1):
            objf2[int(i)-1] = matdiv(objf2[int(i)-1], xLP[int(i)-1])
            #%rescaling for better conditioning
            
        [sol2, fval2, exitflag2, output] = linprog(objf2, A2, b2, Aeq, beq, np.zeros((szLPx+2.*szLPf), 1.), np.dot(Inf, np.ones((szLPx+2.*szLPf), 1.)), np.array([]), options)
        if not_rename((exitflag2 == 1.)):
            'LP2 solution was not found'
            exitflag2
        
        
    else:
        sol2 = sol
        
    
    sol2[0:szLPx] = sol2[0:szLPx]/xLP.conj().T
    #%removing the scaling
    #%append LP solution to empirical portion of histogram
    if matcompat.max(matcompat.size(nonzero((x > 0.)))) == 0.:
        ent = np.dot(-sol2[0:szLPx].conj().T, (xLP*np.log(xLP)).conj().T)
    else:
        ent = np.dot(-histx[int(nonzero((x > 0.)))-1], (x[int(nonzero((x > 0.)))-1]*np.log(x[int(nonzero((x > 0.)))-1])).conj().T)+matdiv(np.sum(histx[int(nonzero((x > 0.)))-1]), 2.*k)-np.dot(sol2[0:szLPx].conj().T, (xLP*np.log(xLP)).conj().T)
        
    
    x = np.array(np.hstack((x, xLP)))
    histx = np.array(np.hstack((histx, sol2.conj().T)))
    [x, ind] = np.sort(x)
    histx = histx[int(ind)-1]
    ind = nonzero((histx > 0.))
    x = x[int(ind)-1]
    histx = histx[int(ind)-1]
    return [ent]
