# Generated with SMOP  0.41
from smop.libsmop import *
# makeFinger.m
from matplotlib.pyplot import hist




@function
def makeFinger(v=None,*args,**kwargs):

    varargin = makeFinger.varargin
    nargin = makeFinger.nargin

    # Input:  vector of integers, v
# Output: vector of fingerprints, f where f(i) = |{j: |{k:v(k)=j}|=i }|
#         i.e. f(i) is the number of elements that occur exactly i times

#         in the vector v

    h1=np.histogram(v,np.array(np.arange(np.min(v),np.max(v)+2))) #Random returns only 3 values

# makeFinger.m:8
    f=np.histogram(h1[0],np.array(np.arange(0 , np.max(h1[0])+2 )))
# makeFinger.m:9

    f=f[0][1:]#f(np.arange(2,end()))
# makeFinger.m:10

    f=ravel(f)
# makeFinger.m:10
    print f
    return f