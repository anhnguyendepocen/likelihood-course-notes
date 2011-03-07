#!/usr/bin/env python
import math
def lnLBernoulli(n, k, p):
    if p <= 0.0:
        if k > 0:
            return float('-inf')
        return 0.0
    if p >= 1.0:
        if k < n:
            return float('-inf')
        return 0.0
    return k*math.log(p) + (n-k)*math.log(1-p)

def lnLAndPHat(n, k):
    pHat = float(k)/n
    return lnLBernoulli(n, k, pHat), pHat


def bernoulliLnLAndPHat(samples):
    for i in samples:
        if i != 0 and i != 1:
            raise ValueError("Expecting vector of 1's and 0's.  Got: %s" % str(samples))
    n = len(samples)
    k = sum(samples)
    return lnLAndPHat(n, k)

def resampledModelLnLandThetaHat(samples):
    '''Returns the lnL and (pHat, rHat) where pHat is the MLE of draws from the
    population and rHat is the MLE of probability that the same individual will
    be sampled on repeated draws.
    '''
    for i in samples:
        if i != 0 and i != 1:
            raise ValueError("Expecting vector of 1's and 0's.  Got: %s" % str(samples))
    as_str = ''.join([str(int(i)) for i in samples])
    assert(len(as_str) == len(samples))
    k00 = as_str.count(
    
print lnLAndPHat(10, 6):
    
