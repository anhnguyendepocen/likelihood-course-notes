#!/usr/bin/env python
import math
import sys

data =  p = [3, 1, 2, 3, 2, 1, 5, 4, 4, 5, 6, 4, 1, 1, 1, 2, 3, 0, 0, 4, ]

def factorial(k):
    if k < 2:
        return 1
    return k * factorial(k-1)

def poisson_ln_l(k, lam):
    kfac = factorial(k)
    return -lam + k*math.log(lam) - math.log(kfac)


lam = float(sys.argv[1])
ln_l = sum([poisson_ln_l(x, lam) for x in data])
print ln_l
