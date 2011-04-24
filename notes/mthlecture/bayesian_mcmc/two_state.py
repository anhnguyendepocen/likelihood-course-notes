#!/usr/bin/env python
import sys
from random import random as uniform

a_to_b = float(sys.argv[1])
assert(a_to_b > 0.0 and a_to_b <= 1.0)
b_to_a = float(sys.argv[2])
assert(b_to_a > 0.0 and b_to_a <= 1.0)
ti_prob_list = [a_to_b, b_to_a]
b_to_b = 1 - b_to_a
state = int(sys.argv[4])
assert(state in [0,1])
if state == 0:
    prob_list = [1.0, 0.0]
else:
    prob_list = [0.0, 1.0]
num_it = int(sys.argv[3])
assert(num_it > 0)

print "Gen\tPrZero\tPrOne"
for i in xrange(num_it):
    print "\t".join([str(i)] + [str(p) for p in prob_list])
    pb = prob_list[0]*a_to_b + prob_list[1]*b_to_b
    prob_list = [1-pb, pb]
