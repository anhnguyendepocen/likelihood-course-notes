#!/usr/bin/env python
import sys
from random import random as uniform

a_to_b = float(sys.argv[1])
assert(a_to_b > 0.0 and a_to_b < 1.0)
b_to_a = float(sys.argv[2])
assert(b_to_a > 0.0 and b_to_a < 1.0)
ti_prob_list = [a_to_b, b_to_a]
if uniform() < 0.5:
    state = 0
else:
    state = 1

num_it = int(sys.argv[3])
assert(num_it > 0)

print "Gen\tState"
for i in xrange(num_it):
    print "%d\t%d" % (i, state)
    ti_prob = ti_prob_list[state]
    if uniform() < ti_prob:
        state = 1 - state

