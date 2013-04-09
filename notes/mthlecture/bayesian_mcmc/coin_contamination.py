#!/usr/bin/env python 
'''
Uses MCMC to estimate the probability of different values of theta (the number
    of double-headed coins) given data from independent trials that entail
    flipping all n coins.

A uniform prior over all values of theta is assumed.

Example invocation 
    python coin_contamination.py 1000000 4 1 4 3
means run:
    - 1 million iterations of MCMC
    - with n=4 coins in each trial
    - starting at the state of theta=1 double-headed coin out of the 4
    - two observations: 4 heads and 3 heads
    
'''
import random
import sys


try:
    num_it = int(sys.argv[1])
    assert(num_it > 0)
    num_coins = int(sys.argv[2])
    assert(num_coins > 0)
    state = int(sys.argv[3])
    assert(num_coins > 0)
    data = tuple([int(i) for i in sys.argv[4:]])
    assert(len(data) > 0)
except:
    sys.exit('Expeciting:\npython coin_contamination.py <# iter> <# coins> <start state> <datum #1> <datum #2>...\n')

max_obs = max(data)
assert(max_obs <= num_coins)
assert(min(data) >= 0)

def n_choose_k(n, k):
    num, denom = 1, 1
    if 2*k > n:
        return n_choose_k(n, n-k)
    for i in range(k):
        num *= (n - i)
        denom *= (i + 1)
    nck = num/denom
    #print n, k, nck
    return nck

def fair_coin_prob(h, n):
    return n_choose_k(n, h)/float(2**n)

# since the range of each datum is small and the set 
#   of parameter values is small, we'll calculate 
#   the likelihood for every combination.
likelihood_factors = []
for n in range(max_obs + 1):
    p = []
    for theta in range(num_coins + 1):
        if theta > n:
            p.append(0.0)
        else:
            p.append(fair_coin_prob(n - theta, num_coins - theta))
    likelihood_factors.append(p)


def calc_likelihood(theta):
    like = 1.0
    for h in data:
        like *= likelihood_factors[h][theta]
    return like

likelihood = calc_likelihood(state)
assert(likelihood > 0.0)

mcmc_samples = [0]*(num_coins + 1)

# This is MCMC using the Metropolis algorithm:
sys.stderr.write("Iter\tlike\ttheta\n")
for i in xrange(num_it):
    sys.stderr.write("%d\t%f\t%d\n" % (i, likelihood, state))
    mcmc_samples[state] += 1
    prev_likelihood = likelihood
    # propose a state from among the adjacent states
    if random.random() < 0.5:
        proposed = state + 1
        if proposed > num_coins:
            proposed = 0
    else:
        proposed = state - 1
        if proposed < 0:
            proposed = num_coins
    # Prior ratio is 1.0, so we could ignore it...
    prior_ratio = (0.2)/(0.2)

    likelihood = calc_likelihood(proposed)
    likelihood_ratio = likelihood/prev_likelihood

    posterior_ratio = likelihood_ratio*prior_ratio

    # Hastings ratio is 1.0, so we can ignore it...
    hastings_ratio = (0.5)/(0.5)

    acceptance_ratio = posterior_ratio*hastings_ratio
    #print state, '->', proposed, 'alpha=', acceptance_ratio, 'prev_likelihood=',prev_likelihood, 'proposed_likelihood=',likelihood
    if random.random() < acceptance_ratio:
        state = proposed
    else:
        # state = old state already, so we don't have to change it
        likelihood = prev_likelihood
            
print "Posterior probabilities from MCMC"
for state in range(num_coins + 1):
    print state, float(mcmc_samples[state])/num_it

print "\nTrue Posterior probabilities (calculated analytically)"
likelihood_list = [calc_likelihood(i) for i in range(num_coins + 1)]
marginal_prob = sum(likelihood_list)
for state, likelihood in enumerate(likelihood_list):
    print state, likelihood/marginal_prob

print "\nTransition Probabilities:\n                       From"
print "     " + "    ".join(["%7d" % i for i in range(num_coins + 1)])
for num_dh in range(num_coins + 1):
    ind_below = num_dh - 1
    ind_above = num_dh + 1 if num_dh < num_coins else 0
    same_state = 0.0
    likelihood = likelihood_list[num_dh]
    if likelihood == 0:
        ti_prob_above, ti_prob_below = 0.5, 0.5
    else:
        like_below, like_above = likelihood_list[ind_below], likelihood_list[ind_above]
        if like_above > likelihood:
            ti_prob_above = 0.5
        else:
            ti_prob_above = 0.5*like_above/likelihood
            same_state += (0.5 - ti_prob_above)
        if like_below > likelihood:
            ti_prob_below = 0.5
        else:
            ti_prob_below = 0.5*like_below/likelihood
            same_state += (0.5 - ti_prob_below)
    ti_probs = [0.0] * (num_coins + 1)
    ti_probs[num_dh] = same_state
    ti_probs[ind_below] = ti_prob_below
    ti_probs[ind_above] = ti_prob_above
    print "%-4d  %s" % (num_dh, " ".join([" %7f " % d for d in ti_probs]))
