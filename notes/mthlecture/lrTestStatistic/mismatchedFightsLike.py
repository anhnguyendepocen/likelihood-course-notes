#!/usr/bin/env python
from scipy import optimize
import sys
import math
import random
def calc_prob_same(prob_strong, w):
    assert (prob_strong >= 0.0)
    assert (prob_strong <= 1.0)
    assert (w >= .5)    
    assert (w <= 1)
    if (w == .5) or (prob_strong == 0.0) or (prob_strong == 1.0):
        return 0.5
    omp = 1.0 - prob_strong
    prob_mismatch = 2*prob_strong*omp
    return prob_mismatch*(w*w + (1-w)*(1-w)) + .5*(1 - prob_mismatch)

def ln_likelihood(prob_strong, w, num_same, num_diff):
    # Return -inf if the parameters are out of range...
    if w != w or prob_strong != prob_strong:
        return float('-inf')
    if (w < .5) or (w > 1.0) or (prob_strong < 0.0) or (prob_strong > 1.0):
        return float('-inf')
    # If the parameters are legal, then return the log-likelihood
    prob_same = calc_prob_same(prob_strong, w)
    prob_diff = 1.0 - prob_same
    ln_prob_same = math.log(prob_same)
    ln_prob_diff = math.log(prob_diff)
    ln_l = num_same*ln_prob_same + num_diff*ln_prob_diff
    return ln_l

def scipy_ln_likelihood(x):
    global num_same, num_diff
    raw = ln_likelihood(x[0], x[1], num_same, num_diff)
    return -raw


def estimate_global_MLE(s, d):
    global num_same, num_diff
    num_same = s
    num_diff = d
    x0 = [.75, .75]
    param_opt = optimize.fmin(scipy_ln_likelihood, x0, xtol=1e-8, disp=False)
    return param_opt[0], param_opt[1], -scipy_ln_likelihood(param_opt)

def maximize_lnL_fixed_w(s, d, w):
    global num_same, num_diff
    num_same = s
    num_diff = d
    x0 = [.75]
    f = lambda x : scipy_ln_likelihood([x[0], w])
    param_opt = optimize.fmin(f, x0, xtol=1e-8, disp=False)
    return param_opt[0], -f(param_opt)

def simulate_data(s, w, n):
    ns = 0
    nd = 0
    prob_same_given_mismatch = w*w + (1-w)*(1-w)
    for i in range(n):
        if random.random() < s:
            zero_strong = True
        else:
            zero_strong = False
        if random.random() < s:
            one_strong = True
        else:
            one_strong = False

        if one_strong == zero_strong:
            if random.random() < 0.5:
                ns = ns + 1
            else:
                nd = nd + 1
        else:
            if random.random() < prob_same_given_mismatch:
                ns = ns + 1
            else:
                nd = nd + 1
    return ns, nd

if __name__ == '__main__':
    # user-interface and sanity checking...
    print_estimates = False
    
    num_same = int(sys.argv[1])
    num_diff = int(sys.argv[2])
    w_null = float(sys.argv[3])
    num_sims = int(sys.argv[4])
        
    if num_diff < 0 or num_same < 0:
        sys.exit("The number of bouts won by the same or different individuals must be non-negative")
    
    
    
    # Calculate and report the MLEs and LRT...
    
    print
    print num_same, 'trials in which the same individual wins both bouts.'
    print num_diff, 'trials in which different individuals wins each bout.'
    print

    prob_strong_MLE, w_MLE, lnL = estimate_global_MLE(num_same, num_diff)

    print "MLE of prob_strong =", prob_strong_MLE
    print "MLE of w =", w_MLE
    print "lnL at MLEs =", lnL
    print "L at MLEs =", math.exp(lnL)
    print

    prob_strong_null, lnL_null = maximize_lnL_fixed_w(num_same, num_diff, w_null)

    print "MLE of prob_strong at null w =", prob_strong_null
    print "null of w =", w_null
    print "lnL at null =", lnL_null
    print "L at null =", math.exp(lnL_null)
    print
    
    lrt = 2*(lnL_null - lnL)

    print "2* log-likelihood ratio = ", lrt
    print



    # Parametric bootstrapping to produce the null distribution of the LRT statistic
    if num_sims < 1:
        sys.exit(0)
        
    print "Generating null distribution of LRT..."
    n = num_same + num_diff

    sys.stderr.write("rep\t")
    if print_estimates:
        sys.stderr.write("s_hat\tw_hat\tnull_s_hat\t")
    sys.stderr.write("lrt\n")
    null_dist = []
    for i in range(num_sims):
        sim_n_same, sim_n_diff = simulate_data(prob_strong_null, w_null, n)
        sim_s_mle, sim_w_mle, sim_max_lnL = estimate_global_MLE(sim_n_same, sim_n_diff)
        sim_s_null, sim_lnL_null = maximize_lnL_fixed_w(sim_n_same, sim_n_diff, w_null)
        sim_lrt = 2*(sim_lnL_null - sim_max_lnL)

        null_dist.append(sim_lrt)

        sys.stderr.write(str(i + 1))
        sys.stderr.write('\t')
        if print_estimates:
            sys.stderr.write(str(sim_s_mle))
            sys.stderr.write('\t')
            sys.stderr.write(str(sim_w_mle))
            sys.stderr.write('\t')
            sys.stderr.write(str(sim_s_null))
            sys.stderr.write('\t')
        sys.stderr.write(str(sim_lrt))
        sys.stderr.write('\n')
    
    null_dist.sort()
    print "5% critical value is approx =", null_dist[int(0.05*num_sims)]
    
    num_more_extreme = 0
    for v in null_dist:
        if v < lrt:
            num_more_extreme = num_more_extreme + 1
        else:
            break
    
    print "Approx P-value =", num_more_extreme/float(num_sims)



