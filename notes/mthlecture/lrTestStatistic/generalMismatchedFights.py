#!/usr/bin/env python
from scipy import optimize
import sys
import math
import random


# we are going to use a summary that is a pair of numbers for each pair of males
#   it will represent: 
#    (the number of bouts won by male 0, the number of bouts won by male 1)


real_data = [ (0, 2),
              (1, 1),
              (1, 1),
              (0, 2),
              (2, 0),
              (1, 1),
              (0, 2),
              (2, 0),
              (1, 1),
              (0, 2),
            ]


the_data = real_data


def ln_likelihood(prob_strong, w):
    global the_data
    # Return -inf if the parameters are out of range...
    if w != w or prob_strong != prob_strong:
        return float('-inf')
    if (w < .5) or (w > 1.0) or (prob_strong < 0.0) or (prob_strong > 1.0):
        return float('-inf')
    
    prob_SS = prob_strong*prob_strong
    prob_SW = prob_strong*(1-prob_strong)
    prob_WS = (1-prob_strong)*prob_strong
    prob_WW = (1-prob_strong)*(1 - prob_strong)
    
    prob_even = prob_WW + prob_SS

    prob0_even = 0.5
    prob1_even = 1 - prob0_even

    prob0_SW = w
    prob1_SW = 1 - prob0_SW

    prob0_WS = 1 - w
    prob1_WS = 1 - prob0_WS

    ln_l = 0.0
    for datum in the_data:
        num0_wins = datum[0]
        num1_wins  = datum[1]
        
        prob_datum_if_even = (prob0_even**num0_wins) * (prob1_even**num1_wins)
        prob_datum_if_SW = (prob0_SW**num0_wins) * (prob1_SW**num1_wins)
        prob_datum_if_WS = (prob0_WS**num0_wins) * (prob1_WS**num1_wins)

        prob_datum = (  prob_even*prob_datum_if_even
                      + prob_SW*prob_datum_if_SW
                      + prob_WS*prob_datum_if_WS )

        if prob_datum <= 0.0:
            return float('-inf')

        ln_l = ln_l + math.log(prob_datum)
    return ln_l

def scipy_ln_likelihood(x):
    raw = ln_likelihood(x[0], x[1])
    return -raw


def estimate_global_MLE():
    x0 = [.75, .75]
    param_opt = optimize.fmin(scipy_ln_likelihood, x0, xtol=1e-8, disp=False)
    return param_opt[0], param_opt[1], -scipy_ln_likelihood(param_opt)

def maximize_lnL_fixed_w(w):
    x0 = [.75]
    f = lambda x : scipy_ln_likelihood([x[0], w])
    param_opt = optimize.fmin(f, x0, xtol=1e-8, disp=False)
    return param_opt[0], -f(param_opt)

def simulate_data(s, w):
    global real_data
    template_data = real_data

    prob_SS = s*s
    prob_SW = s*(1-s)
    prob_WS = (1-s)*s
    prob_WW = (1-s)*(1 - s)
    prob_even = prob_SS + prob_WW
    
    
    sim_data_set = []

    for datum in template_data:
        num_bouts = sum(datum)
        
        # randomly pick the type of pairing the we have 'EVEN', 'SW' or 'WS'
        rand_match_p = random.random()
        if rand_match_p < prob_even:
            match_type = 'EVEN'
        else:
            if rand_match_p < prob_even + prob_SW:
                match_type = 'SW'
            else:
                match_type = 'WS'
        
        # determine the probability that male 0 wins each bout.        
        if match_type == 'EVEN':
            prob_zero_wins = 0.5
        if match_type == 'WS':
            prob_zero_wins = w
        if match_type == 'SW':
            prob_zero_wins = 1 - w
        
        # start out with no bouts won by either, then we are going to 
        #   simulate the result of num_bouts
        n0_won = 0
        n1_won = 0
        for bout in range(num_bouts):
            if random.random() < prob_zero_wins:
                n0_won = n0_won + 1
            else:
                n1_won = n1_won + 1
        
        # add this simulated outcome to our simulated data set
        sim_datum = (n0_won, n1_won)
        sim_data_set.append(sim_datum)

    return sim_data_set

if __name__ == '__main__':
    # user-interface and sanity checking...
    print_estimates = False
    
    w_null = float(sys.argv[1])
    num_sims = int(sys.argv[2])
    
    # Calculate and report the MLEs and LRT...

    prob_strong_MLE, w_MLE, lnL = estimate_global_MLE()

    print "MLE of prob_strong =", prob_strong_MLE
    print "MLE of w =", w_MLE
    print "lnL at MLEs =", lnL
    print "L at MLEs =", math.exp(lnL)
    print

    prob_strong_null, lnL_null = maximize_lnL_fixed_w(w_null)

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
    sys.stderr.write("rep\t")
    if print_estimates:
        sys.stderr.write("s_hat\tw_hat\tnull_s_hat\t")
    sys.stderr.write("lrt\n")
    null_dist = []
    for i in range(num_sims):
        sim_data = simulate_data(prob_strong_null, w_null)
        
        # Update the global variable "the_data" to reflect the simulation
        the_data = sim_data
        sim_s_mle, sim_w_mle, sim_max_lnL = estimate_global_MLE()
        sim_s_null, sim_lnL_null = maximize_lnL_fixed_w(w_null)
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



