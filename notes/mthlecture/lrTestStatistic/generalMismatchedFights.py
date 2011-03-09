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

## Data from 5 sampling schemes: 2 bouts/pair, 3 bouts/pair, ... 7 bouts/pair
##  for each sampling scheme we have 20 observations
#  real_data = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (2, 0), (2, 0), (2, 0), (1, 1), (0, 2), 
#               (0, 2), (2, 0), (0, 2), (1, 1), (1, 1), (2, 0), (0, 2), (1, 1), (1, 1), (1, 1), 
#               (3, 0), (2, 1), (0, 3), (1, 2), (1, 2), (0, 3), (0, 3), (2, 1), (0, 3), (1, 2), 
#               (2, 1), (1, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 1), (2, 1), (0, 3), (1, 2), 
#               (0, 4), (3, 1), (0, 4), (3, 1), (1, 3), (2, 2), (1, 3), (4, 0), (1, 3), (2, 2),
#               (1, 3), (3, 1), (0, 4), (2, 2), (1, 3), (0, 4), (1, 3), (4, 0), (2, 2), (3, 1),
#               (2, 3), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3), (2, 3), (0, 5), (1, 4), (2, 3),
#               (4, 1), (4, 1), (5, 0), (1, 4), (1, 4), (2, 3), (3, 2), (2, 3), (2, 3), (5, 0),
#               (3, 3), (1, 5), (0, 6), (4, 2), (4, 2), (4, 2), (0, 6), (4, 2), (1, 5), (3, 3), 
#               (1, 5), (3, 3), (4, 2), (5, 1), (3, 3), (2, 4), (1, 5), (5, 1), (5, 1), (3, 3),
#               (4, 3), (7, 0), (6, 1), (6, 1), (3, 4), (3, 4), (4, 3), (1, 6), (4, 3), (5, 2),
#               (2, 5), (3, 4), (2, 5), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (2, 5), (1, 6)]

def ln_likelihood(prob_strong, w):
    '''Calculates the log-likelihood of the parameter value {prob_strong, w}
    based on the_data (which is a "global" variable).
    '''
    global the_data
    ###########################################################################
    # Return -inf if the parameters are out of range.  This will keep the optimizer
    #  from returning "illegal" values of the parameters.
    #
    if w != w or prob_strong != prob_strong:
        return float('-inf')
    if (w < .5) or (w > 1.0) or (prob_strong < 0.0) or (prob_strong > 1.0):
        return float('-inf')
    
    prob_SS = prob_strong*prob_strong
    prob_SW = prob_strong*(1 - prob_strong)
    prob_WS = prob_SW
    prob_WW = (1-prob_strong)*(1 - prob_strong)

    # Calculate the probability that the males are evenly matched
    #
    prob_even = prob_WW + prob_SS

    # Calculate the probability of an outcome of a bout (0 or 1) conditional
    #    on what type of pairing we have:
    #
    prob0_even = 0.5
    prob1_even = 1 - prob0_even

    prob0_SW = w
    prob1_SW = 1 - prob0_SW

    prob0_WS = 1 - w
    prob1_WS = 1 - prob0_WS

    # initialize the ln_l to 0, we want this to be the sum of the log-likelihood
    #   for each datum, so we'll start at 0 and add to it.
    #
    ln_l = 0.0
    for datum in the_data:
        # Each datum is a count of the number of bouts won by male 0 (the 0-th element
        #   of the datum object), and the number of bouts won by male #1 (the element
        #   of the datum object at index 1).
        #
        num0_wins = datum[0]
        num1_wins  = datum[1]
        
        # Now we calculate the probability of seeing the data conditional on
        #   each of the pairing scenarios  (even, strong/weak, and weak/strong).
        #
        prob_datum_if_even = (prob0_even**num0_wins) * (prob1_even**num1_wins)
        prob_datum_if_SW = (prob0_SW**num0_wins) * (prob1_SW**num1_wins)
        prob_datum_if_WS = (prob0_WS**num0_wins) * (prob1_WS**num1_wins)

        # By the law of total probability, the probability of this datum is
        #   simply the sum of the probability under each scenario multiplied by
        #   the probability that the scenario would occur
        #
        prob_datum = (  prob_even*prob_datum_if_even
                      + prob_SW*prob_datum_if_SW
                      + prob_WS*prob_datum_if_WS )

        # To avoid taking the log of 0.0, we'll return -infinity for the 
        #   log-likelihood of any scenario that is incompatible with a datum
        #
        if prob_datum <= 0.0:
            return float('-inf')

        # This is where we add the log-likelihood for this datum to the total
        #   for the whole data set.
        #
        ln_l = ln_l + math.log(prob_datum)

    # This is how we send the result back to the function that called this
    #   function.  SciPy's numerical optimization code will use the values
    #   we return to try to find optimal values of prob_strong and w by
    #   repeatedly trying out a lot of plausible combinations.
    #
    return ln_l

def scipy_ln_likelihood(x):
    '''SciPy minimizes functions. We want to maximize the likelihood. This 
    function adapts our ln_likelihood function to the minimization context
    by returning the negative log-likelihood.

    We use this function with SciPy's minimization routine (minimizing the 
    negative log-likelihood will maximize the log-likelihood).
    '''
    raw = ln_likelihood(x[0], x[1])
    return -raw

def estimate_global_MLE():
    '''Uses SciPy's  optimize.fmin to find the MLE. Starts the search at
    s = 0.75, and w = 0.75

    Returns the (s_mle, w_mle, ln_L)
    '''
    x0 = [.75, .75]
    param = optimize.fmin(scipy_ln_likelihood, x0, xtol=1e-8, disp=False)
    return param[0], param[1], -scipy_ln_likelihood(param)

def maximize_lnL_fixed_w(w):
    '''This function allows us to optimize prob_strong while keeping
    w fixed.
    '''
    x0 = [.75]
    def one_param_ln_likelihood(x):
        return scipy_ln_likelihood([x[0], w])
    param = optimize.fmin(one_param_ln_likelihood, x0, xtol=1e-8, disp=False)
    return param[0], -one_param_ln_likelihood(param)

def simulate_data(s, w):
    global real_data
    template_data = real_data

    prob_SS = s*s
    prob_SW = s*(1 - s)
    prob_WS = (1 - s)*s
    prob_WW = (1 - s)*(1 - s)
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
    the_data = real_data
    w_null = float(sys.argv[1])
    num_sims = int(sys.argv[2])
    
    # Calculate and report the MLEs and log-likelihood.
    #
    prob_strong_MLE, w_MLE, lnL = estimate_global_MLE()

    print "MLE of prob_strong =", prob_strong_MLE
    print "MLE of w =", w_MLE
    print "lnL at MLEs =", lnL
    print "L at MLEs =", math.exp(lnL)
    print

    # Calculate the MLEs and log-likelihood for the null value of w.
    #
    prob_strong_null, lnL_null = maximize_lnL_fixed_w(w_null)

    print "MLE of prob_strong at null w =", prob_strong_null
    print "null of w =", w_null
    print "lnL at null =", lnL_null
    print "L at null =", math.exp(lnL_null)
    print
    
    # Calculate the log likelihood ratio test statistic.
    #
    lrt = 2*(lnL_null - lnL)

    print "2* log-likelihood ratio = ", lrt
    print



    # Do parametric bootstrapping to produce the null distribution of the LRT statistic
    #
    if num_sims < 1:
        sys.exit(0)
    print "Generating null distribution of LRT..."

    # For convenience, we'll write the simulated LRT values to standard error...
    #
    sys.stderr.write("rep\tlrt\n")

    # null_dist will be a list that holds all of the simulated LRT values. We
    # use [] to create an empty list.
    #
    null_dist = []

    # a "for loop" will repeat the following instructions num_sims times
    #
    for i in range(num_sims):
        # This simulates a data set assuming that the parameters are at the 
        #   values that the take under the null hypothesis (since we want the
        #   null distribution of the test statistic).
        #
        sim_data = simulate_data(prob_strong_null, w_null)
        # Update the global variable "the_data" to reflect the simulation
        #   so that the likelihood calculations will use the simulated data
        #
        the_data = sim_data
        
        
        # Calculate the LRT on the simulated data using the same functions that
        #   we used when we analyzed the real data
        #
        sim_s_mle, sim_w_mle, sim_max_lnL = estimate_global_MLE()
        sim_s_null, sim_lnL_null = maximize_lnL_fixed_w(w_null)
        sim_lrt = 2*(sim_lnL_null - sim_max_lnL)

        # Add the simulated LRT to our null distribution
        #
        null_dist.append(sim_lrt)

        # Write the value to the screen.
        #
        sys.stderr.write(str(i + 1))
        sys.stderr.write('\t')
        sys.stderr.write(str(sim_lrt))
        sys.stderr.write('\n')
    
    # We want the most extreme (negative) values of the LRT from the simulations
    #   if we sort the list, then these values will be at the front of the list
    #   (available by indexing the list with small numbers)
    #
    null_dist.sort()
    
    # We can report the value of the LRT that is smaller than 95% of the simulated
    #   values...
    #
    num_for_p_point05 = int(0.05*num_sims)
    print "5% critical value is approx =", null_dist[num_for_p_point05]
    
    # And we can calculate the P-value for our data by counting the number of 
    #   simulated replicates that are more extreme than our observed data.
    #   We do this by starting a counter at 0, walking through the stored 
    #   null distribution, and adding 1 to our counter every time we see a
    #   simulated LRT that is more extreme than the "real" lrt.
    #
    num_more_extreme = 0
    for v in null_dist:
        if v < lrt:
            num_more_extreme = num_more_extreme + 1
        else:
            break
    # Then we express this count of the number more extreme as a probability
    #    that a simulated value would be more extreme than the real data.
    #
    print "Approx P-value =", num_more_extreme/float(num_sims)

