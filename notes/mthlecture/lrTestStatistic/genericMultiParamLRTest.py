#!/usr/bin/env python
import sys, os
from scipy import optimize
from math import pow, log, exp
from random import random


# we are going to use a summary that is a pair of numbers for each pair of males
#   it will represent:
#    (the number of bouts won by male 0, the number of bouts won by male 1)

############################################################################
# Begin model-specific initialization code
############################################################################
# This is a list of values to try for the numerical optimizer.  The length of 
#   this list is also used by some functions to determine the dimensionality
#   of the model
INITIAL_PARAMETER_GUESS = [.75, .75]
# This is a list of parameter names to show
#
PARAMETER_NAMES = ['p', 'w']
# we expect the number of parameters to be the same in both lists.
assert(len(INITIAL_PARAMETER_GUESS) == len(PARAMETER_NAMES)) 
############################################################################
# End model-specific initialization code
############################################################################

def parse_data(input):
    line_source = iter(input)
    line_source.next() # this skips the first line

    # create an empty list of data points.
    full_data = []
    for n, line in enumerate(line_source):
        # This skips blank lines
        if not line.strip():
            continue 
        ############################################################################
        # Begin model-specific data-parsing code
        ############################################################################
        # We are expecting two integers separated by whitespace

        # this splits the words
        #
        words = line.strip().split() 
        # The try...except block checks for errors. Any errors will trigger the
        #   "except" block of code.  We'll just exit the program with a
        #   message that tells us what line of the file was problematic.
        #
        try:
            assert(len(words) == 2)
            num_0_wins = int(words[0])
            num_1_wins = int(words[1])
            assert(num_0_wins >= 0)
            assert(num_1_wins >= 0)
        except:
            sys.exit("Expecting each line to have 2 non-negative integers, but line " + str(n+1) + " does not.")

        # the data for each trial is the pair of numbers: how many times male 0
        #   won and how many times male 1 won.
        #
        data_for_trial = (num_0_wins, num_1_wins)

        # add the data for this trial        
        # 
        full_data.append(data_for_trial)
        ############################################################################
        # End model-specific data-parsing code
        ############################################################################
    return full_data

def ln_likelihood(the_data, param_list):
    '''Calculates the log-likelihood of the parameter values in `param_list`
    based on `the_data`
    '''
    ############################################################################
    # Begin model-specific log-likelihood code
    ############################################################################

    p_strong, w = param_list
    ###########################################################################
    # Return -inf if the parameters are out of range.  This will keep the optimizer
    #  from returning "illegal" values of the parameters.
    #
    if w != w or p_strong != p_strong:
        return float('-inf')
    if (w < .5) or (w > 1.0) or (p_strong < 0.0) or (p_strong > 1.0):
        return float('-inf')

    p_SS = p_strong * p_strong
    p_SW = p_strong * (1 - p_strong)
    p_WS = p_SW
    p_WW = (1-p_strong)*(1 - p_strong)

    # Calculate the probability that the males are evenly matched
    #
    p_even = p_WW + p_SS

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
        # Each datum is a count of the number of bouts won by male 0 (the 0-th
        #   element of the datum object), and the number of bouts won by
        #   male #1 (the element of the datum object at index 1).
        #
        n0_wins = datum[0]
        n1_wins  = datum[1]

        # Now we calculate the probability of seeing the data conditional on
        #   each of the pairing scenarios  (even, strong/weak, and weak/strong).
        #
        p_datum_if_even = pow(prob0_even, n0_wins) * pow(prob1_even, n1_wins)
        p_datum_if_SW = pow(prob0_SW, n0_wins) * pow(prob1_SW, n1_wins)
        p_datum_if_WS = pow(prob0_WS, n0_wins) * pow(prob1_WS, n1_wins)

        # By the law of total probability, the probability of this datum is
        #   simply the sum of the probability under each scenario multiplied by
        #   the probability that the scenario would occur
        #
        p_datum = (p_even*p_datum_if_even
                   + p_SW*p_datum_if_SW
                   + p_WS*p_datum_if_WS )

        # To avoid taking the log of 0.0, we'll return -infinity for the
        #   log-likelihood of any scenario that is incompatible with a datum
        #
        if p_datum <= 0.0:
            return float('-inf')

        # This is where we add the log-likelihood for this datum to the total
        #   for the whole data set.
        #
        ln_l = ln_l + log(p_datum)

    # This is how we send the result back to the function that called this
    #   function.  SciPy's numerical optimization code will use the values
    #   we return to try to find optimal values of p_strong and w by
    #   repeatedly trying out a lot of plausible combinations.
    #
    return ln_l
    ############################################################################
    # End model-specific log-likelihood code
    ############################################################################


def simulate_data(template_data, params):
    '''Simulate a data set of the same size as `template_data` but under a 
    model described by the parameter values in `params`
    '''
    ############################################################################
    # Begin model-specific simulation code
    ############################################################################
    # For our model the params are s and w, so we can unpack those from 
    #   the list of parameters
    s, w = params

    p_SS = s*s
    p_SW = s*(1 - s)
    p_WS = (1 - s)*s
    p_WW = (1 - s)*(1 - s)
    p_even = p_SS + p_WW


    sim_data_set = []

    for datum in template_data:
        n_bouts = sum(datum)

        # randomly pick the type of pairing the we have 'EVEN', 'SW' or 'WS'
        rand_match_p = random()
        if rand_match_p < p_even:
            match_type = 'EVEN'
        else:
            if rand_match_p < p_even + p_SW:
                match_type = 'SW'
            else:
                match_type = 'WS'

        # determine the probability that male 0 wins each bout.
        if match_type == 'EVEN':
            p_zero_wins = 0.5
        if match_type == 'WS':
            p_zero_wins = w
        if match_type == 'SW':
            p_zero_wins = 1 - w

        # start out with no bouts won by either, then we are going to
        #   simulate the result of n_bouts
        n0_won = 0
        n1_won = 0
        for bout in range(n_bouts):
            if random() < p_zero_wins:
                n0_won = n0_won + 1
            else:
                n1_won = n1_won + 1

        # add this simulated outcome to our simulated data set
        sim_datum = (n0_won, n1_won)
        sim_data_set.append(sim_datum)

    return sim_data_set
    ############################################################################
    # End model-specific simulation code
    ############################################################################





def calc_global_ml_solution(data):
    '''Uses SciPy's  optimize.fmin to find the mle. Starts the search at
    s = 0.75, and w = 0.75

    Returns the (s_mle, w_mle, ln_L)
    '''

    def scipy_ln_likelihood(x):
        '''SciPy minimizes functions. We want to maximize the likelihood. This
        function adapts our ln_likelihood function to the minimization context
        by returning the negative log-likelihood.
    
        We use this function with SciPy's minimization routine (minimizing the
        negative log-likelihood will maximize the log-likelihood).
        '''
        return -ln_likelihood(data, x)

    x0 = INITIAL_PARAMETER_GUESS
    solution = optimize.fmin(scipy_ln_likelihood, x0, xtol=1e-8, disp=False)
    solution = list(solution)
    ln_l = -scipy_ln_likelihood(solution)
    solution.append(ln_l)
    return solution

def calc_null_ml_solution(data, param_constraints):
    '''This function allows us to optimize those parameters that are set to 
    None in the list `param_constraints`. Other parameters are forced to assume
    the value listed in param_constraints.
    '''
    x0 = []
    adaptor_list = []
    for i, val in enumerate(INITIAL_PARAMETER_GUESS):
        if i >= len(param_constraints) or param_constraints[i] == None:
            x0.append(val)
            adaptor_list.append(None)
        else:
            adaptor_list.append(param_constraints[i])
    
    def intercalate_constraints(x):
        p = []
        x_index = 0
        for el in adaptor_list:
            if el is None:
                p.append(x[x_index])
                x_index = x_index + 1
            else:
                p.append(el)
        return p
        

    def constrained_scipy_ln_likelihood(x):
        all_params = intercalate_constraints(x)
        return -ln_likelihood(data, all_params)
        
    if len(x0) > 0:
        solution = optimize.fmin(constrained_scipy_ln_likelihood,
                                 x0,
                                 xtol=1e-8,
                                 disp=False)
        all_params =  all_params = intercalate_constraints(solution)
    else:
        all_params = list(param_constraints)
    ln_l = ln_likelihood(data, all_params)
    all_params = list(all_params)
    all_params.append(ln_l)
    return all_params




def calc_lrt_statistic(data, null_params):
    '''Returns (log-likelihood ratio test statistic,
                list of MLEs of all parameters and lnL at the global ML point,
                a list of the MLEs of all parameters and lnL under the null)
        
    for `data` with the null hypothesis being the parameters constrained
    to be at the values specified in the list `null_params`
    '''
    # First we calculate the global and null solutions
    #
    global_mle = calc_global_ml_solution(data)
    null_mle = calc_null_ml_solution(data, null_params)

    # the log-likelihood is returned as the last element of the list by these
    #   functions.  We can access this element by referring to element -1 using
    #   the list indexing syntax - which is [] braces:
    #
    global_max_ln_l = global_mle[-1]
    null_max_ln_l = null_mle[-1]

    # Now we can calculate the likelihood ratio test statistic, and return
    #   it as well as the global and null solutions
    #
    lrt = 2*(null_max_ln_l - global_max_ln_l)
    return lrt, global_mle, null_mle
    

def print_help():
    num_args_expected = 2 + len(INITIAL_PARAMETER_GUESS)
    output_stream = sys.stdout
    output_stream.write('Expecting ' + str(num_args_expected) + ''' arguments.
    The first argument should be a filename,
    The last argument should be the number of parametric bootstrapping replicates,
    The intervening arguments should be the values of the parameters in the null
        hypothesis (or 'None' to indicate that the parameter is not constrained
        in the null hypothesis).
    
    The order of the parameters is in this constraint statement is:
        ''')
    for p in PARAMETER_NAMES:
        output_stream.write(p + ' ')
    assert len(INITIAL_PARAMETER_GUESS) > 0
    if len(INITIAL_PARAMETER_GUESS) == 1:
        c_name = PARAMETER_NAMES[0]
        c_val = str(INITIAL_PARAMETER_GUESS[0])
        parg_list[c_val]
    else:
        c_name = PARAMETER_NAMES[1]
        c_val = str(INITIAL_PARAMETER_GUESS[1])
        parg_list = ['None'] * len(INITIAL_PARAMETER_GUESS)
        parg_list[1] = str(INITIAL_PARAMETER_GUESS[1])

    output_stream.write('''

    So if you want to perform 1000 simulations, your data is in the file "data.txt" 
        and you want to test the hypothesis that:
            ''' + c_name + ' = ' + c_val + '''
        then you would use the arguments:

data.txt ''' + ' '.join(parg_list) +  ''' 1000

''')        
        
    
if __name__ == '__main__':
    # user-interface and sanity checking...

    # we 
    import sys
    arguments = sys.argv[1:]
    if len(sys.argv) < 3 + len(INITIAL_PARAMETER_GUESS):
        print_help()
        sys.exit(1)

    filename = arguments[0]
    if not os.path.exists(filename):
        sys.exit('The file "' + filename + '" does not exist')
    inp = open(filename, 'rU') # open the file name for reading (the 'rU' argument)

    real_data = parse_data(inp)

    # The number of simulations is the last parameter...
    #
    n_sims = int(arguments[-1])
    
    # The "middle" arguments will be constraints on the parameters.
    # We'll store the number for every argument that is an argument.  If the
    #   argument can't be turned into a number then we'll reach the except
    #   block.  In this case we'll insert None into the list to indicate that
    #   the parameter is not constrained.
    #
    null_params = []
    for arg in arguments[1:-1]:  # this walks through all arguments except the first and last
        try:
            p = float(arg)
            null_params.append(p)
        except:
            null_params.append(None)
    print "null_params =", null_params
    # Call a function to maximize the log-likelihood under the unconstrained
    #   and null conditions.  This returns the LRT statistic and the 
    #   parameter estimates as lists
    #
    lrt, mle_list, null_mle_list = calc_lrt_statistic(real_data, null_params)

    # We can "unpack" the list into 3 separate variables to make it easier to
    #   report
    ln_l = mle_list[-1]
    ln_l_null = null_mle_list[-1]

    for n, param_name in enumerate(PARAMETER_NAMES):
        print "MLE of", p, "=", mle_list[n]
    print "lnL at MLEs =", ln_l
    print "L at MLEs =", exp(ln_l)

    for n, param_name in enumerate(PARAMETER_NAMES):
        v = null_mle_list[n]
        if null_params[n] is None:
            print "Under the null, the MLE of", param_name, "=", v
        else:
            print "Under the null, ", param_name, " is constrained to be", v
    print "ln_l at null =", ln_l_null
    print "L at null =", exp(ln_l_null)

    print
    print "2* log-likelihood ratio = ", lrt
    print



    # Do parametric bootstrapping to produce the null distribution of the LRT statistic
    #
    if n_sims < 1:
        sys.exit(0)
    print "Generating null distribution of LRT..."

    # We'll write the simulated LRT values to the "standard error stream" which
    #   is written to the screen when we run the program from a terminal.
    # We could use:
    #       pboot_out = open("param_boot.txt", "w")
    # if we wanted to write the results to a file called "param_boot.txt"
    #
    pboot_out = sys.stderr
    pboot_out.write("rep\tlrt\n")

    # null_dist will be a list that holds all of the simulated LRT values. We
    # use [] to create an empty list.
    #
    null_dist = []

    # a "for loop" will repeat the following instructions n_sims times
    #
    sim_params = null_mle_list[:-1]
    for i in range(n_sims):
        # This simulates a data set assuming that the parameters are at the
        #   values that the take under the null hypothesis (since we want the
        #   null distribution of the test statistic).
        #
        sim_data = simulate_data(real_data, sim_params)

        # Calculate the LRT on the simulated data using the same functions that
        #   we used when we analyzed the real data
        #
        sim_lrt, sim_mle_list, sim_null = calc_lrt_statistic(sim_data, null_params)

        # Add the simulated LRT to our null distribution
        #
        null_dist.append(sim_lrt)

        # Write the value to the output stream.  The str() function converts
        #   numbers to strings so that they can be written to the output stream
        #
        pboot_out.write(str(i + 1))
        pboot_out.write('\t')
        pboot_out.write(str(sim_lrt))
        pboot_out.write('\n')

    # We want the most extreme (negative) values of the LRT from the simulations
    #   if we sort the list, then these values will be at the front of the list
    #   (available by indexing the list with small numbers)
    #
    null_dist.sort()

    # We can report the value of the LRT that is smaller than 95% of the simulated
    #   values...
    #
    n_for_p_point05 = int(0.05*n_sims)
    print "5% critical value is approx =", null_dist[n_for_p_point05]

    # And we can calculate the P-value for our data by counting the number of
    #   simulated replicates that are more extreme than our observed data.
    #   We do this by starting a counter at 0, walking through the stored
    #   null distribution, and adding 1 to our counter every time we see a
    #   simulated LRT that is more extreme than the "real" lrt.
    #
    n_more_extreme = 0
    for v in null_dist:
        if v <= lrt:
            n_more_extreme = n_more_extreme + 1
        else:
            break
    # Then we express this count of the number more extreme as a probability
    #    that a simulated value would be more extreme than the real data.
    #
    print "Approx P-value =", n_more_extreme/float(n_sims)

