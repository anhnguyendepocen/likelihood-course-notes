#!/usr/bin/env python

from scipy import optimize
from math import pow, log, exp
from random import random



real_data = [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1,
1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]

############################################################################
# Begin model-specific initialization code
############################################################################
# This is a list of values to try for the numerical optimizer.  The length of 
#   this list is also used by some functions to determine the dimensionality
#   of the model
initial_parameter_guess = [.75, .75]

# This is a list of parameter names to show. Modify this based on what order
#   you want to use for the parameter list. It does not matter which order
#   you choose, but you need to know what the program thinks of as the first
#   and second parameters so that the output will make sense and your code
#   that unpacks the parameter list will know whether the first parameter is
#   the recapture probability or the probability of an individual being asymmetric
#   You should put the names in quotes, so the line would look like:
#  
#   parameter_names = ['mu', 'sigma'] 
#   
#   for inference under the normal model.
#
parameter_names = ['a', 'r']

# we expect the number of parameters to be the same in both lists. This is a 
#   sanity check that helps us see if we made a mistake.
#
assert(len(initial_parameter_guess) == len(parameter_names)) 
############################################################################
# End model-specific initialization code
############################################################################

verbose = False

def ln_likelihood(the_data, param_list):
    '''Calculates the log-likelihood of the parameter values in `param_list`
    based on `the_data`
    '''
    if verbose:
        sys.stderr.write('param = ' + str(param_list) + '\n')
    ############################################################################
    # Begin model-specific log-likelihood code
    ############################################################################
    first_param, second_param = param_list
    a = first_param
    r = second_param

    # do our "sanity-checking" to make sure that we are in the legal range of 
    #   the parameters.
    #
    if first_param != first_param or second_param != second_param:
        return float('-inf')
    if (first_param < 0.0) or (first_param > 1.0):
        return float('-inf')
    if (second_param < 0.0) or (second_param > 1.0):
        return float('-inf')


    ln_l = 0.0
    
    # In our model, the first individual can't be a recapture, so this 
    #   probability calculation is "special" Here we divide the data
    #   into the first observation and a list that contains the remaining
    #   data points.
    #
    first_observation = the_data[0]
    remaining_observations = the_data[1:]
    
    if first_observation == 1:
        p_datum = a
    else:
        p_datum = 1 - a
    
    ln_l = ln_l + log(p_datum)

    # In our model of recapturing individuals, the previous observation
    #   is relevant to our current observation.  So we'll keep track of the
    #   previous observation as the variable `previous`
    #
    previous = first_observation

    # Now we have to consider the rest of the data points
    #
    for datum in remaining_observations:
        if datum == 1:
            p_datum = (1-r)*a
        else:
            p_datum = (1-r)*(1-a)
        if datum == previous:
            p_datum = p_datum + r


        # Add it to the log-likelihood
        #
        ln_l = ln_l + log(p_datum)
        # Now we update the variable `previous` to reflect the latest outcome
        #   so that in the next execution of this loop it really will refer
        #   to the previous value
        #
        previous = datum


    if verbose:
        sys.stderr.write('ln_l = ' + str(ln_l) + '\n')
    # This is how we send the result back to the function that called this
    #   function.  
    return ln_l
    ############################################################################
    # End model-specific log-likelihood code
    ############################################################################


def simulate_data(template_data, param_list):
    '''Simulate a data set of the same size as `template_data` but under a 
    model described by the parameter values in `params`
    '''
    ############################################################################
    # Begin model-specific simulation code
    ############################################################################

    n_observations = len(template_data)

    a, r = param_list

    if random() < a:
        first_datum = 1
    else:
        first_datum = 0
    sim_data_set = [first_datum]

    previous_datum = first_datum
    for sample_index in range(n_observations - 1):
        # We will need to use some pseudo-random numbers generated from
        #   a Uniform distribution between 0 and 1 as the source
        #   of variability for our simulation.  
        # This is how we generate such a number and store it as `u` in Python
        #
        u = random() 
        # Each realization that you simulate may depend on the previous 
        #   value simulated, so you will have to add some more logic here
        #   to actually generate a data point under the model.
        # Each datum that you simulate should be a 0 or a 1 (because that is
        #   how we are recording the data in our real data set.
        #
        if u < r:
            sim_datum = previous_datum
        else:
            if random() < a:
                sim_datum = 1
            else:
                sim_datum = 0
        sim_data_set.append(sim_datum)
        # This updates the `previous_datum` variable so that it is always accurate
        #   when we execute a block of code in this "for loop"
        #
        previous_datum = sim_datum

    return sim_data_set
    ############################################################################
    # End model-specific simulation code
    ############################################################################



################################################################################
#
# YOU SHOULD NOT HAVE TO MODIFY THE CODE BELOW THIS POINT !!!
################################################################################

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

    x0 = initial_parameter_guess
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
    for i, val in enumerate(initial_parameter_guess):
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
    num_args_expected = 1 + len(initial_parameter_guess)
    output_stream = sys.stdout
    output_stream.write('Expecting ' + str(num_args_expected) + ''' arguments.
    The last argument should be the number of parametric bootstrapping replicates,
    The first arguments should be the values of the parameters in the null
        hypothesis (or 'None' to indicate that the parameter is not constrained
        in the null hypothesis).
    
    The order of the parameters is in this constraint statement is:
        ''')
    for p in parameter_names:
        output_stream.write(p + ' ')
    assert len(initial_parameter_guess) > 0
    if len(initial_parameter_guess) == 1:
        c_name = parameter_names[0]
        c_val = str(initial_parameter_guess[0])
        parg_list[c_val]
    else:
        c_name = parameter_names[1]
        c_val = str(initial_parameter_guess[1])
        parg_list = ['None'] * len(initial_parameter_guess)
        parg_list[1] = str(initial_parameter_guess[1])

    output_stream.write('''

    So if you want to perform 1000 simulations and you want to test the
        hypothesis that:
            ''' + c_name + ' = ' + c_val + '''
        then you would use the arguments:

''' + ' '.join(parg_list) +  ''' 1000

''')        
        
    
if __name__ == '__main__':
    # user-interface and sanity checking...

    # we 
    import sys
    
    arguments = sys.argv[1:]
    if len(sys.argv) < 2 + len(initial_parameter_guess):
        print_help()
        sys.exit(1)

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
    for arg in arguments[:-1]:  # this walks through all arguments except the last
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

    for n, param_name in enumerate(parameter_names):
        print "MLE of", param_name, "=", mle_list[n]
    print "lnL at MLEs =", ln_l
    print "L at MLEs =", exp(ln_l)

    for n, param_name in enumerate(parameter_names):
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

