#!/usr/bin/env python

from scipy import optimize
from math import pow, log, exp, sqrt
from random import random, normalvariate
import copy
import numpy

verbose = False
  
USE_FMIN = False



def copy_data_from_dict_to_mat(treatment_dict_pair, treat_id, matrix, status_stream):
    treat_inds = treatment_dict_pair[treat_id]
    tk = treat_inds.keys()
    tk.sort()
    status_stream.write( "        " + str(len(tk)) + " individuals in treatment " + str(treat_id) + ":\n")
    row = matrix[treat_id]
    for j, ind_id in enumerate(tk):
        value = treat_inds[ind_id]
        status_stream.write( "           Indiv. index=" + str(j) + " (id in datafile = " + str(ind_id) + ") value = " + str(value) + "\n")
        row.append(value)


def read_data(filepath):
    '''Reads filepath as a tab-separated csv file and returns a 3dimensional data matrix.'''
    import os
    import csv
    import itertools
    if not os.path.exists(filepath):
        raise ValueError('The file "' + filepath + '" does not exist')
    
    MAX_TREATMENT_VALUE = 1
    # Here we create a csv reader and tell it that a tab (\t) is the column delimiter
    entries = csv.reader(open(filepath, 'rb'), delimiter='\t')

    # Here we check that file has the headers that we exect
    first_row = entries.next()
    expected_headers = ['family', 'treatment', 'individual', 'trait']
    for got, expected in itertools.izip(first_row, expected_headers):
        if got.lower().strip() != expected:
            raise ValueError('Error reading "' + filepath + '": expecting a column labelled "' + expected + '", but found "' + got + '" instead.')

    # It is not too hard to have this loop put the data in the right spot in the
    #   matrix, so that the data file can be somwhat flexible.

    by_family = {}
    for n, row in enumerate(entries):
        fam_id, treatment_id, ind_id, value = row
        try:
            fam_id = int(fam_id)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for family ID, but got ' + str(fam_id))
        try:
            ind_id = int(ind_id)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for individual ID, but got ' + str(ind_id))
        try:
            treatment_id = int(treatment_id)
            assert(treatment_id >= 0 and treatment_id <= MAX_TREATMENT_VALUE)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer in the range [0, ' + str(MAX_TREATMENT_VALUE) + '] for the treatment ID, but got ' + str(treatment_id))
        try:
            value = float(value)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an number for the trait value, but got ' + str(ind_id))
        # a new family corresponds to an empty dictionary of individuals
        #   for the 0, and 1 treatment. So we can create a list with two empty
        #   dictionaries as the "blank" entry.
        empty_family_entry = [{} for i in range(MAX_TREATMENT_VALUE + 1)]
        fam_array = by_family.setdefault(fam_id, empty_family_entry)
        # now we grab the appropriate one for this treatment
        fam_treatment_dict = fam_array[treatment_id]
        if ind_id in fam_treatment_dict:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": a trait value for an individual with family=' + str(fam_id) + ' treatment=' + str(treatment_id) + ' ind=' + ind_id + ' has already been encountered!')
        fam_treatment_dict[ind_id] = value

    fam_keys = by_family.keys()
    fam_keys.sort()

    # Now we create the data matrix. We'll interpret the values in the first
    #   3 columns of the data as the "indices" in our 3D array.  Python uses 
    #   indexing that starts with 0, so the first element should be from
    #   family 0, diet treatment 0, and indiv 0. 
    # This is more complex than you may expect because we'd like to keep the
    #   internal indexing in the same order as the indexing in the data file,
    #   but it would also be nice if we could remove families from the data
    #   file without being forced to renumber everything.
    # So, as we create the matrix, we'll crunch the entries to skip over 
    #   absent families and we'll be verbose in pointing out how the internal
    #   indexing corresponds to the indexing in the input file...
    status_stream = sys.stdout
    status_stream.write("Data read for " + str(len(fam_keys)) + " families...\n")
    empty_family_matrix = [[], []]
    data_m = [copy.deepcopy(empty_family_matrix) for i in fam_keys]
    for i, fam_id in enumerate(fam_keys):
        treatment_dict_pair = by_family[fam_id]
        status_stream.write("    Family index=" + str(i) + " (id in datafile = " + str(fam_id) + "):\n")
        mat = data_m[i]
        for j in xrange(MAX_TREATMENT_VALUE + 1):
            copy_data_from_dict_to_mat(treatment_dict_pair, j, mat, status_stream)
    status_stream.write("Data as a python list:\n" + repr(data_m) + "\n")
    return data_m




############################################################################
# Begin model-specific initialization code
############################################################################
# This is a list of values to try for the numerical optimizer.  The length of 
#   this list is also used by some functions to determine the dimensionality
#   of the model
initial_parameter_guess = [-.5, .5, 0.2, 0.2, 0.2]
MIN_VARIANCE = 1e-6
parameter_bounds = [(None, None), (None, None), (0.0, None), (0.0, None), (MIN_VARIANCE, None), ]
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
parameter_names = ['alpha0', 'alpha1', 'varFam', 'varInteraction', 'varError']

# we expect the number of parameters to be the same in both lists. This is a 
#   sanity check that helps us see if we made a mistake.
#
assert(len(initial_parameter_guess) == len(parameter_names)) 
############################################################################
# End model-specific initialization code
############################################################################




def block_diag(*arrs):
    """Create a new diagonal matrix from the provided arrays.

    Parameters
    ----------
    a, b, c, ... : ndarray
        Input arrays.

    Returns
    -------
    D : ndarray
        Array with a, b, c, ... on the diagonal.

    Code from http://mail.scipy.org/pipermail/scipy-user/attachments/20090520/f30c0928/attachment.obj
    posted by Stefan van der Walt http://mail.scipy.org/pipermail/scipy-user/2009-May/021101.html
    """
    arrs = [numpy.asarray(a) for a in arrs]
    shapes = numpy.array([a.shape for a in arrs])
    out = numpy.zeros(numpy.sum(shapes, axis=0))

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def invert_block_diagonal(block_list):
    inv_block_list = []
    for block in block_list:
        inv_block = numpy.linalg.inv(block)
        inv_block_list.append(inv_block)
    return block_diag(*inv_block_list)

def determinant_block_diagonal(block_list):
    det = 1.0
    for block in block_list:
        det = det* numpy.linalg.det(block)
    return det
    
def ln_likelihood(the_data, param_list):
    '''Calculates the log-likelihood of the parameter values in `param_list`
    based on `the_data`
    '''
    if verbose:
        sys.stderr.write('param = ' + str(param_list) + '\n')
    ############################################################################
    # Begin model-specific log-likelihood code
    ############################################################################
    first_p, second_p, third_p, fourth_p, fifth_p = param_list
    alpha0 = first_p
    alpha1 = second_p
    var_fam = third_p
    var_interaction = fourth_p
    var_error = fifth_p
    
    # do our "sanity-checking" to make sure that we are in the legal range of 
    #   the parameters.
    #
    for p in param_list:
        if p != p:
            return float('-inf')
    if var_fam < 0.0:
        return float('-inf')
    if var_interaction < 0.0:
        return float('-inf')
    if var_error < 0.0:
        return float('-inf')

    expected_values = []
    observed_values = []

    for fam_index, family_data in enumerate(the_data):
        for treatment_index, indiv_data in enumerate(family_data):
            # Every individual from the same family/treatment combination will
            #   have the same expected value, so we can calculate it once
            num_indiv = len(indiv_data)
            
            if treatment_index == 0:
                expected = alpha0
            else:
                expected = alpha1
        
            exp_this_fam_treat = [expected]*num_indiv
            expected_values.extend(exp_this_fam_treat)
            
            observed_values.extend(indiv_data)
    

    non_zero_covariance_blocks = list()

    var_individ = var_fam + var_interaction + var_error
    covar_same_fam_treatment = var_fam + var_interaction
    covar_same_fam = var_fam
    
    for fam_index, family_data in enumerate(the_data):
        treatment_0 = family_data[0]
        num_treatment_0 = len(treatment_0)
        treatment_1 = family_data[1]
        num_treatment_1 = len(treatment_1)
        num_in_family = num_treatment_0 + num_treatment_1
        
        # make an empty covariance matrix for this family...
        empty_row = [None]*num_in_family
        family_cov = [copy.copy(empty_row) for i in xrange(num_in_family)]
        for i in range(num_in_family):
            i_in_treat_0 = i < num_treatment_0
            for j in range(i, num_in_family):
                if i == j:
                    family_cov[i][i] = var_individ
                else:
                    j_in_treat_0 = j < num_treatment_0
                    both_treat_0 =  i_in_treat_0 and j_in_treat_0
                    both_treat_1 = (not i_in_treat_0) and (not j_in_treat_0)
                    if both_treat_0 or both_treat_1:
                        family_cov[i][j] = covar_same_fam_treatment
                        family_cov[j][i] = covar_same_fam_treatment
                    else:
                        family_cov[i][j] = var_fam
                        family_cov[j][i] = var_fam
        # append this non-zero part of the covariance matrix to a list
        #   that will be used to represent a block diagonal matrix for all 
        #   measurements.
        # first we'll convert it from a python list of lists to a numpy matrix
        numpy_fam_cov = numpy.matrix(family_cov)
        non_zero_covariance_blocks.append(numpy_fam_cov)

    inverse_var = invert_block_diagonal(non_zero_covariance_blocks)
    #print "inverse_var =", inverse_var
    #print "inverse_var.__class__ =", inverse_var.__class__
    #print "inverse_var.shape =", inverse_var.shape
    determinant = determinant_block_diagonal(non_zero_covariance_blocks)
    expected_values_column = numpy.matrix(expected_values)
    #print "expected_values =", expected_values_column
    #print "expected_values.shape =", expected_values_column.shape
    observed_values_column = numpy.matrix(observed_values)
    #print "observed_values =", observed_values_column
    residuals_column = expected_values_column - observed_values_column
    #print "residuals=", residuals_column
    #print "residuals.__class__=", residuals_column.__class__
    #print "residuals.shape =", residuals_column.shape
    resid_row = residuals_column.transpose()
    #print "transposed_resid=", resid_row
    vr = inverse_var*resid_row
    #print "vr = ",vr
    
    scaled_dev_sq = residuals_column*vr
    #print "scaled_dev_sq =",scaled_dev_sq
    ln_l = -0.5*(log(determinant) + float(scaled_dev_sq))

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

    first_p, second_p, third_p, fourth_p, fifth_p = param_list
    alpha0 = first_p
    alpha1 = second_p
    var_fam = third_p
    var_interaction = fourth_p
    var_error = fifth_p

    sd_fam = sqrt(var_fam)
    sd_interaction = sqrt(var_interaction)
    sd_error = sqrt(var_error)
    
    expected_values = [alpha0, alpha1]
    sim_family_list = []
    for template_fam in template_data:
        # draw a random family effect
        family_effect = normalvariate(0, sd_fam)
        sim_family = []
        for treatment_index in [0, 1]:
            # draw a random interaction effect
            interaction_effect = normalvariate(0, sd_interaction)
            
            # calculate the expected value of individual (given the treatment,
            #   family, and interaction effects
            treatment_expected = expected_values[treatment_index]
            fam_treat_expected = treatment_expected + family_effect + interaction_effect
            
            # Figure out how many individuals in this family x treatment group
            #
            temp_family_treatment = template_fam[treatment_index]
            num_in_fam_treatment = len(temp_family_treatment)
            # simulate that many values
            sim_value_list = []
            for i in range(num_in_fam_treatment):
                sim_value = normalvariate(fam_treat_expected, sd_error)
                sim_value_list.append(sim_value)
            # store these simulated values
            sim_family.append(sim_value_list)
        # store this family in the list of simulated families
        sim_family_list.append(sim_family)

    return sim_family_list
    ############################################################################
    # End model-specific simulation code
    ############################################################################



################################################################################
#
# YOU SHOULD NOT HAVE TO MODIFY THE CODE BELOW THIS POINT !!!
################################################################################

def calc_global_ml_solution(data):
    '''Uses SciPy's  optimize.fmin_l_bfgs_b to find the mle. Starts the search at
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
    if USE_FMIN:
        solution = optimize.fmin(scipy_ln_likelihood, 
                                          x0,
                                          xtol=1e-8,
                                          disp=False)
    else:
        opt_blob = optimize.fmin_l_bfgs_b(scipy_ln_likelihood, 
                                          x0,
                                          bounds=parameter_bounds,
                                          approx_grad=True,
                                          epsilon=1e-8,
                                          disp=False)
        solution = list(opt_blob[0])
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
    constr_bounds = []
    for i, val in enumerate(initial_parameter_guess):
        if i >= len(param_constraints) or param_constraints[i] == None:
            x0.append(val)
            constr_bounds.append(parameter_bounds[i])
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
            elif isinstance(el, list):
                assert(len(el) == 1)
                assert(el[0] < len(p))
                p.append(x[el[0]])
            else:
                p.append(el)
        return p
        

    def constrained_scipy_ln_likelihood(x):
        all_params = intercalate_constraints(x)
        return -ln_likelihood(data, all_params)
        
    if len(x0) > 0:
        if USE_FMIN:
            solution = optimize.fmin(constrained_scipy_ln_likelihood,
                                     x0,
                                     xtol=1e-8,
                                     disp=False)
            
        else:
            opt_blob = optimize.fmin_l_bfgs_b(constrained_scipy_ln_likelihood,
                                     x0,
                                     bounds=constr_bounds,
                                     approx_grad=True,
                                     epsilon=1e-8,
                                     disp=False)
            solution = opt_blob[0]
        all_params = intercalate_constraints(solution)
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
    num_args_expected = 2 + len(initial_parameter_guess)
    output_stream = sys.stdout
    output_stream.write('Expecting ' + str(num_args_expected) + ''' arguments.
    The last argument should be the number of parametric bootstrapping replicates,

    The first argument should be the path (filename) for the datafile (which
        should be a csv file with tab as the column separator.
    
    The intervening arguments should be values of the parameters in the null
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

    So if the data was in test.csv, and you want to perform 1000 simulations and you want to test the
        hypothesis that:
            ''' + c_name + ' = ' + c_val + '''
        then you would use the arguments:

test.csv ''' + ' '.join(parg_list) +  ''' 1000

''')        
        
    
if __name__ == '__main__':
    # user-interface and sanity checking...

    # we 
    import sys
    try:
        filepath = sys.argv[1] 
        arguments = sys.argv[2:]
        if len(arguments) < 1 + len(initial_parameter_guess):
            print len(arguments)
            print_help()
            sys.exit(1)
    except Exception, e:
        print 'Error:', e, '\n'
        print_help()
        sys.exit(1)

    real_data = read_data(filepath)

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
            try:
                if arg.upper() == 'NONE':
                    null_params.append(None)
                elif arg.upper().startswith("EQUALS"):
                    n = int(arg[len("EQUALS"):])
                    if n == len(null_params):
                        sys.exit("You can't constrain a parameter to be equal to itself")
                    if n > len(null_params):
                        sys.exit("You can't constrain a parameter to be equal to a \"later\" parameter (one with a higher index)")
                    if n < len(null_params) and isinstance(null_params[n], list):
                        sys.exit("You can't constrain a parameter to be equal a parameter that itself is to be equal to a third parameter")
                    null_params.append([n])
                else:
                    raise
            except:
                raise
                print_help()
                sys.exit("Expecting a parameter value to be a number, None, or Equals<some number> where <some number> is replaced with the number of the parameter that is constrained to have the same value (the first parameter should be numbered 0)")
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
        elif isinstance(null_params[n], list):
            constrained_index = null_params[n][0]
            print "Under the null, ", param_name, " is constrained to be equal to", parameter_names[constrained_index], "(", null_mle_list[constrained_index] ,")"
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

