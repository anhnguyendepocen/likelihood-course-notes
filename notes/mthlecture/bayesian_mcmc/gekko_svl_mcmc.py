#!/usr/bin/env python

from scipy import optimize
from math import pow, log, exp, sqrt
from random import random, normalvariate
from copy import copy
import numpy

verbose = False
  
_use_fmin = False




############################################################################
# Begin model-specific initialization code
############################################################################

############################################################################
# Here declare a "class" that allows us to bundle together different pieces
#   of information about our data in a convenient bundle. The read_data 
#   function and simulate data function will create "Skink" objects, and 
#   store them in a list.
#
# Specifically the data_list will have an element for each mating group.
#
# Each of these elements will consist of three lists - one for each diet
#   treatment.  Specifically, if group_data is the data for one mating group 
#   (foursome), then:
#       group_data[0] will be a list of individuals exposed to the "Small"
#               insect diet,
#       group_data[1] will be a list of individuals exposed to the "Medium"
#               insect diet, and 
#       group_data[2] will be a list of individuals exposed to the "Large"
#               insect diet.
# Each of this lists will be a list of "Skink objects"
#
# If s1 and s2 refer to two different Skink objects then you can see if they
#   share a mother using code like this:
#
#   if s1.mom == s2.mom:
#       here is a block of code to execute for shared mothers
#   else:
#       here is a block of code to execute for distinct mothers
# 
# Note that we use the "." operator to examine an attribute inside the "Skink"
#   objects.  The objects are just a convenient way to keep info on mom, dad,
#   treatment, and y-value (the dependent variable, growth rate) bundled
#   together.
############################################################################
class Skink(object):
    def __init__(self, mom, dad, treatment, y):
        self.mom = mom
        self.dad = dad
        self.treatment = treatment
        self.y = y
    def __repr__(self):
        return "Skink(mom=" + str(self.mom) + ", dad=" + str(self.dad) + ", treatment=" + str(self.treatment) + ", y=" + str(self.y) + ")"
    def __str__(self):
        return self.repr()


class ExponentialDistribution(object):
    def __init__(self, mean):
        self.mean = mean
    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''
            f(x) = (mean^-1) e^(-x/mean)
        So:
            ln[f(x)] = -x/mean - ln[mean]
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = (denominator - numerator)/mean
        '''
        return (denominator_variate - numerator_variate)/self.mean
        
# This is a list of values to try for the numerical optimizer.  The length of 
#   this list is also used by some functions to determine the dimensionality
#   of the model
initial_parameter_guess = [10.0, 10.0, 10.0, 1, 4 ]
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
parameter_names = ['mu_S', 'mu_M', 'mu_L', 'var_G', 'var_Error']


# This version of the program will us the L-BFGS constrained optimization
#   approach to maximize the likelihood.  
# Here we'll specify bounds for each of the parameters as pairs (min, max) which
#   specify the minimum and maximum value.  If a parameter is unbounded we can 
#   simply list None as the bound.
# Allowing var_Error to be 0 can result in numerical problems as the lnL goes to
#   very small numbers.  Putting a lower bound will avoid this.
min_var_error = 1e-6
parameter_bounds = [(float('-inf'), float('inf')), 
                    (float('-inf'), float('inf')),
                    (float('-inf'), float('inf')), 
                    (0.0, float('inf')),
                    (min_var_error, float('inf')), ]

prior_distribution = [
    ExponentialDistribution(10.0),
    ExponentialDistribution(10.0),
    ExponentialDistribution(10.0),
    ExponentialDistribution(5.0),
    ExponentialDistribution(1.0),
    ]
# we expect the number of parameters to be the same in both lists. This is a 
#   sanity check that helps us see if we made a mistake.
#
assert(len(initial_parameter_guess) == len(parameter_names)) 
############################################################################
# End model-specific initialization code
############################################################################




def calculate_incidence_row_for_indiv(skink):
    '''This function should return a python list. This list will be a single
    row in the incidence matrix.  
    It should contain a coefficient (e.g. 1 or 0) for each of the parameters 
    that appear in the equation of the linear predictor for this particular 
    skink.
    
    The incidence matrix will be multiplied by a parameter values. So the length
    of each row that you return here should be equal to the number of parameters
    
    '''
    
    treatment = skink.treatment
    mom = skink.mom
    dad = skink.dad
    
    # If you were to want a row that is [1, 2, 0]
    #   then you could substitute:
    #
    #   INCIDENCE_MATRIX_ROW = [1, 2, 0]
    #
    # or you could build the list one element at time:
    #
    # INCIDENCE_MATRIX_ROW = []
    # INCIDENCE_MATRIX_ROW.append(1)
    # INCIDENCE_MATRIX_ROW.append(2)
    # INCIDENCE_MATRIX_ROW.append(0)
    #    
    INCIDENCE_MATRIX_ROW = []
    if treatment == 0:
        INCIDENCE_MATRIX_ROW = [1, 0, 0]
    elif treatment == 1:
        INCIDENCE_MATRIX_ROW = [0, 1, 0]
    else:
        INCIDENCE_MATRIX_ROW = [0, 0, 1]
    
    return INCIDENCE_MATRIX_ROW
    


# you should not need to alter this next line.
the_incidence_matrix = None
    
    
def ln_likelihood(the_data, param_list):
    '''Calculates the log-likelihood of the parameter values in `param_list`
    based on `the_data`
    '''
    global the_incidence_matrix
    if verbose:
        sys.stderr.write('param = ' + str(param_list) + '\n')
    ############################################################################
    # Begin model-specific log-likelihood code
    ############################################################################
    first_p, second_p, third_p, fourth_p, fifth_p = param_list
    mu0 = first_p
    mu1 = second_p
    mu2 = third_p
    var_g = fourth_p
    var_error = fifth_p
    
    # The data is stored in multiple forms.  They will be explained below as needed.
    # You should not have to change the next 4 lines...
    blocked_by_foursome = the_data[0]
    observed_values = the_data[1]
    foursome_sizes = the_data[2]
    flattened_data = the_data[3]
    
    # do our "sanity-checking" to make sure that we are in the legal range of 
    #   the parameters.
    #
    for p in param_list:
        if p != p:
            return float('-inf')
    if var_g < 0.0:
        return float('-inf')
    if var_error < 0.0:
        return float('-inf')


    FIXED_EFFECT_LIST = [mu0, mu1, mu2]
    
    # this next line creates a column vector, to be multiplied by the incidence 
    #   matrix (you should not have to change this)
    fixed_effects = numpy.matrix(FIXED_EFFECT_LIST).transpose()
    
    # this calculates a vector of expected values from the fixed effects and 
    #   incidence matrix  (you should not have to change this)
    expected_values = the_incidence_matrix*fixed_effects
    
    
    # Because the values for the skinks from different foursomes will have a 
    #   covariance of 0, most of the covariance matrix will be 0's and we can
    #   create it in block-diagonal form that is easier to work with (and results
    #   in much faster calculations.
    # Here will create a list of matrices.  This will be the 
    #   non-zero blocks along the diagonal of the covariance matrix.
    # You won't have to change this next line.
    non_zero_covariance_blocks = list()

    var_individ = var_g + var_g + var_error
    covar_full_sibs = var_g + var_g
    covar_same_mom = var_g
    covar_same_dad = var_g
    covar_same_foursome = 0.0
    
    # Here we walk over the data set one foursome at a time.
    for foursome_index, foursome_data in enumerate(flattened_data):
        # num_in_foursome will hold the number of skinks from this foursome
        #   from all parental combinations and treatments.
        num_in_foursome = foursome_sizes[foursome_index]
        
        # make an empty covariance matrix for this foursome (no need to change this).
        empty_row = [None]*num_in_foursome
        # make an empty maxtrix by copying the empty row num_in_foursome times
        #   (you don't have to change this line).
        foursome_cov = [copy(empty_row) for i in xrange(num_in_foursome)]
        
        for i, skink_i in enumerate(foursome_data):
            for j in range(i, num_in_foursome):
                skink_j = foursome_data[j]

                ################################################################
                # here we need to fill in foursome_cov[i][j] with the
                # covariance for skink_i and skink_j
                #
                # We'll do this by examining the attributes (such as skink_i.mom 
                #   and skink_i.dad) in skink_i and skink_j to fill in cov_element
                #
                # and then storing cov_element
                ################################################################
                if i == j:
                    cov_element = var_individ
                else:
                    if skink_i.mom == skink_j.mom:
                        if skink_i.dad == skink_j.dad:
                            cov_element = covar_full_sibs
                        else:
                            cov_element = covar_same_mom
                    else:
                        if skink_i.dad == skink_j.dad:
                            cov_element = covar_same_dad
                        else:
                            cov_element = covar_same_foursome
                
                
                ################################################################
                # this is where we store the covariance element (the cov matrix
                # is symmetric, so we store it in (i, j) and (j, i)
                # If you calculated the covariance and stored it in the variable
                #   called cov_element, then you should not have to change this.
                ################################################################
                foursome_cov[i][j] = cov_element
                foursome_cov[j][i] = cov_element
        # append this non-zero part of the covariance matrix to a list
        #   that will be used to represent a block diagonal matrix for all 
        #   measurements.
        # first we'll convert it from a python list of lists to a numpy matrix
        # (you should not have to change this).
        numpy_fam_cov = numpy.matrix(foursome_cov)
        non_zero_covariance_blocks.append(numpy_fam_cov)

    # here we calculate the inverse (exploiting the block diagonal structure)
    inverse_var = invert_block_diagonal(non_zero_covariance_blocks)

    # here we calculate the log of the determinant (exploiting the block diagonal structure)
    ln_determinant = ln_determinant_block_diagonal(non_zero_covariance_blocks)

    # We can calculate residuals by substracting observed_values from
    #   expected_values.  When we convert the python lists of values 
    #   to numpy.matrix objects, we get a column matrix. 
    residuals_column = expected_values - observed_values

    # We can transpose the column vector of residuals to get a row...
    residuals_row = residuals_column.transpose()
    
    scaled_dev_sq = residuals_row*inverse_var*residuals_column
    ln_l = -0.5*(ln_determinant + float(scaled_dev_sq))

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
    mu0 = first_p
    mu1 = second_p
    mu2 = third_p
    var_g = fourth_p
    var_error = fifth_p
    

    # Here we'll grab the form of the data that is a list for each family
    #   there will be a list of all of the Skinks in that family.  We can
    #   use the skink.mom, skink.dad, skink.treatment attributes of each Skink
    #   to figure out mom, dad, and treatment
    # You should not have to change this line.
    by_foursome_list = template_data[3]

    sd_g = sqrt(var_g)
    sd_error = sqrt(var_error)
    treatment_effects = [mu0, mu1, mu2]

    ####################################################################
    # We'll walk through each foursome, simulating new values for the response "y"
    #
    # NOTE: We are going to modify the data "in-place" in this simulation (unlike
    #   previous simulations where we left the original data untouched).
    ####################################################################
    for foursome_data in by_foursome_list:
        # each foursome has 4 random variables, the parental effects...
        mom0_effect = normalvariate(0, sd_g)
        mom1_effect = normalvariate(0, sd_g)
        dad0_effect = normalvariate(0, sd_g)
        dad1_effect = normalvariate(0, sd_g)

        ####################################################################
        # We'll walk through each skink in the foursome and simulate a value...
        ####################################################################
        for skink in foursome_data:
            treatment = skink.treatment
            mom = skink.mom
            dad = skink.dad
            ########################################
            # At this point the variables "treatment", "mom" and "dad"
            # all refer to the current skink in the current foursome.
            # If you draw a random value for the dependent variable 
            # and store it in skink.y, then the simulation will be complete

            t_effect = treatment_effects[skink.treatment]
            if skink.mom == 0:
                m_effect = mom0_effect
            else:
                m_effect = mom1_effect
            if skink.mom == 0:
                d_effect = dad0_effect
            else:
                d_effect = dad1_effect
            
            full_effect = t_effect + m_effect + d_effect
            SIM_VALUE = normalvariate(full_effect, sd_error)
            
            ####################################################################
            # Here we can record SIM_VALUE as this skink's response. This
            #   this will complete our simulation of a dependent variable for
            #   this individual skink.
            ####################################################################
            skink.y = SIM_VALUE

    # You won't need to change the next line, but I'll explain it anyway...
    # This is a bit cryptic. Our simulation has been modifying the Skink objects
    #   in by_foursome_list. Because these *same* skink objects are also stored
    #   in the first element of template_data (in a form that more clearly 
    #   reveals the factorial nature of the study), we can call process_data
    #   on template_data[0] to transform the simulated data to make it easier 
    #   to process each simulation.
    # process_data is also called by read_data.  So we are really just using this
    #   function to put our simulations into the same structure as the original
    #   data. 
    return process_data(template_data[0])
    ############################################################################
    # End model-specific simulation code
    ############################################################################



################################################################################
#
# YOU SHOULD NOT HAVE TO MODIFY THE CODE BELOW THIS POINT !!!
################################################################################



def calculate_incidence_matrix(the_data):
    '''This function is called to fill in the incidence matrix for the model.
    
    You should not need to touch this code.  It calls calculate_incidence_row_for_indiv
    repeatedly to actually fill the matrix.
    '''
    global the_incidence_matrix
    the_incidence_matrix_row_list = []
    for foursome_index, foursome_data in enumerate(the_data):
        for treatment_index, indiv_data in enumerate(foursome_data):
            for indiv in indiv_data:
                the_incidence_matrix_row = calculate_incidence_row_for_indiv(indiv)
                the_incidence_matrix_row_list.append(the_incidence_matrix_row)
    the_incidence_matrix = numpy.matrix(the_incidence_matrix_row_list)





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

def ln_determinant_block_diagonal(block_list):
    det = 0.0
    for block in block_list:
        det = det + log(numpy.linalg.det(block))
    return det

def propose_parameter(old_p, bounds, window_width=1.0e-2):
    ''' Returns a new, proposed value that is drawn from
        U(old_p - window_width/2, old_p + window_width/2)
    
    '''
    # draw a U(-0.5, 0.5) variable
    u = random() - 0.5


    offset = u*window_width
    
    new_p = old_p + offset
    min_b, max_b = bounds
    while new_p < min_b or new_p > max_b:
        if new_p < min_b:
            diff = min_b - new_p
            new_p = diff + min_b
        if new_p > max_b:
            diff = new_p - max_b
            new_p = max_b - diff
    # this a symmetric proposal, so the ln_hastings_ratio is always 0.0
    return new_p, 0.0

def calc_ln_prior_ratio(old_p, new_p, prior_distribution):
    '''Calls the prior_distribution's calc_ln_prob_ratio method
    to determin the log of:
        the prior density of new_p / the prior density of old_p
    '''
    return prior_distribution.calc_ln_prob_ratio(new_p, old_p)
    
def do_mcmc(data, num_iterations, param_output_stream, initial_point):
    curr_params = []
    proposed_params = []
    for p in initial_point:
        curr_params.append(p)
        proposed_params.append(p)

        
    curr_lnL = ln_likelihood(data, curr_params)
    num_params = len(curr_params)
    num_accepted = 0
    for iteration in xrange(num_iterations):
        for m in xrange(num_params):
            curr_p = curr_params[m]
            bounds = parameter_bounds[m]
            
            new_p, ln_hastings_ratio = propose_parameter(curr_p, bounds)
            proposed_params[m] = new_p
            
            proposed_lnL = ln_likelihood(data, proposed_params)

            prior_dist = prior_distribution[m]
            ln_prior_ratio = calc_ln_prior_ratio(curr_p, new_p, prior_dist)
            
            ln_like_ratio = proposed_lnL - curr_lnL
            ln_acceptance_prob = ln_like_ratio + ln_prior_ratio + ln_hastings_ratio
            
            if ln_acceptance_prob > 0.0 or log(random()) < ln_acceptance_prob:
                num_accepted = num_accepted + 1
                curr_lnL = proposed_lnL
                curr_params[m] = new_p
            else:
                # reject the move, set the appropriate element of the proposed_params
                #   list back to its previous value
                proposed_params[m] = curr_p
        params_as_str_list = [str(i) for i in curr_params]
        params_tab_separated = '\t'.join(params_as_str_list)
        param_output_stream.write(str(iteration) + '\t' + str(curr_lnL) + '\t' + params_tab_separated + '\n')
    return num_accepted
    




def process_data(data_set):
    '''To make it easier to deal with the data, we'll calculate a few summaries
    of the data.  Three items will be returned:
        the data_set,
        the observations,
        the number of individuals in each mating group.
    '''
    block_sizes = []
    observed_values = []
    flattened_data = []
    for group_data in data_set:
        sz = 0
        flat = []
        for group_treatment_data in group_data:
            sz = sz + len(group_treatment_data)
            flat.extend(group_treatment_data)
            for skink in group_treatment_data:
                observed_values.append(skink.y)
        flattened_data.append(flat)
        block_sizes.append(sz)
    observed_values = numpy.matrix(observed_values).transpose()
    return data_set, observed_values, block_sizes, flattened_data

def read_data(filepath):
    '''Reads filepath as a tab-separated csv file and returns a 3dimensional data matrix.'''
    import os
    import csv
    import itertools
    if not os.path.exists(filepath):
        raise ValueError('The file "' + filepath + '" does not exist')
    
    TREATMENT_CODES = 'SML'    
    MAX_TREATMENT_VALUE = len(TREATMENT_CODES) - 1
    
    # Here we create a csv reader and tell it that a tab (\t) is the column delimiter
    entries = csv.reader(open(filepath, 'rbU'), delimiter=',')

    # Here we check that file has the headers that we exect
    first_row = entries.next()
    expected_headers = ['mating group (foursome)', 'mom', 'dad', 'treatment', 'y (growth rate)']
    for got, expected in itertools.izip(first_row, expected_headers):
        if got.lower().strip() != expected:
            raise ValueError('Error reading "' + filepath + '": expecting a column labelled "' + expected + '", but found "' + got + '" instead.')

    # It is not too hard to have this loop put the data in the right spot in the
    #   matrix, so that the data file can be somwhat flexible.

    by_family = {}
    for n, row in enumerate(entries):
        fam_id, mom_id, dad_id, treatment_code, value = row
        try:
            fam_id = int(fam_id)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for family ID, but got ' + str(fam_id))
        try:
            mom_id = int(mom_id)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for mom ID, but got ' + str(mom_id))
        try:
            dad_id = int(dad_id)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for mom ID, but got ' + str(dad_id))
        try:
            treatment_id = TREATMENT_CODES.index(treatment_code.upper())
            assert(treatment_id >= 0 and treatment_id <= MAX_TREATMENT_VALUE)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting a single letter code (one of "' + TREATMENT_CODES + '") for the treatment, but got ' + str(treatment_code))
        try:
            value = float(value)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an number for the trait value, but got ' + str(value))
        # a new family corresponds to an empty dictionary of individuals
        #   for the 0, and 1 treatment. So we can create a list with two empty
        #   dictionaries as the "blank" entry.
        empty_family_entry = [[] for i in range(MAX_TREATMENT_VALUE + 1)]
        fam_array = by_family.setdefault(fam_id, empty_family_entry)
        # now we grab the appropriate one for this treatment
        list_for_fam_treatment = fam_array[treatment_id]
        list_for_fam_treatment.append(Skink(mom=mom_id, 
                                            dad=dad_id,
                                            treatment=treatment_id,
                                            y=value))

    fam_keys = by_family.keys()
    fam_keys.sort()

    # Now we'll sort the data matrix and print some status information
    status_stream = sys.stdout
    status_stream.write("Data read for " + str(len(fam_keys)) + " families...\n")
    full_data = []
    for i, fam_id in enumerate(fam_keys):
        treatments_list = by_family[fam_id]
        status_stream.write("  Mating group index=" + str(i) + " (id in datafile = " + str(fam_id) + "):\n")
        full_data.append(treatments_list)
        for treat_ind, indiv_list in enumerate(treatments_list):
            treatment_code = TREATMENT_CODES[treat_ind]
            status_stream.write('    ' + str(len(indiv_list)) + ' individuals in with treatment code "' + treatment_code + '" (numerical code ' + str(treat_ind) + ')\n')
    return process_data(full_data)
    

def print_help():
    num_args_expected = 2 + len(initial_parameter_guess)
    output_stream = sys.stdout
    output_stream.write('Expecting ' + str(num_args_expected) + ''' arguments.
    The last argument should be the number MCMC iterations,

    The first argument should be the path (filename) for the datafile (which
        should be a csv file with tab as the column separator.
    
    The intervening arguments should be starting values for each of the 
        parameters
    
    The order of the parameters is in this constraint statement is:
        ''')
    for p in parameter_names:
        output_stream.write(p + ' ')
    assert len(initial_parameter_guess) > 0
    output_stream.write('\n\n')        


    
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
    calculate_incidence_matrix(real_data[0])

    # The number of simulations is the last parameter...
    #
    num_iterations = int(arguments[-1])
    
    for n, v in enumerate(arguments[:-1]):
        try:
            initial_parameter_guess[n] = float(v)
        except:
            sys.exit('Expecting an initial value for the parameter "' + parameter_names[n] + '" but got: "' + v + '"')
        min_bound, max_bound = parameter_bounds[n]
        if min_bound is not None and initial_parameter_guess[n] < min_bound:
            sys.exit('Expecting an initial value for the parameter "' + parameter_names[n] + '" to be >=' + min_bound + ' but got "' + v + '"')
        if max_bound is not None and initial_parameter_guess[n] > max_bound:
            sys.exit('Expecting an initial value for the parameter "' + parameter_names[n] + '" to be <=' + max_bound + ' but got "' + v + '"')
    
    param_output_stream = sys.stderr
    param_output_stream.write("Iteration\tlnL")
    for p in parameter_names:
        param_output_stream.write('\t' + p)
    param_output_stream.write('\n')
    
    num_accepted = do_mcmc(real_data,
                           num_iterations, 
                           param_output_stream, 
                           initial_parameter_guess)
    
    print "Accepted " + str(num_accepted) + " updates\n"
    print "Ran " + str(num_iterations) + " iterations over all " + str(len(initial_parameter_guess)) + " parameters.\n"



