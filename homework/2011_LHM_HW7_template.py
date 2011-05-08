#!/usr/bin/env python
'''
I opted for a slower, but (hopefully) easier system that the data structure
    used in the latent_gekko example.  Either style works (and all of the 
    theory is the same in both cases).  I just thought this template might be easier to
    work from.
    
I've tried to use ALL_CAPS to flag things that you will definitely have to change.

Unlike previous assignments, you'll need to look a bit at the data processing (see the
    process_data function).
'''
from scipy import optimize
from math import pow, log, exp, sqrt
from random import random, normalvariate
from copy import copy

verbose = False





class Coconut(object):
    '''This object models an individual plant.'''
    def __init__(self, field, year, plus_n, plus_p, mass):
        self.field = field
        self.year = year
        self.plus_n = plus_n
        self.plus_p = plus_p
        self.mass = mass

    def __repr__(self):
        attributes_str = ", ".join([i + "=" + str(getattr(self, i)) for i in ["field", "year", "plus_n", "plus_p", "mass"]])
        return "Coconut(" + attributes_str + ")"

    def __str__(self):
        return self.repr()

class Dataset(object):
    pass

class SameFieldTreatmentYearGroup(list):
    '''This is basically a list Coconut objects designed to hold 
    individuals from the same field, treatment, and year.
    
    The length of this list will be determined by number of individuals in that
    treatment x field x year combination.
    
    By putting individuals in to "homogenous" groups like this we can quickly
        sum effects by multiplication (num*effect) rather than walking over every
        individual in the group
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0
        self.treatment_variable = []
        self.field_variable = None
        self.year_variable = None
    def calc_sum_treatment(self):
        '''Returns the sum of the fertilization treatments effects across the group'''
        if not self.treatment_variable:
            return 0.0
        return self.num * sum([i.value for i in self.treatment_variable])
    def calc_sum_field(self):
        '''Returns the sum of the field effects across the group'''
        return self.num * self.field_variable.value
    def calc_sum_year(self):
        '''Returns the sum of the year effects across the group'''
        return self.num * self.year_variable.value
    def calc_sum_field_treatment(self):
        '''Returns the sum of the field and treatment effects across the group'''
        return self.calc_sum_treatment() + self.calc_sum_field()
    def calc_sum_year_treatment(self):
        '''Returns the sum of the year and treatment effects across the group'''
        return self.calc_sum_treatment() + self.calc_sum_year()
    def calc_sum_field_year(self):
        '''Returns the sum of the field and treatment effects across the group'''
        return self.calc_sum_field() + self.calc_sum_year()
    def calc_sum_effects(self):
        '''Returns the sum of the field and treatment effects across the group'''
        return self.calc_sum_treatment() + self.calc_sum_field() + self.calc_sum_year()
        
class SharedEffectGroupList(list):
    '''
    This list will hold SameFieldTreatmentYearGroup objects.  The intent is that
    you can put SameFieldTreatmentYearGroup objects that share an effect
    into this list.  
    
    Then when you need to update that effect it will be easy to do calculations
        only over the groups of individuals that "use" that effect.  This save
        wasted calculations.
    
    Note that the num attribute and the sum_mass attribute depend on the data
        and not the parameters, so they only have to updated when the data is 
        read.
    
    It would be possible to make a more efficient implementation by "caching"
        previous sums of effects and assuring that we only call recalculating
        those parts of the sum that have changed rather than calling
        calculuate_sum_over_effects(), but calculuate_sum_over_effects()
        will work fine for this exercise
        
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0

    def recalculate_data_sums(self):
        n = 0
        mass = 0.0
        for element in self:
            n = n + element.num
            mass = mass + element.sum_mass
        self.num = n
        self.sum_mass = mass

    def recalculate_all_sums(self):
        self.recalculate_data_sums()
        return self.calculuate_sum_over_effects()


    def calculuate_sum_over_effects(self):
        ft = 0.0
        for group in self:
            ft = ft + group.calc_sum_effects()
        return ft


################################################################################
# The following four classes encapsulate probability distrbutions that can
#
class Parameter(object):
    '''This class will keep information about each parameter bundled together
    for easy reference
    '''
    def __init__(self, name='<unknown parameter>', value=0.0, prior=None, proposal_window=None, min_bound=None, max_bound=None):
        self.name = name
        self.value = value
        self.prior = prior
        self.proposal_window = proposal_window
        self.min_bound = min_bound if min_bound is not None else float('-inf')
        self.max_bound = max_bound if max_bound is not None else float('inf')

class LatentVariable(Parameter):
    '''For the purpose of MCMC a latent variable acts just like a parameter.
    So here we create a class for LatentVariable objects, but the word "pass"
    tells python that we have nothing to add.
    
    The "LatentVariable(Parameter)" syntax means that a LatentVariable object
        can be treated like a Parameter object.  In computing jargon we would
        say that LatentVariable is derived from Parameter
    '''
    pass


class ParamIndex:
    '''This "class" is just a mechanism for associating a parameter name
    with an index.  It is useful, because during the MCMC command, you'll need
    to update each parameter (in other programming languages this is done with
    a facility called an "enum").
    
    The parameters can be stored in a global list of parameter (called
    global_parameter_list in this code), but it can be hard to remember what
    index into that list will give you the variance_e parameter, for instance.
    
    By writing 

        var_e_param = global_parameter_list[ParameIndex.VAR_E]

    you can use ParameIndex.VAR_E to represent the number.
    
    You have to keep the argument to "range" equal to the number of parameters
        names that you want to register.
    And you have to keep the order of the parameter names identical to the 
        order of the parameter objects in the global_parameter_list.
    
    You should not use this class to register names for the latent variables
        (the number of those depends on the dataset size, so you can't write the
        code in a generic way).
    '''
    
    SOME_PARAM_NAME, ANOTHER_PARAM_NAME, VAR_E = range(3)


################################################################################
# The following four classes encapsulate probability distrbutions that can
#   be used as priors in MCMC calculations.
# To create an object that corresponds to a specific instance from one of these
#   classes of distributions use syntax like this:
#
#   x = NormalDistribution(mean=1.0, variance=1.0)
#   y = ExponentialDistribution(mean=1.0)
#   z = GammaDistribution(mean=2.2, variance=4.0)
#
# Note that the mean and variance arguments can be Parameter objects or 
#   LatentVariable objects, if these properties of the distribution can 
#   change. So
#    
#   m = Parameter(value=5)
#   y = NormalDistribution(mean=m, variance=2)
#
#   # the mean of y is now 5
#
#   m.value = 4
#
#   # the mean of y is now 4
#
################################################################################
class ContinuousDistribution(object):
    '''The continous distribution class just has some helpers that make it 
    easier to write code when the key qualities of a distribution (mean and 
    variance) could either be specified by fixed numbers or Parameter objects
    
    These methods are used in the classes derived from ContinuousDistribution
    '''
    def get_mean(self):
        try:
            return self.mean.value
        except:
            return self.mean        

    def get_variance(self):
        try:
            return self.var.value
        except:
            if self.var is None:
                try:
                    sd = self.sd.value
                except:
                    sd = self.sd
                return sd*sd
            else:
                return self.var

class GammaDistribution(ContinuousDistribution):
    def __init__(self, mean, variance):
        ''' mean = shape*scale
            var = shape*scale*scale
        '''
        self.var = variance
        self.mean = mean

    def get_scale_shape(self):
        '''Returns the scale and shape parameters of the Gamma Distribution 
        (calculated from the mean and variance).
        '''
        mean = self.get_mean()
        variance = self.get_variance()
        scale = variance/mean
        shape = mean/scale
        return scale, shape

    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''Returns the log of the ratio of the probability density evaluated
        at `numerator_variate` over the density evaluated at the `denominator_variate`.
        
         f(x) = x^{shape-1}e^{-x/scale}/Gamma function
        So:
            ln[f(x)] = (shape-1)ln(x) -x/scale - ln[Some constant based on the Gamma function]
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = (denominator - numerator)/mean
        '''
        scale, shape = self.get_scale_shape()
        ln_numerator =  (shape-1)*log(numerator_variate) - numerator_variate/scale
        ln_denominator = (shape-1)*log(denominator_variate) - denominator_variate/scale
        return ln_numerator - ln_denominator

class ExponentialDistribution(ContinuousDistribution):
    def __init__(self, mean):
        assert(mean >= 0.0)
        self.mean = mean
    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''Returns the log of the ratio of the probability density evaluated
        at `numerator_variate` over the density evaluated at the `denominator_variate`.
        
            f(x) = (mean^-1) e^(-x/mean)
        So:
            ln[f(x)] = -x/mean - ln[mean]
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = (denominator - numerator)/mean
        '''
        return (denominator_variate - numerator_variate)/self.get_mean()

class NormalDistribution(ContinuousDistribution):
    def __init__(self, mean, sd=None, variance=None):
        '''Takes the mean and standard deviation.'''
        self.mean = mean
        self.sd = sd
        self.var = variance

    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''Returns the log of the ratio of the probability density evaluated
        at `numerator_variate` over the density evaluated at the `denominator_variate`.
         f(x) = (2 pi sd^2) ^ -0.5  e^(-(x-mean)^2/(2sd^2))
        So:
            ln[f(x)] = -0.5 ln[2 pi sd^2] - (x-mean)^2/(2sd^2)
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = ((denominator - mean)^2 - (numerator-mean)^2)/(2sd^2)
                                                  = (denominator^2 - numerator^2 + 2*(numerator-denominator)*mean)/(2sd^2)
        '''
        mean = self.get_mean()
        variance = self.get_variance()

        d_sq = denominator_variate*denominator_variate
        n_sq = numerator_variate*numerator_variate
        third_term = 2*mean*(numerator_variate - denominator_variate)
        return (d_sq - n_sq + third_term)/(2*variance)

################################################################################
# Here we create the parameter objects.  Our parameters are associated
#   with a prior and a proposal, in addition to having a value, a name, and 
#   possibly minumum or maximum bounds.  Thus, it it easier to bundle
#   all of that info in one class (See Parmeter above
#
################################################################################
global_parameter_list = [
    YOU_WILL_NEED_TO_CREATE_YOUR_PARAMETERS_BUT_NOT_LATENT_VARIABLES_HERE
    THE SYNTAX IS LIKE THIS
    Parameter(name='some_descriptive_name_here',
              value=10.0, # or whatever value
              prior=ExponentialDistribution(mean=-5), # USE A REAL PRIOR HERE, -5 is not a valid mean
              proposal_window=1.0, # this controls how drastic the proposals are.
              min_bound=None, # set to 0.0 if the parameter cannot be negative
              max_bound=None), # put a number as the value if the parameter has a maximum
    Parameter(name='next_parameter_',
             ...),
    # REMEMBER TO KEEP THE ORDER OF THE PARAMETERS IDENTICAL TO THE ORDER OF NAMES
    #    IN ParamIndex, IF YOU ARE GOING TO USE ParamIndex IN YOUR CODE
    ]



def propose_parameter(old_p, param_obj):
    '''Sliding window move which uses the param_obj.proposal_window to 
    determine the window around old_p to use. The move also keeps the value
    within the bounds on param_obj.
    
    The proposed value that is drawn from:
        U(old_p - window_width/2, old_p + window_width/2)
    Returns the proposed value and the log of the Hastings Ratio
    '''
    # draw a U(-0.5, 0.5) variable
    u = random() - 0.5

    window_width = param_obj.proposal_window
    offset = u*window_width
    
    new_p = old_p + offset
    
    min_b = param_obj.min_bound
    max_b = param_obj.max_bound
    
    while new_p < min_b or new_p > max_b:
        if new_p < min_b:
            diff = min_b - new_p
            new_p = diff + min_b
        if new_p > max_b:
            diff = new_p - max_b
            new_p = max_b - diff
    # this a symmetric proposal, so the ln_hastings_ratio is always 0.0
    return new_p, 0.0


def metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
    '''Uses the Metropolis-Hastings algorithm to decide whether the move should
    be accepted or not. Returns True to indicate acceptance (False otherwise).
    '''
    ln_acceptance = ln_like_ratio +  ln_prior_ratio + ln_hastings_ratio
    if ln_acceptance > 0.0:
        return True
    if log(random()) < ln_acceptance:
        return True
    return False

def update_param_var_e(relevant_slice_of_data, curr_param_list):
    '''This is just an example (which is not valid) to demonstrate syntax.
    You'll need some function like this for every parameter.
    
    do_mcmc is written to assume that the function will return the number 
        of updates that are accepted
    '''
    param_obj = curr_param_list[ParamIndex.VAR_E]

    current_value = param_obj.value
    proposed_value, ln_hastings_ratio = propose_parameter(current_value, param_obj)
    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(proposed_value, current_value)

    ln_like_ratio = YOU_WILL_HAVE_TO_USE_THE_DATA_AND_PARAMETER_VALUES_TO_CALCULATE_THE_LN_OF_THE_LIKELIHOOD_RATIO
    
    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        param_obj.value = proposed_value
        return 1
    return 0




def update_latent_effects_for_nitrogen(relevant_groups_of_data, curr_param_list, latent_effects):
    '''
    It is easiest to walk through the list of each type of latent effect, so this 
    function receives the list of latent_effects (a list with LatentVariable objects)
    and should be sent a corresponding list of the relevant data slices.
    
    The ideas is that for every latent variable, you can get a SharedEffectGroupList
        that will contain all of the SameFieldTreatmentYearGroup items that
        "use" this latent effect.  So if you base the likelihood calculations
        on just that SharedEffectGroupList, then you will not be examining 
        data that is irrelevant (you want be calculating terms that will cross 
        out in the likelihood ratio calculation).
        
    '''
    # the following assert assures that you have the right number of SharedEffectGroupList objects (one for each latent_effect).
    assert(len(relevant_groups_of_data) == len(latent_effects))
    var_error = curr_param_list[ParamIndex.VAR_E]

    num_accepted = 0
    for effect_index, data_for_effect in enumerate(relevant_groups_of_data):

        latent_variable = latent_effects[effect_index]
        current_value = latent_variable.value
        # We need to make sure that the sums over the other parameter/latent variables are up to date
        #   before we use them...
        # There are more efficient ways to do this (namely by only updating the parts of the
        #   sum that change), but it won't make a huge difference for this application.
        # Here we'll use a trick.  We'll set the effect to 0, sum the effects
        #   for each group of plants, and then restore the effect to its
        #   previous value.  This lets us get the sum of all of the other effects
        #   (and they will be included the appropriate number of times).
        latent_variable.value = 0.0
        sum_other_effects = data_for_effect.calculuate_sum_over_effects()
        latent_variable.value = current_value
        
        # At this point sum_other_effects will hold the summation of all of the
        #   effects other than `latent_variable` over all of the individuals
        #   that latent_variable applies to.  If you did your calculation
        #   of the ln_likelihood_ratio in step 2.5 of the homework correctly 
        #   (hint, hint), then at least sum updates can be simplified by summing
        #   over effects. This allows us to avoid iterating over every individual.
        
        proposed_value, ln_hastings_ratio = propose_parameter(current_value, latent_variable)
    
        # num will be the number of relevant individuals...
        num = data_for_effect.num
        # sum_mass will be the sum of the mass for all relevant individuals...
        sum_mass = data_for_effect.sum_mass

        ln_like_ratio = SOMETHING_GOES_HERE

        ln_prior_ratio = SOMETHING_ELSE_GOES_HERE
    
        if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
            num_accepted += 1
            latent_variable.value = proposed_value

    return num_accepted        


def write_sampled_parameters(param_output_stream, iteration, all_variables_list):    
    # We convert every parameter or latent variable value to a string of characters to be written
    params_as_str_list = [str(i.value) for i in all_variables_list]
    # We introduce tabs between each element
    params_tab_separated = '\t'.join(params_as_str_list)
    # We write a line to the parameter output stream.
    param_output_stream.write(str(iteration) + '\t' + params_tab_separated + '\n')

def do_mcmc(data_and_latent_variables, num_iterations, param_output_stream, sample_freq):
    data = data_and_latent_variables[0]

    # These are some useful "slices" of the data (see process_data)
    by_homogenous_group = data.by_same_year_field_treatment
    all_individuals = data.individuals
    by_year = data.by_year
    by_field = data.by_field
    by_field_nitro = data.by_field_nitro
    by_field_phosph = data.by_field_phosph
    by_field_both = data.by_field_both
    
    # Each of the following "lv_list" holds LatentVariable objects that will
    #   need to be updated during MCMC (just like the parameters need to be updated)
    year_lv_list = data_and_latent_variables[1]
    field_lv_list = data_and_latent_variables[2]
    field_nitro_lv_list = data_and_latent_variables[3]
    field_phospho_lv_list = data_and_latent_variables[4]
    field_both_lv_list = data_and_latent_variables[5]
    
    num_accepted = 0
    
    # make a huge list of all of the parameters and latent variables (this will make it 
    #   easier to print out the state at each iteration of the MCMC)
    all_variables_list = global_parameter_list + year_lv_list + field_lv_list + field_nitro_lv_list + field_phospho_lv_list + field_both_lv_list
        
    
    for iteration in xrange(num_iterations):

        accepted = update_param_var_e(SLICE_OF_DATA_OR_OTHER_INFO_YOU_NEED_TO_UPDATE_THIS_PARAMETER, global_parameter_list)
        num_accepted += accepted

        accepted = update_param_SOME_OTHER_PARAMETER(DATA_SLICE_YOU_NEED_TO_UPDATE_THIS_PARAMETER, global_parameter_list)
        num_accepted += accepted

        accepted = update_latent_effects_for_nitrogen(RELEVANT_SLICE_OF_DATA, global_parameter_list, field_nitro_lv_list)
        num_accepted += accepted

        accepted = update_SOME_OTHER_latent_effects(RELEVANT_SLICE_OF_DATA, global_parameter_list, THE_LIST_OF_LATENT_VARIABLEL)
        num_accepted += accepted

        # the funky % syntax means that we will only call write_sampled_parameters
        #   if iteration is a multiple of sample_freq
        if iteration % sample_freq == 0:
            print("iter=" + str(iteration))
            write_sampled_parameters(param_output_stream, iteration, all_variables_list)
    return num_accepted

def process_data(no_fert_by_yf, with_nitrogen_by_yf, with_phosphorus_by_yf, with_both_by_yf, all_years, all_fields):
    '''This function will create the LatentVariable instances, and it will also
        create groupings of the Coconut objects that will make it easy to traverse
        the entire data set in a variety of ways. You can think of these groupings
        as different slices of the full matrix of data.
    
    The function returns:
        0. the data set;
        1. a list of "year effect" latent variables;
        2. a list of "field effect" latent variables.
        3. a list of "field x nitrogen effect" latent variables.
        4. a list of "field x phosphorus effect" latent variables.
        5. a list of "field x both fertilizers effect" latent variables.
    
    The data_set object has the following attributes
        by_same_year_field_treatment a list containing all the data grouped
            into SameFieldTreatmentYearGroup objects (each individual in the 
            a SameFieldTreatmentYearGroup shares the same year, field, and 
            treatment.
        individuals (a list of all Coconut objects)
        by_year (a list of SharedEffectGroupList objects)
        by_field (a list of SharedEffectGroupList objects)
        by_field_nitro (a list of SharedEffectGroupList objects)
        by_field_phosph (a list of SharedEffectGroupList objects)
        by_field_both (a list of SharedEffectGroupList objects)
    
    
    The arguments are dictionaries that contain all of the individuals. There
        are four dictionaries (one for each treatment) and all so a set of all years,
        and a set field numbers. Python dictionaries store associations between
        arbitary keys and objects.  They are convenient when you don't know how
        many items you'll be dealing with.  You won't need to understand 
        how dictionaries work to complete the homework.  You can just take 
        my word that loops over things like sorted_years and sorted_fields
        are indeed letting us iterate over the data for those field and years.
    '''
    sorted_years = list(all_years)
    sorted_years.sort()
    sorted_fields = list(all_fields)
    sorted_fields.sort()

    # create latent variables for each year
    year_lv_list = []
    for year in sorted_years:
        # here, we create a latent variable for each year...
        # We create a new LatentVariable in the same way that we create
        #   a Parameter object, so refer to the notes above (near global_parameter_list creation)
        #   for further info.
        year_lv = LatentVariable(name="year_" + str(year),
                                 value=SOMETHING,
                                 prior=SOMETHING
                                 proposal_window=SOMETHING)
        year_lv_list.append(year_lv)

    field_lv_list = []
    field_nitro_lv_list = []
    field_phospho_lv_list = []
    field_both_lv_list = []
    for field in sorted_fields:
        # here we create latent variables for each field and treatment combination
        #   and store them in the proper list..
    
        field_lv = LatentVariable(name="field_" + str(field),
                                  SOME_STUFF_HERE)
        field_lv_list.append(field_lv)
        
        nitro_lv = LatentVariable(name="field_" + str(field) + "_nitro",
                                  SOME_STUFF_HERE)
        field_nitro_lv_list.append(nitro_lv)
        
        phospho_lv = LatentVariable(name="field_" + str(field) + "_phospho",
                                  SOME_STUFF_HERE)
        field_phospho_lv_list.append(phospho_lv)
        
        both_lv = LatentVariable(name="field_" + str(field) + "_both",
                                 SOME_STUFF_HERE)
        field_both_lv_list.append(both_lv)
    
    # it is possible that you won't need to modify the rest of this function.
    #   but you may.  It depends on the parameterization that you chose.
    
    # Check out the comments below.
    
    
    
    data = Dataset()
    data.by_same_year_field_treatment = []
    data.individuals = []
    data.by_year = []
    data.by_field = []
    data.by_field_nitro = []
    data.by_field_phosph = []
    data.by_field_both = []
    
    for year_index, year in enumerate(sorted_years):
        # same_year_group will be a "slice" of the data that was collected 
        #   for this year only.  This will be useful when we want to
        #   update the latent variable associated with this year (because
        #   the data contained in same_year_group will be the only bits of 
        #   data that are relevant to that latent variable).
        same_year_group = SharedEffectGroupList()
        
        # year_lv is the LatentVariable for this year.
        year_lv = year_lv_list[year_index]
        
        for field_index, field in enumerate(sorted_fields):
            # field_lv is the LatentVariable for this field.
            field_lv = field_lv_list[field_index]

            key = (year, field)
            
            # The next few lines get a SameFieldTreatmentYearGroup for each of the
            #   four treatments for this year x field combination, and perform
            #   some basic initialization of those objects (such as counting
            #   the number of individuals and summing the masses within
            #   a SameFieldTreatmentYearGroup object.            
            no_fertilizer = no_fert_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            no_fertilizer.num = len(no_fertilizer)
            no_fertilizer.sum_mass = sum([el.mass for el in no_fertilizer])
            
            nitro = with_nitrogen_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            nitro.num = len(nitro)
            nitro.sum_mass = sum([el.mass for el in nitro])

            phospho = with_phosphorus_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            phospho.num = len(phospho)
            phospho.sum_mass = sum([el.mass for el in phospho])

            both = with_both_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            both.num = len(both)
            both.sum_mass = sum([el.mass for el in both])


            # This is where we associate the groups of objects that are exposed
            #   to the same conditions (the SameFieldTreatmentYearGroup lists of
            #   Coconut objects), with the latent variables that are thought to
            #   influence them.  The SameFieldTreatmentYearGroup class
            #   uses:
            #       year_variable the latent variable for this year
            #       field_variable the latent variable for this field
            #       treatment_variable list of latent variables pertaining to fertilization
            #   to allow it to sum over the effects for the group.
            #  So we have to "tell" each object what its latent variables are.
            #
            # Depending on what parameterization you use for latent variables
            #   you may need to modify the following lines.
            # First we get the treatment variables for this field.
            nitro_treament = field_nitro_lv_list[field_index]
            phospo_treatment = field_phospho_lv_list[field_index]
            interaction_effect = field_both_lv_list[field_index]
            
            # Now for each SameFieldTreatmentYearGroup we set the appropriate
            #   attributes so that it "knows" which latent variables apply to it.
            no_fertilizer.treatment_variable = []
            no_fertilizer.field_variable = field_lv
            no_fertilizer.year_variable = year_lv
            
            nitro.treatment_variable = [nitro_treament]
            nitro.field_variable = field_lv
            nitro.year_variable = year_lv

            phospho.treatment_variable = [phospo_treatment]
            phospho.field_variable = field_lv
            phospho.year_variable = year_lv

            both.treatment_variable = [nitro_treament, phospo_treatment, interaction_effect]
            both.field_variable = field_lv
            both.year_variable = year_lv


            # Finally we store the individuals and the SameFieldTreatmentYearGroup
            #   in the appropriate slots in our `data` object.
            # Every individual should go in data.individuals, so we extend that list...
            data.individuals.extend(no_fertilizer)
            data.individuals.extend(nitro)
            data.individuals.extend(phospho)
            data.individuals.extend(both)
            
            # data.by_same_year_field_treatment is a list, but it holds 
            #   SameFieldTreatmentYearGroup objects rather than the individual
            #   Coconut objects.  So we just add each of the SameFieldTreatmentYearGroup
            #   objects to the end of the by_same_year_field_treatment slice.
            data.by_same_year_field_treatment.append(no_fertilizer)
            data.by_same_year_field_treatment.append(nitro)
            data.by_same_year_field_treatment.append(phospho)
            data.by_same_year_field_treatment.append(both)

            # same_year_group is also a list of SameFieldTreatmentYearGroup objects
            #    but it only pertains to this year.
            same_year_group.append(no_fertilizer)
            same_year_group.append(nitro)
            same_year_group.append(phospho)
            same_year_group.append(both)
            
        # After we have considered every field in this year, we add all of the
        #   accumulated data to the by_year group.  recalculate_all_sums()
        #   calculates the sum of the mass and other quantities for
        #   the same_year_group.
        same_year_group.recalculate_all_sums()
        data.by_year.append(same_year_group)


    # This loop creates the slices that correspond to different treatments
    #   See notes below.
    for field_index, field in enumerate(sorted_fields):
        same_field_group = SharedEffectGroupList()
        same_field_nitro_group = SharedEffectGroupList()
        same_field_phospho_group = SharedEffectGroupList()
        same_field_both_group = SharedEffectGroupList()

        for year_index, year in enumerate(sorted_years):
            key = (year, field)
            
            # the following four lines get the SameFieldTreatmentYearGroup
            #   objects for each of the four treatments for this field x year combination.
            no_fertilizer = no_fert_by_yf[key]
            nitro = with_nitrogen_by_yf[key]
            phospho = with_phosphorus_by_yf[key]
            both = with_both_by_yf[key]

            # In one possible parameterization you have:
            #   - a field effect that applies to all four treatmens
            #   - a nitrogen effect that applies to two treatmens
            #   - a phosphorous effect that applies to two treatmens
            #   - an interaction effect that applies to only one
            # The following lines add the groups of data to the relevant
            #   lists assuming this parameterization.  If you are using an 
            #   alternative, then you'll have to change which
            #   lists a particular SameFieldTreatmentYearGroup 
            #   gets added to:
            same_field_group.append(no_fertilizer)
            same_field_group.append(nitro)
            same_field_group.append(phospho)
            same_field_group.append(both)

            same_field_nitro_group.append(nitro)
            same_field_nitro_group.append(both)

            same_field_phospho_group.append(phospho)
            same_field_phospho_group.append(both)

            same_field_both_group.append(both)
        
        # Now that we have data in groups we pre-calculate things, and add
        #    the groups to the Dataset object.
        same_field_group.recalculate_all_sums()
        data.by_field.append(same_field_group)   

        same_field_nitro_group.recalculate_all_sums()
        data.by_field_nitro.append(same_field_nitro_group)

        same_field_phospho_group.recalculate_all_sums()
        data.by_field_phosph.append(same_field_phospho_group)

        same_field_both_group.recalculate_all_sums()
        data.by_field_both.append(same_field_both_group)
    
    # now we can do some "sanity checks" to make sure that we didn't miss anything
    #   we will "assert" some statements that will be true (if we don't have any
    #   bugs).
    data.num = len(data.individuals)
    data.sum_mass = sum([i.mass for i in data.individuals])
    assert(data.num == sum([i.num for i in data.by_year]))
    assert(data.num == sum([i.num for i in data.by_field]))
    assert(data.sum_mass -  sum([i.sum_mass for i in data.by_year]) < 1e-5)
    assert(data.sum_mass - sum([i.sum_mass for i in data.by_field]) < 1e-5)
    
    return data, year_lv_list, field_lv_list, field_nitro_lv_list, field_phospho_lv_list, field_both_lv_list

def read_data(filepath):
    '''Reads filepath as a tab-separated csv file.
    Returns a Dataset object and five lists of latent variables.

    The returned values come from process_data, so see that function for details.
    
    You probably won't need to dig into this function to complete the homework.
    '''
    import os
    import csv
    import itertools
    if not os.path.exists(filepath):
        raise ValueError('The file "' + filepath + '" does not exist')
    
    # Here we create a csv reader and tell it that a tab (\t) is the column delimiter
    entries = csv.reader(open(filepath, 'rbU'), delimiter=',')

    # Here we check that file has the headers that we exect
    first_row = entries.next()
    expected_headers = ['year', 'field', 'nitrogen', 'phosphorus', 'individual', 'mass']
    for got, expected in itertools.izip(first_row, expected_headers):
        if got.lower().strip() != expected:
            raise ValueError('Error reading "' + filepath + '": expecting a column labelled "' + expected + '", but found "' + got + '" instead.')


    
    with_nitrogen_by_yf = {}
    with_phosphorus_by_yf = {}
    with_both_by_yf = {}
    no_fert_by_yf = {}
    all_years = set()
    all_fields = set()
    for n, row in enumerate(entries):
        year, field, nitrogen, phosphorus, individual, mass = row
        try:
            year = int(year)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for year, but got ' + str(year))
        try:
            field = int(field)
            assert(field >= 0)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an integer for field, but got ' + str(field))
        try:
            nitrogen = int(nitrogen)
            assert(nitrogen == 0 or nitrogen == 1)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an 0 or 1 for nitrogen, but got ' + str(nitrogen))
        try:
            phosphorus = int(phosphorus)
            assert(phosphorus == 0 or phosphorus == 1)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting an 0 or 1 for phosphorus, but got ' + str(phosphorus))
        try:
            mass = float(mass)
            assert(mass >= 0.0)
        except:
            raise ValueError("Error reading data row " + str(1 + n) + ' of "' + filepath + '": expecting a positive number for mass, but got ' + str(mass))

        all_years.add(year)
        all_fields.add(field)

        coconut = Coconut(field=field,
                          year=year,
                          plus_n=nitrogen,
                          plus_p=phosphorus,
                          mass=mass)

        # Get the group with the same year, field, and treatment
        if nitrogen == 1:
            if phosphorus == 1:
                coconut.same_yft_group = with_both_by_yf.setdefault((year, field), SameFieldTreatmentYearGroup())
            else:
                coconut.same_yft_group = with_nitrogen_by_yf.setdefault((year, field), SameFieldTreatmentYearGroup())
        else:
            if phosphorus == 1:
                coconut.same_yft_group = with_phosphorus_by_yf.setdefault((year, field), SameFieldTreatmentYearGroup())
            else:
                coconut.same_yft_group = no_fert_by_yf.setdefault((year, field), SameFieldTreatmentYearGroup())

        coconut.same_yft_group.append(coconut)
    
    return process_data(no_fert_by_yf, 
                        with_nitrogen_by_yf,
                        with_phosphorus_by_yf,
                        with_both_by_yf, 
                        all_years,
                        all_fields)
    

def print_help():
    num_args_expected = 3 + len(global_parameter_list)
    output_stream = sys.stdout
    output_stream.write('Expecting ' + str(num_args_expected) + ''' arguments.
    The second-to-last argument should be the number MCMC iterations,
    The last argument should be the number MCMC iterations between sampling the
        chain (the periodicity for thinning)

    The first argument should be the path (filename) for the datafile (which
        should be a csv file with tab as the column separator.
    
    The intervening arguments should be starting values for each of the 
        parameters
    
    The order of the parameters is in this constraint statement is:
        ''')
    for p in global_parameter_list:
        output_stream.write(p.name + ' ')
    output_stream.write('\n\n')        


    
if __name__ == '__main__':
    # user-interface and sanity checking...

    # we 
    import sys
    try:
        filepath = sys.argv[1] 
        arguments = sys.argv[2:]
        if len(arguments) < 2 + len(global_parameter_list):
            print len(arguments)
            print_help()
            sys.exit(1)
    except Exception, e:
        print 'Error:', e, '\n'
        print_help()
        sys.exit(1)

    # The number of simulations is the last parameter...
    #
    num_iterations = int(arguments[-2])
    sample_freq = int(arguments[-1])
    for n, v in enumerate(arguments[:-2]):
        param = global_parameter_list[n]
        try:
            param.value = float(v)
        except:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" but got: "' + v + '"')
        if param.min_bound is not None and param.value < param.min_bound:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" to be >=' + param.min_bound + ' but got "' + v + '"')
        if param.max_bound is not None and param.value > param.max_bound:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" to be <=' + param.max_bound + ' but got "' + v + '"')
        print "param #", n, param.name, "=", param.value

    data_and_latent_variables = read_data(filepath)

    # the observed Dataset object is the first thing returned...    
    obs_data = data_and_latent_variables[0]
    # next come the lists of latent variables.
    year_lv_list = data_and_latent_variables[1]
    field_lv_list = data_and_latent_variables[2]
    field_nitro_lv_list = data_and_latent_variables[3]
    field_phospho_lv_list = data_and_latent_variables[4]
    field_both_lv_list = data_and_latent_variables[5]
    
    param_output_stream = sys.stderr
    param_output_stream.write("Iteration")
    for p in global_parameter_list:
        param_output_stream.write('\t' + p.name)
    for p in year_lv_list:
        param_output_stream.write('\t' + p.name)
    for p in field_lv_list:
        param_output_stream.write('\t' + p.name)
    for p in field_nitro_lv_list:
        param_output_stream.write('\t' + p.name)
    for p in field_phospho_lv_list:
        param_output_stream.write('\t' + p.name)
    for p in field_both_lv_list:
        param_output_stream.write('\t' + p.name)
    param_output_stream.write('\n')
    
    num_accepted = do_mcmc(data_and_latent_variables,
                           num_iterations, 
                           param_output_stream,
                           sample_freq)
    
    print "Accepted " + str(num_accepted) + " updates\n"
    print "Ran " + str(num_iterations) + " iterations over all " + str(len(global_parameter_list)) + " parameters.\n"



