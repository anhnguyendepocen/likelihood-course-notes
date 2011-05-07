#!/usr/bin/env python

from scipy import optimize
from math import pow, log, exp, sqrt
from random import random, normalvariate
from copy import copy

verbose = False

class Coconut(object):
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
    '''Holds Coconut objects for a particular field, treatment, and year.
    
    The length of this list will be determined by number of individuals in that
    treatment x field x year combination.
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0
        self.treatment_variable = None
        self.field_variable = None
        self.year_variable = None
    def calc_sum_treatment(self):
        '''Returns the sum of the fertilization treatments effects across the group'''
        if self.treatment_variable is None:
            return 0
        return self.num * self.treatment_variable.value
    def calc_sum_field(self):
        '''Returns the sum of the field effects across the group'''
        return self.num * self.field_variable.value
    def calc_sum_year(self):
        '''Returns the sum of the field effects across the group'''
        return self.num * self.year_variable.value
    def calc_sum_field_treatment(self):
        return self.calc_sum_treatment() + self.calc_sum_field()
    def calc_sum_year_treatment(self):
        return self.calc_sum_treatment() + self.calc_sum_year()
    def calc_sum_field_year(self):
        return self.calc_sum_field() + self.calc_sum_year()
        
class GroupOfSameFieldTreatmentYearGroup(list):
    def calc_sum_field_treatment(self):
        ft = 0.0
        for group in self:
            ft = ft + group.calc_sum_field_treatment()
        return ft
    def calc_sum_year_treatment(self):
        ft = 0.0
        for group in self:
            ft = ft + group.calc_sum_year_treatment()
        return ft
    def calc_sum_field_year(self):
        ft = 0.0
        for group in self:
            ft = ft + group.calc_sum_field_year()
        return ft

class SameYearGroup(list):
    '''Holds all of the SameFieldTreatmentYearGroup objects that share
    a particular year
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0
        self.sum_field_treatment_effects = 0.0

class SameFieldGroup(list):
    '''Holds all of the SameFieldTreatmentYearGroup objects that share
    a particular field
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0
        self.sum_year_treatment_effects = 0.0

class SameFieldTreatmentGroup(list):
    '''Holds all of the SameFieldTreatmentYearGroup objects that share
    a particular field and treatment
    '''
    def __init__(self):
        self.num = 0
        self.sum_mass = 0.0
        self.sum_field_year_effects = 0.0

class GammaDistribution(object):
    def __init__(self, mean, variance):
        ''' mean = shape*scale
            var = shape*scale*scale
        '''
        self.scale = variance / mean
        self.shape = mean / self.scale
        

    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''
            f(x) = x^{shape-1}e^{-x/scale}/Gamma function
        So:
            ln[f(x)] = (shape-1)ln(x) -x/scale - ln[Some constant based on the Gamma function]
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = (denominator - numerator)/mean
        '''
        ln_numerator =  (self.shape-1)*log(numerator_variate) - numerator_variate/self.scale
        ln_denominator = (self.shape-1)*log(denominator_variate) - denominator_variate/self.scale
        return ln_numerator - ln_denominator

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

class NormalDistribution(object):
    def __init__(self, mean, sd=None, variance=None):
        '''Takes the mean and standard deviation.'''
        self.mean = mean
        if variance is None:
            self.var = sd*sd
        else:
            self.var = variance
    def calc_ln_prob_ratio(self, numerator_variate, denominator_variate):
        '''
            f(x) = (2 pi sd^2) ^ -0.5  e^(-(x-mean)^2/(2sd^2))
        So:
            ln[f(x)] = -0.5 ln[2 pi sd^2] - (x-mean)^2/(2sd^2)
        
        So: 
            ln[f(numerator)] - ln[f(denominator)] = ((denominator-mean)^2 - (numerator-mean)^2)/(2sd^2)
                                                  = (denominator^2 - numerator^2-mean)^2 - (numerator-mean)^2)/(2sd^2)
        '''
        d_sq = denominator_variate*denominator_variate
        n_sq = numerator_variate*numerator_variate
        third_term = 2*self.mean*(numerator_variate - denominator_variate)
        return (d_sq - n_sq + third_term)/(2*self.var)

    
class Parameter(object):
    '''This class will keep information about each parameter bundled together
    for easy reference
    '''
    def __init__(self, name, initial_value, prior=None, proposal_window=None, min_bound=None, max_bound=None):
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value
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
    VAR_A, MU_B, VAR_B, MU_G, VAR_G, MU_D, VAR_D, MU_R, VAR_R, VAR_E = range(10)

global_parameter_list = [
    Parameter(name='var_a',
              initial_value=10.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=None,
              max_bound=None),
    Parameter(name='mu_b',
              initial_value=250,
              prior=GammaDistribution(mean=250.0, variance=50.0),
              proposal_window=5.0,
              min_bound=None,
              max_bound=None),
    Parameter(name='var_b',
              initial_value=1.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=0.0,
              max_bound=None),
    Parameter(name='mu_g',
              initial_value=250,
              prior=NormalDistribution(mean=5.0, variance=10.0),
              proposal_window=5.0,
              min_bound=None,
              max_bound=None),
    Parameter(name='var_g',
              initial_value=1.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=0.0,
              max_bound=None),
    Parameter(name='mu_d',
              initial_value=250,
              prior=NormalDistribution(mean=5.0, variance=10.0),
              proposal_window=5.0,
              min_bound=None,
              max_bound=None),
    Parameter(name='var_d',
              initial_value=1.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=0.0,
              max_bound=None),
    Parameter(name='mu_r',
              initial_value=250,
              prior=NormalDistribution(mean=5.0, variance=10.0),
              proposal_window=5.0,
              min_bound=None,
              max_bound=None),
    Parameter(name='var_r',
              initial_value=1.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=0.0,
              max_bound=None),
    Parameter(name='var_e',
              initial_value=1.0,
              prior=ExponentialDistribution(mean=10.0),
              proposal_window=1.0,
              min_bound=0.0,
              max_bound=None),
    ]



def propose_parameter(old_p, param_obj):
    ''' Returns a new, proposed value that is drawn from
        U(old_p - window_width/2, old_p + window_width/2)
    
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

def calc_ln_prior_ratio(old_p, new_p, full_param_value_list, param_obj):
    '''Calls the prior_distribution's calc_ln_prob_ratio method
    to determin the log of:
        the prior density of new_p / the prior density of old_p
    '''
    return param_obj.prior.calc_ln_prob_ratio(new_p, old_p)

def metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
    ln_acceptance = ln_like_ratio +  ln_prior_ratio + ln_hastings_ratio
    return bool(ln_acceptance > 0.0 or log(random()) < ln_acceptance)
        
def update_alpha_0(curr_params, data):
    var_error = curr_params[ParamIndex.VAR_ERROR]
    param_val = curr_params[ParamIndex.ALPHA_0]
    param_obj = global_parameter_list[ParamIndex.ALPHA_0]
    family_list_for_treatment = data.treatment_list[0]
    
    num_s = family_list_for_treatment.num
    sum_y = family_list_for_treatment.sum_y
    sum_b = family_list_for_treatment.sum_b
    
    param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)

    numerator = 2*(param_val_star - param_val)*(sum_y - sum_b) - num_s*(param_val_star**2 - param_val**2)
    denominator = 2*var_error
    ln_like_ratio = numerator/denominator

    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)

    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        return param_val_star, True
    return param_val, False


def update_alpha_1(curr_params, data):
    var_error = curr_params[ParamIndex.VAR_ERROR]
    param_val = curr_params[ParamIndex.ALPHA_1]
    param_obj = global_parameter_list[ParamIndex.ALPHA_1]
    family_list_for_treatment = data.treatment_list[1]
    
    num_s = family_list_for_treatment.num
    sum_y = family_list_for_treatment.sum_y
    sum_b = family_list_for_treatment.sum_b
    sum_c = family_list_for_treatment.sum_c
    
    param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)

    numerator = 2*(param_val_star - param_val)*(sum_y - sum_b - sum_c) - num_s*(param_val_star**2 - param_val**2)
    denominator = 2*var_error
    ln_like_ratio = numerator/denominator

    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)

    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        return param_val_star, True
    return param_val, False
    
    
def update_var_g(curr_params, data):
    param_val = curr_params[ParamIndex.VAR_G]
    param_obj = global_parameter_list[ParamIndex.VAR_G]

    b_sq = data.sum_b_sq
    num_fam = data.num_fam
    
    param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)

    ln_like_ratio = (b_sq)/(2*param_val) - (b_sq)/(2*param_val_star) - num_fam*log(param_val_star/param_val)/2.0

    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)

    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        return param_val_star, True
    return param_val, False

def update_var_l_interaction(curr_params, data):
    param_val = curr_params[ParamIndex.VAR_L_INTERACTION]
    param_obj = global_parameter_list[ParamIndex.VAR_L_INTERACTION]

    c_sq = data.sum_c_sq
    num_fam = data.num_fam
    
    param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)

    ln_like_ratio = (c_sq)/(2*param_val) - (c_sq)/(2*param_val_star) - num_fam*log(param_val_star/param_val)/2.0

    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)

    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        return param_val_star, True
    return param_val, False

def update_var_error(curr_params, data, fam_effect_list, fam_l_iteraction_list):
    alpha_0 = curr_params[ParamIndex.ALPHA_0]
    alpha_1 = curr_params[ParamIndex.ALPHA_1]
    param_val = curr_params[ParamIndex.VAR_ERROR]
    param_obj = global_parameter_list[ParamIndex.VAR_ERROR]

    sum_sq_resid = 0.0
    for fam_ind, family in enumerate(data):
        fam_effect = fam_effect_list[fam_ind]
        fam_l_interaction = fam_l_iteraction_list[fam_ind]
        for indiv in family[0]:
            resid = indiv.y - alpha_0 - fam_effect
            sum_sq_resid += resid*resid
        for indiv in family[1]:
            resid = indiv.y - alpha_1 - fam_effect - fam_l_interaction
            sum_sq_resid += resid*resid
    
    param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)

    ln_like_ratio = sum_sq_resid*(1/(2*param_val) - 1/(2*param_val_star)) - data.num*log(param_val_star/param_val)/2.0

    ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)

    if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
        return param_val_star, True
    return param_val, False
        


def update_fam_effects(curr_params, data, fam_effect_list, fam_l_iteraction_list):
    alpha_0 = curr_params[ParamIndex.ALPHA_0]
    alpha_1 = curr_params[ParamIndex.ALPHA_1]
    var_g = curr_params[ParamIndex.VAR_G]
    var_error = curr_params[ParamIndex.VAR_ERROR]

    all_fam_treatment0 = data.treatment_list[0]
    all_fam_treatment1 = data.treatment_list[1]
    num_accepted = 0
    denominator = 2*var_error

    param_obj = Parameter(name='fam_effect',
              initial_value=0.0,
              prior=NormalDistribution(mean=0.0, sd=sqrt(var_g)),
              proposal_window=1.0,
              min_bound=None,
              max_bound=None)
              
    for fam_ind, family in enumerate(data):
        param_val = fam_effect_list[fam_ind]
        fam_l_interaction = fam_l_iteraction_list[fam_ind]
        treat0 = family[0]
        treat1 = family[1]
    
        param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)
    
        sq_diff = param_val**2 - param_val_star**2
        diff = param_val_star - param_val
        numerator = family.num*sq_diff
        numerator += 2*(family.sum_y - treat0.num*alpha_0 - treat1.num*(alpha_1 + fam_l_interaction))*diff
        ln_like_ratio = numerator/denominator
        ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)
    
        if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
            num_accepted += 1
            fam_effect_list[fam_ind] = param_val_star
            all_fam_treatment0.sum_b += diff
            all_fam_treatment1.sum_b += diff
            data.sum_b_sq -= sq_diff

    return num_accepted        


def update_fam_l_iteractions(curr_params, data, fam_effect_list, fam_l_iteraction_list):
    alpha_0 = curr_params[ParamIndex.alpha_0]
    alpha_1 = curr_params[ParamIndex.ALPHA_1]
    var_interaction = curr_params[ParamIndex.VAR_L_INTERACTION]
    var_error = curr_params[ParamIndex.VAR_ERROR]

    all_fam_treatment1 = data.treatment_list[1]
    
    param_obj = Parameter(name='fam_l_intearction',
          initial_value=0.0,
          prior=NormalDistribution(mean=0.0, sd=sqrt(var_interaction)),
          proposal_window=1.0,
          min_bound=None,
          max_bound=None)

    num_accepted = 0
    denominator = 2*var_error
    for fam_ind, family in enumerate(data):
        fam_effect = fam_effect_list[fam_ind]
        param_val = fam_l_iteraction_list[fam_ind]
        treat1 = family[1]
    
        param_val_star, ln_hastings_ratio = propose_parameter(param_val, param_obj)
    
        sq_diff = param_val**2 - param_val_star**2
        diff = param_val_star - param_val
        numerator = treat1.num*sq_diff
        numerator += 2*(treat1.sum_y - treat1.num*(alpha_1 + fam_effect))*diff
        ln_like_ratio = numerator/denominator
        ln_prior_ratio = param_obj.prior.calc_ln_prob_ratio(param_val_star, param_val)
    
        if metrop_hastings(ln_like_ratio, ln_prior_ratio, ln_hastings_ratio):
            num_accepted += 1
            fam_l_iteraction_list[fam_ind] = param_val_star
            all_fam_treatment1.sum_c += diff
            data.sum_c_sq -= sq_diff

    return num_accepted        
    
    
def do_mcmc(data_collections, num_iterations, param_output_stream, sample_freq):
    data = data_collections[0]
    curr_params = []
    proposed_params = []
    for p in global_parameter_list:
        curr_params.append(p.initial_value)
        proposed_params.append(p.initial_value)

    curr_lnL = 0.0  # Arbitrary, but does not matter, as we will do everything
                    #   with ln L ratios ...
    num_params = len(curr_params)
    num_accepted = 0
    
    fam_effects = []
    fam_l_iteractions = []
    blocked_by_family = data[0]
    for fam in xrange(data.num_fam):
        fam_effects.append(0.0)
        fam_l_iteractions.append(0.0)
    
    for iteration in xrange(num_iterations):

        p, accepted = update_alpha_0(curr_params, data)
        curr_params[ParamIndex.ALPHA_0] = p
        num_accepted += accepted

        p, accepted = update_alpha_1(curr_params, data)
        curr_params[ParamIndex.ALPHA_1] = p
        num_accepted += accepted
        
        p, accepted = update_var_g(curr_params, data)
        curr_params[ParamIndex.VAR_G] = p
        num_accepted += accepted
        
        p, accepted = update_var_l_interaction(curr_params, data)
        curr_params[ParamIndex.VAR_L_INTERACTION] = p
        num_accepted += accepted

        p, accepted = update_var_error(curr_params, data, fam_effects, fam_l_iteractions)
        curr_params[ParamIndex.VAR_ERROR] = p
        num_accepted += accepted
    
        accepted = update_fam_effects(curr_params, data, fam_effects, fam_l_iteractions)
        num_accepted += accepted

        accepted = update_fam_l_iteractions(curr_params, data, fam_effects, fam_l_iteractions)
        num_accepted += accepted

        if iteration % sample_freq == 0:
            params_as_str_list = [str(i) for i in curr_params + fam_effects + fam_l_iteractions]
            params_tab_separated = '\t'.join(params_as_str_list)
            param_output_stream.write(str(iteration) + '\t' + str(curr_lnL) + '\t' + params_tab_separated + '\n')
    return num_accepted

def process_data(no_fert_by_yf, with_nitrogen_by_yf, with_phosphorus_by_yf, with_both_by_yf, all_years, all_fields):
    '''Here we get dictionaries that contain all of the individuals. There
    are four dictionaries (one for each treatment) and all so a set of all years,
    and a set field numbers.
    
    This function will create the LatentVariable instances, and it will also
    create groupings of the Coconut objects that will make it easy to traverse
    the entire data set in a variety of ways.
    
    The function returns:
        0. the data set;
        1. a list of "year effect" latent variables;
        2. a list of "field effect" latent variables.
        3. a list of "field x nitrogen effect" latent variables.
        4. a list of "field x phosphorus effect" latent variables.
        5. a list of "field x both fertilizers effect" latent variables.
    
    The data_set object has the following attributes
        individuals (a list of all Coconut objects)
        by_year (a list of SameYearGroup objects)
        by_field (a list of SameFieldGroup objects)
        by_field_nitro_only (a list of SameFieldTreatmentGroup objects)
        by_field_phosph_only (a list of SameFieldTreatmentGroup objects)
        by_field_both (a list of SameFieldTreatmentGroup objects)
    '''
    sorted_years = list(all_years)
    sorted_years.sort()
    sorted_fields = list(all_fields)
    sorted_fields.sort()

    # create latent variables for each year
    year_lv_list = []
    for year in sorted_years:
        year_lv = LatentVariable(name="year_" + str(year),
                                 initial_value=0.0,
                                 prior=NormalDistribution(0, variance=global_parameter_list[ParamIndex.VAR_A]),
                                 proposal_window=1.0)
        year_lv_list.append(year_lv)

    # create latent variables for each field x treatment
    field_lv_list = []
    field_nitro_lv_list = []
    field_phospho_lv_list = []
    field_both_lv_list = []
    for field in sorted_fields:
        field_lv = LatentVariable(name="field_" + str(field),
                                  initial_value=0.0,
                                  prior=NormalDistribution(mean=global_parameter_list[ParamIndex.MU_B],
                                                           variance=global_parameter_list[ParamIndex.VAR_B]),
                                  proposal_window=1.0)
        field_lv_list.append(field_lv)
        
        nitro_lv = LatentVariable(name="field_" + str(field) + "_nitro",
                                  initial_value=0.0,
                                  prior=NormalDistribution(mean=global_parameter_list[ParamIndex.MU_G],
                                                           variance=global_parameter_list[ParamIndex.VAR_G]),
                                  proposal_window=1.0)
        field_nitro_lv_list.append(nitro_lv)
        
        phospho_lv = LatentVariable(name="field_" + str(field) + "_phospho",
                                  initial_value=0.0,
                                  prior=NormalDistribution(mean=global_parameter_list[ParamIndex.MU_D],
                                                           variance=global_parameter_list[ParamIndex.VAR_D]),
                                  proposal_window=1.0)
        field_phospho_lv_list.append(phospho_lv)
        
        both_lv = LatentVariable(name="field_" + str(field) + "_both",
                                  initial_value=0.0,
                                  prior=NormalDistribution(mean=global_parameter_list[ParamIndex.MU_R],
                                                           variance=global_parameter_list[ParamIndex.VAR_R]),
                                  proposal_window=1.0)
        field_both_lv_list.append(both_lv)
    
    
    data = Dataset()
    data.individuals = []
    data.by_year = []
    data.by_field = []
    data.by_field_nitro_only = []
    data.by_field_phosph_only = []
    data.by_field_both = []
    
    for year_index, year in enumerate(sorted_years):
        same_year_group = SameYearGroup()
        year_lv = year_lv_list[year_index]
        for field_index, field in enumerate(sorted_fields):
            field_lv = field_lv_list[field_index]

            key = (year, field)
            neither = no_fert_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            neither.num = len(neither)
            neither.sum_mass = sum([el.mass for el in neither])
            data.individuals.extend(neither)
            same_year_group.append(neither)
            neither.treatment_variable = None
            neither.field_variable = field_lv
            neither.year_variable = year_lv
            
            nitro = with_nitrogen_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            nitro.num = len(nitro)
            nitro.sum_mass = sum([el.mass for el in nitro])
            data.individuals.extend(nitro)
            same_year_group.append(nitro)
            nitro.treatment_variable = field_nitro_lv_list[field_index]
            nitro.field_variable = field_lv
            nitro.year_variable = year_lv

            phospho = with_phosphorus_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            phospo.num = len(phospho)
            phosho.sum_mass = sum([el.mass for el in phospho])
            data.individuals.extend(phosph)
            same_year_group.append(phosph)
            phospho.treatment_variable = field_phospho_lv_list[field_index]
            phospho.field_variable = field_lv
            phospho.year_variable = year_lv

            both = with_both_by_yf.setdefault(key, SameFieldTreatmentYearGroup())
            both.num = len(both)
            both.sum_mass = sum([el.mass for el in both])
            data.individuals.extend(both)
            same_year_group.append(both)
            both.treatment_variable = field_both_lv_list[field_index]
            both.field_variable = field_lv
            both.year_variable = year_lv

        same_year_group.num = sum([i.num for i in same_year_group])
        same_year_group.sum_mass = sum([i.sum_mass for i in same_year_group])
        data.by_year.append(same_year_group)

    for field_index, field in enumerate(sorted_fields):
        same_field_group = SameFieldGroup()
        same_field_nitro_group = SameFieldTreatmentGroup()
        same_field_phospho_group = SameFieldTreatmentGroup()
        same_field_both_group = SameFieldTreatmentGroup()

        for year_index, year in enumerate(sorted_years):
            key = (year, field)
            neither = no_fert_by_yf[key]
            same_field_group.append(neither)
            
            nitro = with_nitrogen_by_yf[key]
            same_field_group.append(nitro)
            same_field_nitro_group.append(nitro)

            phospho = with_phosphorus_by_yf[key]
            same_field_group.append(phosph)
            same_field_phospho_group.append(phospho)

            both = with_both_by_yf[key]
            same_field_group.append(both)
            same_field_both_group.append(both)
            
        same_field_group.num = sum([i.num for i in same_field_group])
        same_field_group.sum_mass = sum([i.sum_mass for i in same_field_group])
        same_field_group.sum_year_treatment_effects = self.calc_sum_year_treatment()
        data.by_field.append(same_field_group)
        
        same_field_nitro_group.num = sum([i.num for i in same_field_nitro_group])
        same_field_nitro_group.sum_mass = sum([i.sum_mass for i in same_field_nitro_group])
        same_field_group.sum_field_year_effects = self.calc_sum_field_year()
        data.by_field_nitro_only.append(same_field_nitro_group)

        same_field_phospho_group.num = sum([i.num for i in same_field_phospho_group])
        same_field_phospho_group.sum_mass = sum([i.sum_mass for i in same_field_phospho_group])
        same_field_group.sum_field_year_effects = self.calc_sum_field_year()
        data.by_field_phosph_only.append(same_field_phospho_group)

        same_field_both_group.num = sum([i.num for i in same_field_both_group])
        same_field_both_group.sum_mass = sum([i.sum_mass for i in same_field_both_group])
        same_field_group.sum_field_year_effects = self.calc_sum_field_year()
        data.by_field_both.append(same_field_both_group)
          
    return data, year_lv_list, field_lv_list, field_nitro_lv_list, field_phospho_lv_list, field_both_lv_list

def read_data(filepath):
    '''Reads filepath as a tab-separated csv file and returns a 3dimensional data matrix.'''
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
        all_fields.add(year)

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
    num_args_expected = 2 + len(global_parameter_list)
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
        if len(arguments) < 1 + len(global_parameter_list):
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
            param.initial_value = float(v)
        except:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" but got: "' + v + '"')
        if param.min_bound is not None and param.initial_value < param.min_bound:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" to be >=' + param.min_bound + ' but got "' + v + '"')
        if param.max_bound is not None and param.initial_value > param.max_bound:
            sys.exit('Expecting an initial value for the parameter "' + param.name + '" to be <=' + param.max_bound + ' but got "' + v + '"')

    real_data = read_data(filepath)
    
    param_output_stream = sys.stderr
    param_output_stream.write("Iteration\tlnL")
    for p in global_parameter_list:
        param_output_stream.write('\t' + p.name)
    for n, f in enumerate(real_data[0]):
        param_output_stream.write('\tfam' + str(n))
    for n, f in enumerate(real_data[0]):
        param_output_stream.write('\tfam_l_' + str(n))
    param_output_stream.write('\n')
    
    num_accepted = do_mcmc(real_data,
                           num_iterations, 
                           param_output_stream,
                           sample_freq)
    
    print "Accepted " + str(num_accepted) + " updates\n"
    print "Ran " + str(num_iterations) + " iterations over all " + str(len(global_parameter_list)) + " parameters.\n"



