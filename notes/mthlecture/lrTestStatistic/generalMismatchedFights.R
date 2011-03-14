#!/usr/bin/env R
data = read.table("simpleData.txt", header=TRUE);

############################################################################
# Begin model-specific initialization code
############################################################################
# This is a list of values to try for the numerical optimizer.  The length of 
#   this list is also used by some functions to determine the dimensionality
#   of the model

INITIAL.PARAMETER.GUESS = c(0.75, 0.75)

############################################################################
# End model-specific initialization code
############################################################################

calc.lrt.statistic = function(data, null.params) {
    # `data` should be a list with `num0wins` and `num1wins` elements
    # `null.params` should be a vector of parameter values

    calc.ln.likelihood = function(theta) {
        ############################################################################
        # Begin model-specific likelihood calculation code
        ############################################################################
    
        p = theta[1];
        w = theta[2];
    
        ###########################################################################
        # Return -inf if the parameters are out of range.  This will keep the optimizer
        #  from returning "illegal" values of the parameters.
        #
        if (p < 0 || p > 1.0 || w < 0.5 || w > 1.0) {
            return(-Inf);
        }
    
        p.SS = p * p;
        p.SW = p * (1 - p);
        p.WS = p.SW;
        p.WW = (1 - p) * (1 - p);
        
        # Calculate the probability that the males are evenly matched
        #
        p.Even = p.WW + p.SS;
    
        # Calculate the probability of our data if the match is Even
        #
        prob0.If.Even = 0.5 ;
        prob1.If.Even = 1 - prob0.If.Even ;
        p.Data.If.Even0 = prob0.If.Even ^ data$num0wins ;
        p.Data.If.Even1 = prob1.If.Even ^ data$num1wins ;
        p.data.And.Even = p.Data.If.Even0 * p.Data.If.Even1 * p.Even ;
        
        # Calculate the probability of our data if the match is Strong vs Weak
        #
        prob0.If.SW = w ;
        prob1.If.SW = 1 - w ;
        p.Data.If.SW0 = prob0.If.SW ^ data$num0wins ;
        p.Data.If.SW1 = prob1.If.SW ^ data$num1wins ;
        p.data.And.SW = p.Data.If.SW0 * p.Data.If.SW1 * p.SW ;
    
        # Calculate the probability of our data if the match is Strong vs Weak
        #
        prob0.If.WS = 1 - w ;
        prob1.If.WS = w ;
        p.Data.If.WS0 = prob0.If.WS ^ data$num0wins ;
        p.Data.If.WS1 = prob1.If.WS ^ data$num1wins ;
        p.data.And.WS = p.Data.If.WS0 * p.Data.If.WS1 * p.WS ;
        
        # the probability of our data is the sum of the joint probability of the 
        #   data and each of the three match types (Even, SW, and WS):
        #
        p.data = p.data.And.Even + p.data.And.SW + p.data.And.WS;
    
        log.p.data = log(p.data);
        return(sum(log.p.data));
        ############################################################################
        # End model-specific likelihood calculation code
        ############################################################################
    
    }
    
    optimizer.ln.like = function(theta) {
        return(-calc.ln.likelihood(theta));
    }

    ############################################################################
    # First we calculate the ML score when there are no constraints (the
    #   "global" ML solution).
    ############################################################################
    global.sln = optim(par=INITIAL.PARAMETER.GUESS,
                       fn=optimizer.ln.like,
                       method="Nelder-Mead");

    ############################################################################
    # Now we'll need to optimize the lnL subject to the constraints of the 
    #   null hypothesis.
    ############################################################################
    # Here we note which parameters are free in the null
    #
    is.free.param = is.na(null.params);
    
    intercalate.and.score = function (theta) {
        full.params = null.params;
        full.params[is.free.param] = theta;
        return(-calc.ln.likelihood(full.params));
    }
    
    x0 = INITIAL.PARAMETER.GUESS[is.free.param]
    if (sum(is.free.param) > 1) {
        null.sln = optim(par=x0,
                     fn=intercalate.and.score,
                     method="Nelder-Mead");
    }
    else {
        # we are optimizing probabilities that have to be >= 0.5
        opt.out = optimize(intercalate.and.score, c(0.5,1));
        fp = null.params;
        fp[is.na(fp)] = opt.out$minimum;
        null.sln = list(par=fp, 
                        value=opt.out$objective,
                        counts=NA,
                        convergence=NA,
                        message=NULL);
    }
    # our minimizer is returning the -log-likelihood, so we reverse the LRT here.
    #
    lrt = 2 * (global.sln$value - null.sln$value);
    return(list(lrt=lrt, global=global.sln, null=null.sln));
}

simulate.data = function(template.data, params) {
    ############################################################################
    # Begin model-specific simulation code
    ############################################################################
    s = params[1];
    w = params[2];
    num0wins.vec = c();
    p.SS = s*s;
    p.SW = s*(1 - s);
    p.WS = (1 - s)*s;
    p.WW = (1 - s)*(1 - s);
    p.Even = p.SS + p.WW;
    
    # We'll simulate the number of times that male 0 wins a bout, and obtain
    #   the number of times that male 1 wins by subtraction
    
    n.bouts = template.data$num0wins + template.data$num1wins ;

    for (pair.index in c(1:length(n.bouts))) {
        matchp = runif(1, 0, 1);
        if (matchp < p.Even) {
            p.zero = 0.5;
        }
        else if (matchp < (p.Even + p.SW)) {
            p.zero = w;
        }
        else {
            p.zero = 1 - w;
        }
        r = runif(n.bouts[pair.index], 0, 1);
        num.0.won.this.rep = length(r[r <= p.zero]);
        num0wins.vec = c(num0wins.vec, num.0.won.this.rep);
    }
    num1wins.vec = n.bouts - num0wins.vec;
    return(list(num0wins=num0wins.vec, num1wins=num1wins.vec));
}
null.params = c(NA, 0.55);
n.sims = 1000;
summary = calc.lrt.statistic(data, null.params);

print(paste("MLE of prob strong = ", summary$global$par[1]));
print(paste("MLE of w = ", summary$global$par[2]));
print(paste("lnL at MLEs = ", -summary$global$value));
print(paste("L at MLEs = ", exp(-summary$global$value)));
print(paste("MLE of prob strong at null = ", summary$null$par[1]));
print(paste("null of w = ", summary$null$par[2]));
print(paste("lnL of null = ", -summary$null$value));
print(paste("L of null = ", exp(-summary$null$value)));
print(paste("2* log-likelihood ratio = ", summary$lrt));

null.lrt = c()
for (i in c(1:n.sims)) {
    sim.data = simulate.data(data, summary$null$par);
    sim.summary = calc.lrt.statistic(sim.data, null.params);
    null.lrt = c(null.lrt, sim.summary$lrt);
}
null.lrt = sort(null.lrt);
print(paste("Approx critical value for P=0.05 =", null.lrt[1 + n.sims*0.95]));
print(paste("Approx P-value =", length(null.lrt[null.lrt < summary$lrt])/n.sims));
