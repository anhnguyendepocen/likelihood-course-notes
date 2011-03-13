#!/usr/bin/env R
data = read.table("simpleData.txt", header=TRUE);

INITIAL.PARAMETER.GUESS = c(0.75, 0.75)

calc.ln.likelihood = function(theta) {

    p = theta[1];
    w = theta[2];

    ###########################################################################
    # Return -inf if the parameters are out of range.  This will keep the optimizer
    #  from returning "illegal" values of the parameters.
    #
    if (p < 0 || p > 1.0 || w < 0.5 || w > 1.0) {
        return (-Inf);
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
    return (sum(log.p.data));
}

fn = function(theta) {
    return (-calc.ln.likelihood(theta));
}

print(calc.ln.likelihood(c(0.75, 0.75)))

print(calc.ln.likelihood(c(0.75, 1.75)))

print(calc.ln.likelihood(c(0.75, 0.5)))
