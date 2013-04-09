num.it = 100000
num.coins = 4
state = 1
data = c(4, 3)

calc.likelihood = function(theta) {
    n.fair = num.coins - theta;
    n.fair.heads = data - theta;
    flip.prob = 0.5**n.fair.heads;
    return(prod(choose(n.fair, n.fair.heads)*flip.prob))
}

likelihood = calc.likelihood(state);


mcmc.samples = rep(0, num.coins + 1);
# This is MCMC using the Metropolis algorithm:
cat("Iter\tlike\ttheta\n", file="MCMC-samples.txt", append=FALSE);
for (i in 1:num.it) {
    if (i %% 1000 == 0) {
        print(paste('iteration =', i));
    }
    write(paste(i, likelihood, state, sep='\t'), file="MCMC-samples.txt", append=TRUE);
    mcmc.samples[state + 1] = mcmc.samples[state + 1] + 1;
    prev.likelihood = likelihood;

    # propose a state from among the adjacent states
    if (runif(1, 0, 1) < 0.5) {
        proposed = state + 1;
        if (proposed > num.coins) {
            proposed = 0;
        }
    }
    else {
        proposed = state - 1;
        if (proposed < 0) {
            proposed = num.coins;
        }
    }
    # Prior ratio is 1.0, so we could ignore it...
    prior.ratio = (0.2)/(0.2);

    likelihood = calc.likelihood(proposed);
    likelihood.ratio = likelihood/prev.likelihood;

    posterior.ratio = likelihood.ratio*prior.ratio;

    # Hastings ratio is 1.0, so we can ignore it...
    hastings.ratio = (0.5)/(0.5);

    acceptance.ratio = posterior.ratio*hastings.ratio;
    #print state, '->', proposed, 'alpha=', acceptance.ratio, 'prev.likelihood=',prev.likelihood, 'proposed.likelihood=',likelihood
    if (runif(1, 0, 1) < acceptance.ratio) {
        state = proposed;
    }
    else {
        # reject
        # state = old state already, so we don't have to change it
        likelihood = prev.likelihood;
    }
}
cat("Posterior probabilities from MCMC\n", file="MCMC-output.txt", append=FALSE);
for (state in 1:(1 + num.coins)) {
    cat(paste(state - 1, mcmc.samples[state]/num.it, '\n', sep=" "), file="MCMC-output.txt", append=TRUE);
}

cat("\nTrue Posterior probabilities (calculated analytically)\n", file="MCMC-output.txt", append=TRUE);
likelihood.list = unlist(lapply(0:num.coins, calc.likelihood));
marginal.prob = sum(likelihood.list);
for (state in 1:(1 + num.coins)) {
    cat(paste(state - 1 , likelihood.list[state]/marginal.prob, '\n', sep=" "), file="MCMC-output.txt", append=TRUE);
}
