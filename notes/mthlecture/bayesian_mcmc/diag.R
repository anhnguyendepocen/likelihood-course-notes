library(coda)
first = read.table('losamp2.txt', header=T, sep='\t')
first$Iteration = NULL
first.draws = mcmc(first)
summary(first.draws)


second = read.table('highsamp2.txt', header=T, sep='\t')
second$Iteration = NULL
second.draws = mcmc(second)
summary(second.draws)


all.draws = mcmc.list(list(first.draws, second.draws))
gelman.diag(all.draws)
gelman.plot(all.draws)