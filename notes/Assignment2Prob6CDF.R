#!/usr/bin/env R
pdf("Assignment2Prob6.pdf")

n = 75
p = 0.4
omp = 1 - p
ten = rbinom(10, n, p)
twenty = rbinom(20, n, p)
hundred = rbinom(100, n, p)
plot(ecdf(ten), do.points=FALSE, verticals=TRUE, 
    xlim=c(0,n), main="CDFs of binomial and normal", lty=4)
lines(ecdf(twenty), do.points=FALSE, verticals=TRUE, lty=2)
lines(ecdf(hundred), do.points=FALSE, verticals=TRUE,lty=3)
x = seq(0, n, 0.01)
lines(x, pnorm(x, mean=n*p, sqrt(n*p*omp)), lty=1)
legend(50, .5, c("Binom-10", "Binom-20", "Binom-100", "Normal"), lty=c(4,2,3,1))
dev.off()
