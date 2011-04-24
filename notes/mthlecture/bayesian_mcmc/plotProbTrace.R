fn = commandArgs(TRUE)
d = read.table(fn, header=TRUE, sep="\t");
pdf(paste(fn, '.pdf', sep=""));
plot(d$Gen, d$PrZero, xlab="i", ylab="Pr(x_i=0)", type="l", ylim=c(0,1));
dev.off();

