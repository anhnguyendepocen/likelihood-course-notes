fn = commandArgs(TRUE)
d = read.table(fn, header=TRUE, sep="\t");
pdf(paste(fn, '.pdf', sep=""));
plot(d$Gen, d$theta, xlab="i", ylab="theta", type="l", ylim=c(0,4));
dev.off();

