d <- read.table("pboot.txt", header=TRUE, sep="\t");
pdf("lrtnull.pdf");
hist(d$lrt, main="Null Distribution of LRT statistics");
dev.off();
