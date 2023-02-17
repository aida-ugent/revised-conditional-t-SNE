# Might be worth to use the (Graph) Lisi normalization from Theis Lab?
# This would allow for a comparision of the results in their paper
# https://www.nature.com/articles/s41592-021-01336-8#code-availability

library("scPOP")

args <- commandArgs(trailingOnly = TRUE)
# trailingOnly=TRUE means that only your arguments are returned, check:
# print(commandArgs(trailingOnly=FALSE))

data_path <- args[1]
destination_path <- args[2]
perplexity <- as.integer(args[3])
rm(args)
print(paste("Computing LISI with perplexity", perplexity))

embedding = read.table(data_path, sep=",", header=TRUE)

features = embedding[, c('x', 'y')]
metadata <- embedding[colnames(embedding)[colnames(embedding) != 'x' & colnames(embedding) != 'y']]

lisi_res <- lisi(features, metadata, colnames(metadata), perplexity=perplexity)

# normalize lisi values
for (i in range(1:length(colnames(metadata)))) {
    probabilities <- table(metadata[, i])/length(metadata[, i])
    expected_lisi <- 1/ sum(probabilities^2)
    cname <- paste0(colnames(metadata)[i], "_", "norm")
    lisi_res[, cname] <- lisi_res[, i]/expected_lisi
}

write.table(lisi_res, file=destination_path, sep=',', row.names = FALSE, quote = FALSE)