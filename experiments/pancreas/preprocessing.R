#devtools::install_github('satijalab/seurat-data')
library(Seurat)
library(SeuratData)
library(ggplot2)

InstallData("panc8")
data("panc8")
panc8
#?panc8

table(panc8$dataset)
table(panc8$tech)
table(panc8$celltype)

panc8 <- NormalizeData(panc8)
panc8 <- FindVariableFeatures(object = panc8)
panc8 <- ScaleData(object = panc8)
panc8 <- RunPCA(object = panc8)

saveRDS(panc8@meta.data, file = "metadata.rds")
saveRDS(Embeddings(object = panc8, reduction = "pca"), file = "pancreas_pca.rds")