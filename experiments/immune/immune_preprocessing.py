# Human Immune Preprocessing
# 
# Batches 
# ['10X' 'Freytag' 'Oetjen_A' 'Oetjen_P' 'Oetjen_U' 'Sun_sample1_CS' 'Sun_sample2_KC' 'Sun_sample3_TB' 'Sun_sample4_TC' 'Villani']
# 
# 
# Celltypes 
# ['CD10+ B cells' 'CD14+ Monocytes' 'CD16+ Monocytes' 'CD20+ B cells' 'CD4+ T cells' 'CD8+ T cells' 'Erythrocytes' 'Erythroid progenitors' 'HSPCs' 'Megakaryocyte progenitors' 'Monocyte progenitors' 'Monocyte-derived dendritic cells' 'NK cells' 'NKT cells' 'Plasma cells' 'Plasmacytoid dendritic cells']
# based on the scib-pipeline: https://github.com/theislab/scib-pipeline/blob/main/scripts/preprocessing/runPP.py

# wget https://figshare.com/ndownloader/files/25717328 -O immune.h5ad

import scanpy as sc
import scib
import numpy as np
import os

adata = sc.read('immune.h5ad')
hvgs = adata.var.index

# remove HVG if already precomputed
if 'highly_variable' in adata.var:
    del adata.var['highly_variable']

adata = scib.preprocessing.hvg_batch(
    adata,
    batch_key='batch',
    target_genes=2000,
    adataOut=True
)

# we do scaling as Luecken et al. reported that it results in higher batch removal scores in general
adata = scib.preprocessing.scale_batch(adata, "batch")

metadata = adata.obs[['final_annotation', 'batch',
                      'tissue', 'chemistry', 'study']]
metadata.rename(columns={"final_annotation": "celltype"}, inplace=True)
metadata.reset_index()
metadata.to_csv(os.path.join("Immune_ALL_human_meta_batchaware.csv"),
              sep=',')

sc.pp.pca(adata, n_comps=50, 
        use_highly_variable=True,
        svd_solver='arpack')
np.savetxt(os.path.join("Immune_ALL_human_pca.csv"),
          X=adata.obsm['X_pca'],
          delimiter=',')