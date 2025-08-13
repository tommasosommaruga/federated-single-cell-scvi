import numpy as np
import scanpy as sc
import anndata as ad
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

adata = ad.read_h5ad("0_data/pancreas.h5ad")
print(adata.X.min(), adata.X.max())
print("Missing values:", np.isnan(adata.X).sum())
missing_genes = [gene for gene in adata.var_names if gene is None or gene == ""]
print("Missing genes:", missing_genes)