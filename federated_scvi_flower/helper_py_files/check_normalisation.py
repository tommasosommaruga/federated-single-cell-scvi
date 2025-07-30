import numpy as np
import scanpy as sc
import anndata as ad
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

adata = ad.read_h5ad("data/pancreas_val.h5ad")
print(adata.X.min(), adata.X.max())
