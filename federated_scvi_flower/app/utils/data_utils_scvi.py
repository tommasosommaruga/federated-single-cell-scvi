import os
import anndata
import numpy as np
import scanpy as sc
import json
import pandas as pd
import scipy.sparse
from anndata import AnnData
import logging

# Utility to load HVG list
def load_hvg_list(hvg_list_path):
    with open(hvg_list_path) as f:
        return json.load(f)

def load_batch_list(path: str):
    with open(path, "r") as f:
        return json.load(f)

def create_dummy_adata(num_cells=10, num_genes=2000, batches=["batch_1", "batch_2"]):
    X = np.random.normal(0, 1, size=(num_cells, num_genes)).astype(np.float32)
    obs = pd.DataFrame({"batch": np.random.choice(batches, num_cells)})
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(num_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata
       
def ensure_hvg_genes(adata, hvg_list, partition_id=None):
    # Add missing genes as zero columns and reorder to match hvg_list
    missing_genes = [g for g in hvg_list if g not in adata.var_names]
    if missing_genes:
        logging.warning(f"[ensure_hvg_genes][partition {partition_id}] Filling {len(missing_genes)} missing genes with zeros: {missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")
        n_cells = adata.n_obs
        zero_mat = scipy.sparse.csr_matrix((n_cells, len(missing_genes)))
        zero_adata = AnnData(X=zero_mat,obs=adata.obs.copy(),var=None)
        zero_adata.var_names = list(missing_genes)
        adata = anndata.concat([adata, zero_adata], axis=1, join='outer', merge='first')
    adata = adata[:, hvg_list].copy()
    return adata

def load_partitioned_anndata(partition_id, num_partitions, prefix=None):
    if not prefix:
        prefix = "."
    adata_path = os.path.join(prefix, "0_data", "pancreas_train.h5ad")
    adata = anndata.read_h5ad(adata_path)
    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Loaded filtered AnnData: shape={adata.shape}, genes={list(adata.var_names[:10])}")
    
    if "tech" in adata.obs:
        techs = sorted(adata.obs["tech"].unique().tolist())
        tech_groups = np.array_split(techs, num_partitions)
        client_techs = tech_groups[partition_id]
        mask = adata.obs["tech"].isin(client_techs)
        adata_part = adata[mask].copy()
    else:
        idxs = np.array_split(np.arange(adata.n_obs), num_partitions)
        adata_part = adata[idxs[partition_id]].copy()

    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Partitioned AnnData: shape={adata_part.shape}, genes={list(adata_part.var_names[:10])}")
    return adata_part
