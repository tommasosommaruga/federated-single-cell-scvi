import os
import anndata
import numpy as np
import scanpy as sc
import json
import pandas as pd
import scipy.sparse
from anndata import AnnData
import logging

def ensure_hvg_genes(adata, hvg_list, partition_id=None):
    if partition_id is not None:
        logging.info(f"[ensure_hvg_genes][partition {partition_id}] BEFORE: shape={adata.shape}, genes={list(adata.var_names[:10])}")
    else:
        logging.info(f"[ensure_hvg_genes] BEFORE: shape={adata.shape}, genes={list(adata.var_names[:10])}")
    # Add missing genes as zero columns and reorder to match hvg_list
    missing_genes = [g for g in hvg_list if g not in adata.var_names]
    if missing_genes:
        logging.warning(f"[ensure_hvg_genes][partition {partition_id}] Filling {len(missing_genes)} missing genes with zeros: {missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")
        n_cells = adata.n_obs
        zero_mat = scipy.sparse.csr_matrix((n_cells, len(missing_genes)))
        zero_adata = AnnData(
            X=zero_mat,
            obs=adata.obs.copy(),
            var=None
        )
        zero_adata.var_names = list(missing_genes)
        adata = anndata.concat([adata, zero_adata], axis=1, join='outer', merge='first')
    adata = adata[:, hvg_list].copy()
    gene_count = adata.shape[1]
    if partition_id is not None:
        logging.info(f"[ensure_hvg_genes][partition {partition_id}] AFTER: shape={adata.shape}, genes={list(adata.var_names[:10])}")
        if gene_count != len(hvg_list):
            logging.error(f"[ERROR][partition {partition_id}] Gene count mismatch after ensure_hvg_genes: shape={adata.shape}, expected={len(hvg_list)}, got={gene_count}, genes={list(adata.var_names[:20])}")
    else:
        logging.info(f"[ensure_hvg_genes] AFTER: shape={adata.shape}, genes={list(adata.var_names[:10])}")
        if gene_count != len(hvg_list):
            logging.error(f"[ERROR] Gene count mismatch after ensure_hvg_genes: shape={adata.shape}, expected={len(hvg_list)}, got={gene_count}, genes={list(adata.var_names[:20])}")
    return adata

def load_partitioned_anndata(partition_id, num_partitions):
    adata_path = os.path.join("data", "pancreas.h5ad")
    adata = anndata.read_h5ad(adata_path)
    # Convert counts layer to int32 if present
    if hasattr(adata, 'layers') and 'counts' in adata.layers:
        adata.layers['counts'] = adata.layers['counts'].astype('int32')
    # Apply the same filter as the server
    if "tech" in adata.obs:
        ref_mask = (~adata.obs["tech"].isin(["smartseq2", "celseq2"])).values
        adata = adata[ref_mask].copy()
    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Loaded filtered AnnData: shape={adata.shape}, genes={list(adata.var_names[:10])}")
    hvg_path = os.path.join("data", "hvg_list.json")
    with open(hvg_path) as f:
        hvg_list = json.load(f)
    if "tech" in adata.obs:
        techs = adata.obs["tech"].unique().tolist()
        techs.sort()
        if num_partitions <= len(techs):
            tech = techs[partition_id % len(techs)]
            mask = adata.obs["tech"] == tech
            adata_part = adata[mask].copy()
        else:
            idxs = np.array_split(np.arange(adata.n_obs), num_partitions)
            adata_part = adata[idxs[partition_id]].copy()
    else:
        idxs = np.array_split(np.arange(adata.n_obs), num_partitions)
        adata_part = adata[idxs[partition_id]].copy()
    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Partitioned AnnData: shape={adata_part.shape}, genes={list(adata_part.var_names[:10])}")
    adata_part = ensure_hvg_genes(adata_part, hvg_list, partition_id=partition_id)
    return adata_part 