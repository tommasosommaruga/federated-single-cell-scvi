import scvi
import torch
import numpy as np
from federated_scvi_flower.data_utils_scvi import ensure_hvg_genes
import scanpy as sc

def get_scvi_model(adata, hvg_list=None, partition_id=None):
    if hvg_list is not None:
        adata = ensure_hvg_genes(adata, hvg_list, partition_id=partition_id)
    # Remove or comment out repetitive debug prints
    # print(f"[DEBUG][partition {partition_id}] HVG list length: {len(hvg_list)}, first 20: {hvg_list[:20]}")
    # print(f"[DEBUG][partition {partition_id}] AnnData var_names length: {len(adata.var_names)}, first 20: {list(adata.var_names[:20])}")
    # print(f"[DEBUG][partition {partition_id}] Genes in HVG list but not in AnnData: {set(hvg_list) - set(adata.var_names)}")
    # print(f"[DEBUG][partition {partition_id}] Genes in AnnData but not in HVG list: {set(adata.var_names) - set(hvg_list)}")
    # FATAL CHECK: Ensure adata.var_names matches hvg_list exactly
    if hvg_list is not None and list(adata.var_names) != list(hvg_list):
        print(f"[FATAL][partition {partition_id}] adata.var_names and hvg_list do not match!")
        print(f"adata.var_names[:20]: {list(adata.var_names[:20])}")
        print(f"hvg_list[:20]: {hvg_list[:20]}")
        print(f"adata.var_names length: {len(adata.var_names)}, hvg_list length: {len(hvg_list)}")
        raise ValueError("Gene list mismatch between AnnData and HVG list")
    # Check and fix highly_variable column
    if 'highly_variable' in adata.var.columns:
        adata.var['highly_variable'] = True
    # Ensure 'batch' column exists in adata.obs
    if "batch" not in adata.obs:
        adata.obs["batch"] = adata.obs["tech"] if "tech" in adata.obs else "batch0"
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer="counts")
    model = scvi.model.SCVI(adata)
    return model

def get_weights(model):
    # scvi.model.SCVI wraps the torch model in .module
    return [v.detach().cpu().numpy() for v in model.module.state_dict().values()]

def set_weights(model, weights):
    state_dict = model.module.state_dict()
    new_state_dict = {k: torch.tensor(w) for k, w in zip(state_dict.keys(), weights)}
    model.module.load_state_dict(new_state_dict, strict=True)

def train_scvi(model, adata, max_epochs=10):
    model.train(max_epochs=max_epochs)
    # Use .iloc[-1] to get the last value by position, not by index and Negate to get the true ELBO (should be negative, like model.get_elbo)
    return -float(model.history["elbo_train"].iloc[-1]) if "elbo_train" in model.history else 0.0

def evaluate_scvi(model, adata):
    # Ensure 'batch' exists in adata.obs for evaluation
    if "batch" not in adata.obs:
        adata.obs["batch"] = adata.obs["tech"] if "tech" in adata.obs else "batch0"
    # Evaluate ELBO on the given AnnData
    elbo = model.get_elbo(adata)
    return float(elbo)

def plot_latent_umap(adata, latent_key="X_scVI", color=["tech", "celltype"], save=None, show=True):
    import os
    # Ensure directory exists if saving
    if save is not None:
        dirpath = os.path.dirname(os.path.abspath(save))
        if dirpath and dirpath != os.path.abspath(""):
            print(f"DEBUG: Creating directory (and parents if needed): {dirpath}")
            os.makedirs(dirpath, exist_ok=True)
    # Compute neighbors and UMAP if not already present
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep=latent_key)
        sc.tl.umap(adata)
    sc.pl.umap(adata, color=color, save=save, show=show) 