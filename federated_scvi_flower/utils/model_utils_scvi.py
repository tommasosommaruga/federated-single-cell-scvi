import scvi
import torch
import numpy as np
from federated_scvi_flower.utils.data_utils_scvi import ensure_hvg_genes
import scanpy as sc
import os
import pandas as pd
from lightning.pytorch.callbacks import Callback

def setup_scvi_anndata(adata, all_batches=None):
    adata.obs["batch"] = adata.obs.get("batch", adata.obs.get("tech", "batch0"))
    
    if all_batches is not None: adata.obs["batch"] = pd.Categorical(adata.obs["batch"], categories=all_batches)
    else: adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Mark all genes as highly variable if that info is present
    if "highly_variable" in adata.var.columns: adata.var["highly_variable"] = True
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer="counts")

def get_scvi_model(adata, hvg_list=None, partition_id=None):
    if hvg_list is not None and list(adata.var_names) != list(hvg_list):
        raise ValueError("Gene list mismatch between AnnData and HVG list")
    model = scvi.model.SCVI(adata, use_layer_norm="both", use_batch_norm="none", encode_covariates=True, dropout_rate=0.2, n_layers=2)
    return model

def get_weights(model):
    # scvi.model.SCVI wraps the torch model in .module
    return [v.detach().cpu().numpy() for v in model.module.state_dict().values()]

def set_weights(model, weights):
    state_dict = model.module.state_dict()
    new_state_dict = {k: torch.tensor(w) for k, w in zip(state_dict.keys(), weights)}
    model.module.load_state_dict(new_state_dict, strict=True)

def evaluate_scvi(model, adata):
    # Ensure 'batch' exists in adata.obs for evaluation
    if "batch" not in adata.obs:
        adata.obs["batch"] = adata.obs["tech"] if "tech" in adata.obs else "batch0"
    # Evaluate ELBO on the given AnnData
    elbo = model.get_elbo(adata)
    return -float(elbo)

def plot_latent_umap(adata, latent_key="X_scVI", color=["tech", "celltype"], save=None, show=True):
    # Ensure directory exists if saving
    if save is not None:
        dirpath = os.path.dirname(os.path.abspath(save))
        if dirpath and dirpath != os.path.abspath(""):
            print(f"DEBUG: Creating directory (and parents if needed): {dirpath}")
            os.makedirs(dirpath, exist_ok=True)
    # Compute neighbors and UMAP if not already present
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep=latent_key)
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
        sc.tl.umap(adata)
    sc.pl.umap(adata, color=color, save=save, show=show) 

# TRAINING ALL EPOCHS TOGETHER IF YOU DON'T NEED TO TRACK LOSS, FEDERATED APPROACH
def train_scvi(model, adata, max_epochs=10):
    model.train(max_epochs=max_epochs)
    # Use .iloc[-1] to get the last value by position, not by index and Negate to get the true ELBO (should be negative, like model.get_elbo)
    return float(model.history["elbo_train"].iloc[-1]) if "elbo_train" in model.history else 0.0

# TRAINING WITH LOSS TRACKING BUT UNEFFICIENT (NO CALLBACK)
# def train_scvi_tracking(model, adata_train, adata_test, max_epochs=10):
#     train_losses = []
#     test_losses = []
#     for epoch in range(max_epochs):
#         model.train(max_epochs=1)  # Train one epoch at a time
        
#         # Evaluate training loss
#         train_loss = evaluate_scvi(model, adata_train)
#         train_losses.append(train_loss)
        
#         # Evaluate test loss
#         test_loss = evaluate_scvi(model, adata_test)
#         test_losses.append(test_loss)
        
#         print(f"Epoch {epoch + 1}/{max_epochs} - Train loss: {train_loss}, Test loss: {test_loss}")
        
#     return train_losses, test_losses

class SCVILossLogger(Callback):
    def __init__(self, model, adata_train, adata_test):
        self.model = model
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.train_losses = []
        self.test_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = evaluate_scvi(self.model, self.adata_train)
        test_loss = evaluate_scvi(self.model, self.adata_test)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        print(f"Epoch {trainer.current_epoch + 1} - Train loss: {train_loss:.2f}, Test loss: {test_loss:.2f}")

# Training function
def train_scvi_with_loss_tracking(model, adata_train, adata_test, max_epochs=100):
    loss_logger = SCVILossLogger(model, adata_train, adata_test)
    model.train(max_epochs=max_epochs, callbacks=[loss_logger])
    return loss_logger.train_losses, loss_logger.test_losses