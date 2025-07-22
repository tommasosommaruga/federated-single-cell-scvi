import anndata
import numpy as np
import json
import torch
import os
import matplotlib.pyplot as plt
import scanpy as sc

from federated_scvi_flower.model_utils_scvi import get_scvi_model, setup_scvi_anndata
from federated_scvi_flower.data_utils_scvi import ensure_hvg_genes, load_batch_list

# Load train/test data and HVG list
adata_train = anndata.read_h5ad("data/pancreas_train.h5ad")
adata_test = anndata.read_h5ad("data/pancreas_test.h5ad")
with open("data/hvg_list.json") as f:
    hvg_list = json.load(f)

# Ensure HVG genes are consistent between train and test
adata_train = ensure_hvg_genes(adata_train, hvg_list)
adata_test = ensure_hvg_genes(adata_test, hvg_list)

# Assign batch information from 'tech' and convert to categorical
adata_train.obs["batch"] = adata_train.obs["tech"].astype("category")
adata_test.obs["batch"] = adata_test.obs["tech"].astype("category")

# Get all batches from train and test datasets to unify batch categories
all_batches = load_batch_list("data/batch_list.json")

# Setup AnnData for scVI with unified batch categories
setup_scvi_anndata(adata_train, all_batches=all_batches)
setup_scvi_anndata(adata_test, all_batches=all_batches)

# Initialize model on train data
model = get_scvi_model(adata_train, hvg_list)

# Load saved model weights (handle DataParallel wrapping if present)
state_dict = torch.load("federated_scvi_flower/final_server_model.pt")
if hasattr(model, "module"):
    model.module.load_state_dict(state_dict, strict=True)
else:
    model.load_state_dict(state_dict, strict=True)
model.is_trained = True

# Compute latent embeddings for train and test
adata_train.obsm["X_scVI"] = model.get_latent_representation()
adata_test.obsm["X_scVI"] = model.get_latent_representation(adata_test)

# Combine train and test datasets for joint visualization
adata_combined = anndata.concat(
    [adata_train, adata_test],
    label="dataset",
    keys=["train", "test"],
    join="inner",
    index_unique=None,
)

# Combine latent embeddings accordingly
combined_latent = np.vstack([adata_train.obsm["X_scVI"], adata_test.obsm["X_scVI"]])
adata_combined.obsm["X_scVI"] = combined_latent

# Prepare directory for saving figures
save_dir = "figures/federated"
os.makedirs(save_dir, exist_ok=True)
sc.settings.figdir = save_dir

# Compute neighbors and UMAP on latent space
sc.pp.neighbors(adata_combined, use_rep="X_scVI")
sc.tl.umap(adata_combined)

# Plot UMAP colored by dataset, technology, and cell type
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

sc.pl.umap(adata_combined, color="dataset", ax=axes[0], show=False, title="UMAP by Dataset")
sc.pl.umap(adata_combined, color="tech", ax=axes[1], show=False, title="UMAP by Technology")
sc.pl.umap(adata_combined, color="celltype", ax=axes[2], show=False, title="UMAP by Cell Type")

# Save figure
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "combined_umap_comparison.png"))
plt.close(fig)
print(f'File combined_umap_comparison.png saved in the {save_dir} folder!')