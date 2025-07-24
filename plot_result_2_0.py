import anndata
import numpy as np
import json
import torch
import os
import matplotlib.pyplot as plt
import scanpy as sc
from scvi.model import SCVI
from federated_scvi_flower.model_utils_scvi import get_scvi_model, setup_scvi_anndata
from federated_scvi_flower.data_utils_scvi import ensure_hvg_genes, load_batch_list, load_hvg_list


def generate_scvi_umap(adata_train: anndata.AnnData = None, adata_test: anndata.AnnData = None,
                        hvg_path: str = "data/hvg_list.json", batch_list_path: str = "data/batch_list.json", 
                        model_weights_path: str = "federated_scvi_flower/final_server_model.pt", 
                        save_dir: str = "figures/federated", show_plot: bool = False, plot_type: str = "federated"):

    # Load datasets if not provided
    if adata_train is None: adata_train = anndata.read_h5ad("data/pancreas_train.h5ad")
    if adata_test is None: adata_test = anndata.read_h5ad("data/pancreas_test.h5ad")
    hvg_list = load_hvg_list(hvg_path)

    if plot_type == "federated":
        all_batches = load_batch_list(batch_list_path)

        # Filter for HVGs
        adata_train = ensure_hvg_genes(adata_train, hvg_list)
        adata_test = ensure_hvg_genes(adata_test, hvg_list)

        # Assign and unify batch info
        adata_train.obs["batch"] = adata_train.obs["tech"].astype("category")
        adata_test.obs["batch"] = adata_test.obs["tech"].astype("category")

        setup_scvi_anndata(adata_train, all_batches=all_batches)
        setup_scvi_anndata(adata_test, all_batches=all_batches)

    model = SCVI.load(model_weights_path, adata_train)
    model.is_trained = True

    # Get latent embeddings
    adata_train.obsm["X_scVI"] = model.get_latent_representation()
    adata_test.obsm["X_scVI"] = model.get_latent_representation(adata_test)

    # Combine train and test data
    adata_combined = anndata.concat(
        [adata_train, adata_test],
        label="dataset",
        keys=["train", "test"],
        join="inner",
        index_unique=None,
    )

    adata_combined.obsm["X_scVI"] = np.vstack([
        adata_train.obsm["X_scVI"],
        adata_test.obsm["X_scVI"]
    ])

    # Make output folder
    os.makedirs(save_dir, exist_ok=True)
    sc.settings.figdir = save_dir

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_combined, use_rep="X_scVI")
    sc.tl.leiden(adata_combined, flavor="igraph", n_iterations=2)
    sc.tl.umap(adata_combined)

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(adata_combined, color="dataset", ax=axes[0], show=False, title="UMAP by Dataset")
    sc.pl.umap(adata_combined, color="tech", ax=axes[1], show=False, title="UMAP by Technology")
    sc.pl.umap(adata_combined, color="celltype", ax=axes[2], show=False, title="UMAP by Cell Type")

    fig.tight_layout()
    file_path = os.path.join(save_dir, f"umap_comparison_{plot_type}.png")
    fig.savefig(file_path)
    if show_plot: plt.show()
    else: plt.close(fig)

    print(f"UMAP figure saved at: {file_path}")

# if __name__ == "__main__":
#     generate_scvi_umap()