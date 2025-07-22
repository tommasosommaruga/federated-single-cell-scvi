import os
import sys
import json
import torch
import anndata
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

# Project-specific modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from federated_scvi_flower.data_utils_scvi import load_partitioned_anndata, ensure_hvg_genes, load_hvg_list, load_batch_list
from federated_scvi_flower.model_utils_scvi import train_scvi_with_loss_tracking, get_scvi_model, evaluate_scvi, setup_scvi_anndata

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = ""
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()

print("Last run with scvi-tools version:", scvi.__version__)
num_partitions = 3
partition_id = 2

# Directories
save_dir = "data"
model_dir = os.path.join(save_dir, f"independent_client_{partition_id}")
hvg_path = os.path.join(save_dir, "hvg_list.json")
batch_list_path = os.path.join(save_dir, "batch_list.json")

# Load data
adata = load_partitioned_anndata(partition_id=partition_id, num_partitions=num_partitions)
adata_test = anndata.read_h5ad(os.path.join("data", "pancreas_test.h5ad"))

# HVG filtering
hvg_list = load_hvg_list(hvg_path)
adata_test = ensure_hvg_genes(adata_test, hvg_list)

# AnnData setup
all_batches = load_batch_list(batch_list_path)
setup_scvi_anndata(adata, all_batches=all_batches)

# Load or train model
if os.path.exists(model_dir):
    print(f"[independent_client_{partition_id}] Loading pre-trained model...")
    scvi_model = scvi.model.SCVI.load(model_dir, adata=adata)
    train_loss = evaluate_scvi(scvi_model, adata)
    test_loss = evaluate_scvi(scvi_model, adata_test)
    print(f"[independent_client_{partition_id}] Train loss: {train_loss}")
    print(f"[independent_client_{partition_id}] Test loss: {test_loss}")
else:
    print(f"[independent_client_{partition_id}] Training SCVI model...")
    scvi_model = get_scvi_model(adata, hvg_list, partition_id=partition_id)
    train_losses, test_losses = train_scvi_with_loss_tracking(scvi_model, adata, adata_test, max_epochs=100)

    final_train_loss = train_losses[-1] if train_losses else 0.0
    final_test_loss = test_losses[-1] if test_losses else 0.0
    print(f"[independent_client_{partition_id}] Training complete. Final train loss: {final_train_loss}")
    print(f"[independent_client_{partition_id}] Test evaluation complete. Final test loss: {final_test_loss}")

    scvi_model.save(model_dir, overwrite=True)

    # Save loss curves
    loss_log_path = os.path.join(save_dir, f"loss_curve_client_{partition_id}.csv")
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "test_loss": test_losses
    })
    loss_df.to_csv(loss_log_path, index=False)
    print(f"[independent_client_{partition_id}] Loss curve saved to {loss_log_path}")

# Latent representation + plotting
adata.obsm["X_scVI"] = scvi_model.get_latent_representation()
sc.settings.figdir = f"figures/independent_client_{partition_id}"
os.makedirs(sc.settings.figdir, exist_ok=True)

sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=["tech", "celltype"],
    frameon=False,
    show=False,
    save=f"_client_{partition_id}_plot.png"
)

