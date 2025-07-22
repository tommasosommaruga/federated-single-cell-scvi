import os
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federated_scvi_flower.data_utils_scvi import load_batch_list, ensure_hvg_genes, load_hvg_list
from federated_scvi_flower.model_utils_scvi import get_scvi_model, train_scvi, evaluate_scvi, setup_scvi_anndata

# Force CPU usage (disable CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set seed for reproducibility
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

# Set some visual style
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()

# Optional: improve matmul precision
torch.set_float32_matmul_precision("high")

# Dataset location
save_dir = "data"
model_dir = os.path.join(save_dir, "centralised_trained_scvi_model")
# adata_path = os.path.join(save_dir, "pancreas.h5ad")
# adata = sc.read(adata_path, backup_url="https://figshare.com/ndownloader/files/24539828",)

adata = anndata.read_h5ad("data/pancreas_train.h5ad")
adata_test = anndata.read_h5ad("data/pancreas_test.h5ad")

hvg_list = load_hvg_list("data/hvg_list.json")
adata = ensure_hvg_genes(adata, hvg_list)
adata_test = ensure_hvg_genes(adata_test, hvg_list)
all_batches = load_batch_list("data/batch_list.json")
setup_scvi_anndata(adata, all_batches=all_batches)
scvi_ref = get_scvi_model(adata)

# Check if the model already exists
if os.path.exists(model_dir):
    print("Loading pre-trained model from disk...")
    scvi_ref = scvi.model.SCVI.load(model_dir, adata=adata)
    train_loss = evaluate_scvi(scvi_ref, adata)
    print(f"[centralised] Model loaded. Train loss: {train_loss}")
    test_loss = evaluate_scvi(scvi_ref, adata_test)
    print(f"[centralised] Evaluation complete. Test loss: {test_loss}")
else:
    print(f"[centralised] AnnData shape before SCVI init: {adata.shape}, genes: {list(adata.var_names[:10])}")
    print("[centralised] Training SCVI model...")
    train_losses, test_losses = train_scvi(scvi_ref, adata, adata_test, max_epochs=100)
    final_train_loss = train_losses[-1] if train_losses else 0.0
    final_test_loss = test_losses[-1] if test_losses else 0.0

    print(f"[centralised] Training complete. Final train loss: {final_train_loss}")
    print(f"[centralised] Evaluation complete. Test loss: {final_test_loss}")
    scvi_ref.save(model_dir, overwrite=True)
    loss_log_path = os.path.join(save_dir, f"loss_curve_centralised.csv")

    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "test_loss": test_losses
    })

    loss_df.to_csv(loss_log_path, index=False)
    print(f"[centralised] Train and test loss per epoch saved to {loss_log_path}")


# Store latent representation
SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = scvi_ref.get_latent_representation()

save_dir = "figures/centralised"
os.makedirs(save_dir, exist_ok=True)
sc.settings.figdir = save_dir

# Run clustering and UMAP
sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
sc.tl.umap(adata)

# Plot and save UMAP
sc.pl.umap(
    adata,
    color=["tech", "celltype"],
    frameon=False,
    show=False,         # Do not display plot in window
    save="_plot.png"    # Will save under `figures/umap_plot.png` by default
)
