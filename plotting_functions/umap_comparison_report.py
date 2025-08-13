import scanpy as sc
import scvi
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federated_scvi_flower.app.utils.model_utils_scvi import setup_scvi_anndata, get_scvi_model

PANCREAS_DATA_PATH = "0_data/pancreas_test.h5ad"
CENTRALIZED_MODEL_PATH = "models/centralised_model"
FEDERATED_MODEL_PATH = "federated_scvi_flower/app/models/3_clients_20_rounds_5_epochs/model.pt"   

BATCH_KEY = "tech"
CELL_TYPE_KEY = "celltype"

# 1. Load the Testing Dataset
print(f"Loading data from: {PANCREAS_DATA_PATH}")
adata = sc.read_h5ad(PANCREAS_DATA_PATH)
print("Data loaded successfully.")

# 2. Preprocess Raw Data for UMAP
# This step prepares the raw data for UMAP.
print("Preprocessing raw data for UMAP...")
with open("0_data/hvg_list.json") as f:
    hvg_list = json.load(f)
adata = adata[:, hvg_list] 
adata_raw_umap = adata.copy()
sc.tl.pca(adata_raw_umap, svd_solver='arpack')
sc.pp.neighbors(adata_raw_umap, use_rep='X_pca')
sc.tl.umap(adata_raw_umap)
print("Raw data UMAP calculated.")
# 3. Load SCVI Models and Extract Latent Spaces
# Initialize SCVI for model loading
scvi.settings.seed = 55 # For reproducibility
adata = adata.copy()
setup_scvi_anndata(adata)

# Load Centralized Model
print(f"Loading centralized model from: {CENTRALIZED_MODEL_PATH}")
try:
    model_centralized = scvi.model.SCVI.load(CENTRALIZED_MODEL_PATH, adata=adata)
    latent_centralized = model_centralized.get_latent_representation()
    print("Centralized model loaded and latent space extracted.")
except Exception as e:
    print(f"Error loading centralized model or extracting latent space: {e}")
    model_centralized = None

# Load Federated Model
print(f"Loading federated model from: {FEDERATED_MODEL_PATH}")
try:
    model_federated = get_scvi_model(adata, hvg_list)
    # Load saved model weights (handle DataParallel wrapping if present)
    state_dict = torch.load(FEDERATED_MODEL_PATH)
    if hasattr(model_federated, "module"):
        model_federated.module.load_state_dict(state_dict, strict=True)
    else:
        model_federated.load_state_dict(state_dict, strict=True)
    model_federated.is_trained = True

    latent_federated = model_federated.get_latent_representation()
    print("Federated model loaded and latent space extracted.")
except Exception as e:
    print(f"Error loading federated model or extracting latent space: {e}")
    model_federated = None

# 4. Prepare AnnData Objects for Latent Space UMAPs
adata_centralized_umap = adata.copy()
adata_federated_umap = adata.copy()

if model_centralized:
    # Store latent in X_pca so scanpy's umap can find it
    adata_centralized_umap.obsm['X_pca'] = latent_centralized
    # Compute neighbors on this X_pca representation
    sc.pp.neighbors(adata_centralized_umap, use_rep='X_pca')
    # Run UMAP without the 'obsm' argument, it will now use X_pca by default
    sc.tl.umap(adata_centralized_umap)
else:
    adata_centralized_umap = None # Mark as not available for plotting

if model_federated:
    adata_federated_umap.obsm['X_pca'] = latent_federated
    sc.pp.neighbors(adata_federated_umap, use_rep='X_pca')
    sc.tl.umap(adata_federated_umap)
else:
    adata_federated_umap = None # Mark as not available for plotting


# 5. Plotting
print("Generating UMAP plots...")

# Define the states and their corresponding AnnData objects for column-wise plotting
plot_states_ordered = [
    ('Raw', adata_raw_umap),
    ('Centralized SCVI', adata_centralized_umap),
    ('Federated SCVI', adata_federated_umap)
]

# Filter out any None entries if models failed to load
plot_states_ordered = [(title, ad) for title, ad in plot_states_ordered if ad is not None]

# Define the keys for rows (Batch and Cell Type)
plot_keys_for_rows = {
    'Batch': BATCH_KEY,
    'Cell Type': CELL_TYPE_KEY
}

num_rows = len(plot_keys_for_rows) # Will be 2 (Batch, Cell Type)
num_cols = len(plot_states_ordered) # Will be 3 (Raw, Centralized, Federated, if all loaded)

fig, axes = plt.subplots(
    nrows=num_rows,
    ncols=num_cols,
    figsize=(5 * num_cols, 6 * num_rows), # Adjust figure size dynamically
    squeeze=False # Ensure axes is always a 2D array, even for 1xN or Nx1 plots
)

# Loop through rows (Batch, Cell Type)
for row_idx, (row_label, color_key_to_use) in enumerate(plot_keys_for_rows.items()):
    # Loop through columns (Raw, Centralized, Federated)
    for col_idx, (col_title, current_adata) in enumerate(plot_states_ordered):
        ax = axes[row_idx, col_idx]

        # Determine the actual key to plot for the current row (e.g., 'tech' or 'batch' for batch row)
        effective_color_key = color_key_to_use
        if row_label == 'Batch': # Only apply batch key logic for the 'Batch' row
            effective_color_key = BATCH_KEY if BATCH_KEY in current_adata.obs else ("batch" if "batch" in current_adata.obs else None)
        # For 'Cell Type' row, effective_color_key will simply be CELL_TYPE_KEY

        if effective_color_key and effective_color_key in current_adata.obs:
            # Set legend_loc for the first column of each row (leftmost plots)
            legend_loc = 'right margin' if col_idx == 0 else 'none'
            
            # Use a consistent title for each subplot
            plot_title = f'{col_title}'
            if row_idx == 0: # Add "Batch" or "Cell Type" to the column title only for the first row
                plot_title += ' (Batch)'
            else:
                plot_title += ' (Cell Type)'

            sc.pl.umap(current_adata, color=effective_color_key, title=plot_title,
                       ax=ax, show=False, legend_loc=legend_loc,
                       frameon=False, s=10)
        else:
            ax.set_title(f'{col_title} ({row_label}) - No Data Key') # More generic message
            ax.axis('off')

output_folder = os.path.join("umap", "comparison")
os.makedirs(output_folder, exist_ok=True)

# Adjust overall layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
plt.suptitle('UMAP Visualization of Pancreas Data: Raw vs. SCVI Centralised vs. Federated Model', fontsize=18, y=0.98)
plt.savefig(os.path.join(output_folder, 'raw_vs_cen_vs_fed.png'))
plt.close()
print("Plots generated and saved to:", os.path.abspath(output_folder))