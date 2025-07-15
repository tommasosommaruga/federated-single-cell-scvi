import logging
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from federated_scvi_flower.model_utils_scvi import get_scvi_model, get_weights, set_weights, plot_latent_umap
import anndata
import scanpy as sc
import scvi
import os
import json
import pandas as pd
import scipy.sparse
from federated_scvi_flower.data_utils_scvi import ensure_hvg_genes

# Utility to load HVG list
def load_hvg_list(hvg_list_path):
    with open(hvg_list_path) as f:
        return json.load(f)

# Set up critical file logger
critical_logger = logging.getLogger("server_critical")
critical_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("federated_scvi_flower/server_critical.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
if not critical_logger.hasHandlers():
    critical_logger.addHandler(file_handler)

def server_fn(context: Context) -> ServerAppComponents:
    num_clients = int(context.run_config.get("num_clients", 2))
    num_rounds = int(context.run_config.get("num_rounds", 3))
    epochs = int(context.run_config.get("local-epochs", 1))
    print(f"[server_scvi] num_clients: {num_clients}, context.run_config.: {context.run_config}")
    global model, adata_ref

    adata_path = os.path.join("data", "pancreas.h5ad")
    adata = anndata.read_h5ad(adata_path)
    # Convert counts layer to int32 if present
    if hasattr(adata, 'layers') and 'counts' in adata.layers:
        adata.layers['counts'] = adata.layers['counts'].astype('int32')
    hvg_list_path = context.run_config.get("hvg_list_path", "data/hvg_list.json")
    hvg_list = load_hvg_list(hvg_list_path)
    ref_mask = (~adata.obs["tech"].isin(["smartseq2", "celseq2"])).values
    adata_ref = adata[ref_mask].copy()
    adata_ref = ensure_hvg_genes(adata_ref, hvg_list)
    # Check for duplicates
    adata_var_names_set = set(adata_ref.var_names)
    hvg_list_set = set(hvg_list)
    if len(adata_ref.var_names) != len(set(adata_ref.var_names)):
        critical_logger.info("DUPLICATES FOUND in adata_ref.var_names!")
    if len(hvg_list) != len(set(hvg_list)):
        critical_logger.info("DUPLICATES FOUND in hvg_list!")
    # Check for missing genes
    missing_in_adata = [g for g in hvg_list if g not in adata_var_names_set]
    missing_in_hvg = [g for g in adata_ref.var_names if g not in hvg_list_set]
    critical_logger.info(f"Genes in HVG list but not in adata_ref: {missing_in_adata}")
    critical_logger.info(f"Genes in adata_ref but not in HVG list: {missing_in_hvg}")
    # Check order
    if list(adata_ref.var_names) != list(hvg_list):
        critical_logger.info("ORDER MISMATCH between adata_ref.var_names and hvg_list!")
        
    model = get_scvi_model(adata_ref, hvg_list)
    initial_parameters = ndarrays_to_parameters(get_weights(model))
    strategy = FedAvg(
        initial_parameters=initial_parameters,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"epochs": epochs}
    )
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

app = ServerApp(server_fn=server_fn)

# Save the final model and AnnData after training
def save_final_model_and_adata(model, adata, path_prefix="federated_scvi_flower/final_server_model"):
    import torch
    model_path = f"{path_prefix}.pt"
    adata_path = f"{path_prefix}_adata.h5ad"
    torch.save(model.module.state_dict(), model_path)
    adata.write(adata_path)
    print(f"Saved final model to {model_path} and AnnData to {adata_path}")

# Patch the ServerApp to call this after training
import atexit
model = None
adata_ref = None
def _save_on_exit():
    try:
        global model, adata_ref
        if model is not None and adata_ref is not None:
            save_final_model_and_adata(model, adata_ref)
        else:
            print("[WARN] Model or AnnData not defined at exit, not saving.")
    except Exception as e:
        print(f"[WARN] Could not save final model: {e}")
atexit.register(_save_on_exit) 