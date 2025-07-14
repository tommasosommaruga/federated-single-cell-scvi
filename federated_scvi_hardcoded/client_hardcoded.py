import flwr as fl
import scvi
import torch
import scanpy as sc
import pandas as pd
import numpy as np
import sys
from flwr.common import Context

def load_client_data(client_id: int):
    """Loads a partition of the pancreas dataset."""
    full_adata = scvi.data.pancreas()
    unique_batches = sorted(full_adata.obs.batch.unique())
    partition = unique_batches[client_id % len(unique_batches)]
    adata = full_adata[full_adata.obs.batch == partition].copy()
    adata.layers["counts"] = adata.X.copy()
    return adata

class ScviClientHardcoded(fl.client.NumPyClient):
    def __init__(self, adata):
        self.adata = adata
        self.model = None

    def setup_model(self, config):
        """Set up the model based on the hardcoded configuration from the server."""
        hvg_list = config["hvg_list"]
        cat_mappings = config["cat_mappings"]

        # 1. PREPROCESSING
        # Subset to the agreed-upon HVG list
        genes_to_keep = [g for g in hvg_list if g in self.adata.var_names]
        self.adata = self.adata[:, genes_to_keep].copy()

        # Apply the global categorical mapping
        self.adata.obs["_batch_indices"] = self.adata.obs["batch"].map(cat_mappings["batch"]).astype("category")
        self.adata.obs["_labels_indices"] = self.adata.obs["cell_type"].map(cat_mappings["cell_type"]).astype("category")

        # 2. MODEL SETUP
        scvi.model.SCVI.setup_anndata(
            self.adata,
            layer="counts",
            batch_key="_batch_indices",
            labels_key="_labels_indices",
        )
        self.model = scvi.model.SCVI(self.adata, use_layer_norm="both", use_batch_norm="none",
                                     encode_covariates=True, dropout_rate=0.2, n_layers=2)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.module.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.module.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if self.model is None:
            self.setup_model(config)
        
        self.set_parameters(parameters)
        self.model.train(max_epochs=1, batch_size=128)
        return self.get_parameters(config={}), self.adata.n_obs, {}

def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    # The context object contains information about the client, including its ID
    cid = context.node_config["cid"]
    # Load data
    adata = load_client_data(int(cid))
    # Create a client
    return ScviClientHardcoded(adata).to_client()

# Flower ClientApp
app = fl.client.ClientApp(
    client_fn=client_fn,
)
