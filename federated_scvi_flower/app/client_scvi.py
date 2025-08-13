import flwr as fl
import numpy as np
import anndata
import scanpy as sc
from flwr.client import ClientApp
from flwr.common import Context
from app.utils.data_utils_scvi import load_batch_list, load_partitioned_anndata, ensure_hvg_genes, load_hvg_list
from app.utils.model_utils_scvi import get_scvi_model, setup_scvi_anndata, get_weights, set_weights, train_scvi, evaluate_scvi
import os
import gc
import random
import torch

SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

class ScviClient(fl.client.NumPyClient):
    def __init__(self, adata, adata_test, client_id, hvg_list, all_batches):
        self.adata = adata
        self.adata_test = adata_test
        self.client_id = client_id
        
        # Setup anndata with the unified batch categories for train and test
        setup_scvi_anndata(self.adata, all_batches=all_batches)
        setup_scvi_anndata(self.adata_test, all_batches=all_batches)
        
        self.model = get_scvi_model(self.adata, hvg_list, partition_id=self.client_id)
        self.train_loss = None
        self.test_loss = None

    def get_parameters(self, config):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("epochs", 5))
        train_loss = train_scvi(self.model, self.adata, max_epochs=epochs)
        self.train_loss = train_loss
        gc.collect()
        return get_weights(self.model), self.adata.n_obs, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluate on held-out test set
        test_loss = evaluate_scvi(self.model, self.adata_test)
        self.test_loss = test_loss
        gc.collect()
        return float(test_loss), self.adata_test.n_obs, {"test_loss": test_loss}

def client_fn(context: Context):
    hvg_list_path = context.run_config.get("hvg_list_path", "0_data/hvg_list.json")
    batch_list_path = context.run_config.get("batch_list_path", "0_data/batch_list.json")

    partition_id = context.node_config["partition-id"]
    num_clients = context.run_config.get("num_clients", 3)
    
    # Load partitioned training data using your utility function
    adata_train = load_partitioned_anndata(partition_id, num_clients)
    
    # Load full test data (not partitioned)
    adata_test = anndata.read_h5ad(os.path.join("0_data", "pancreas_test.h5ad"))

    # Load HVG list and ensure genes in test set
    hvg_list = load_hvg_list(hvg_list_path)
    adata_train = ensure_hvg_genes(adata_train, hvg_list, partition_id=partition_id)
    adata_test = ensure_hvg_genes(adata_test, hvg_list)

    # Assign batch info from 'tech' for train and test
    adata_train.obs['batch'] = adata_train.obs['tech']
    adata_test.obs['batch'] = adata_test.obs['tech']
    # print(f"Client {partition_id} - Training data shape: {adata_train.shape}, Test data shape: {adata_test.shape}")
    # print(f"Client {partition_id} - Batches in train: {adata_train.obs['batch'].unique()}, test: {adata_test.obs['batch'].unique()}")
    # Load all batch categories (for consistent batch handling)
    all_batches = load_batch_list(batch_list_path)

    # Initialize and return client
    client = ScviClient(adata_train, adata_test, client_id=partition_id, hvg_list=hvg_list, all_batches=all_batches)
    return client.to_client()


app = ClientApp(client_fn=client_fn) 