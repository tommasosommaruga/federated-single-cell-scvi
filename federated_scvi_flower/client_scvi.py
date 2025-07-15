import os
import logging
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
import sys
import time
import anndata
import scanpy as sc
import scvi
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_scvi_flower.data_utils_scvi import load_partitioned_anndata, ensure_hvg_genes
import json
from federated_scvi_flower.model_utils_scvi import get_scvi_model, get_weights, set_weights, train_scvi, evaluate_scvi

# Utility to load HVG list
def load_hvg_list(hvg_list_path):
    with open(hvg_list_path) as f:
        return json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScviClient(fl.client.NumPyClient):
    def __init__(self, adata, client_id, hvg_list):
        self.adata = adata
        self.client_id = client_id
        # Remove or comment out repetitive debug prints
        # print(f"[DEBUG][partition {self.client_id}] HVG list length: {len(hvg_list)}, first 20: {hvg_list[:20]}")
        # print(f"[DEBUG][partition {self.client_id}] AnnData var_names length: {len(adata.var_names)}, first 20: {list(adata.var_names[:20])}")
        # print(f"[DEBUG][partition {self.client_id}] Genes in HVG list but not in AnnData: {set(hvg_list) - set(adata.var_names)}")
        # print(f"[DEBUG][partition {self.client_id}] Genes in AnnData but not in HVG list: {set(adata.var_names) - set(hvg_list)}")
        self.model = get_scvi_model(self.adata, hvg_list, partition_id=self.client_id)
        self.train_loss = None
        self.test_loss = None
        logger.info(f"Initialized SCVI client {client_id} with {adata.n_obs} cells.")

    def get_parameters(self, config):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("epochs", 5))
        logger.info(f"[federated_client][partition {self.client_id}] About to train for {epochs} epochs.")
        train_loss = train_scvi(self.model, self.adata, max_epochs=epochs)
        self.train_loss = train_loss
        logger.info(f"[federated_client][partition {self.client_id}] Training complete. Final train loss: {train_loss}")
        return get_weights(self.model), self.adata.n_obs, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss = evaluate_scvi(self.model, self.adata)
        self.test_loss = test_loss
        logger.info(f"[federated_client][partition {self.client_id}] Evaluation complete. Test loss: {test_loss}")
        return float(test_loss), self.adata.n_obs, {"test_loss": test_loss}

def client_fn(context: Context):
    # For this experiment, all clients use the same full AnnData (no partitioning)
    hvg_list_path = context.run_config.get("hvg_list_path", "data/hvg_list.json")
    adata = anndata.read_h5ad("data/pancreas.h5ad")
    # Convert counts layer to int32 if present
    if hasattr(adata, 'layers') and 'counts' in adata.layers:
        adata.layers['counts'] = adata.layers['counts'].astype('int32')
    if "tech" in adata.obs:
        ref_mask = (~adata.obs["tech"].isin(["smartseq2", "celseq2"])).values
        adata = adata[ref_mask].copy()
    hvg_list = load_hvg_list(hvg_list_path)
    adata = ensure_hvg_genes(adata, hvg_list)
    # Use a dynamic partition_id/client_id from context.node_config
    partition_id = context.node_config["partition-id"]
    client = ScviClient(adata, client_id=partition_id, hvg_list=hvg_list)
    return client.to_client()

app = ClientApp(client_fn=client_fn) 