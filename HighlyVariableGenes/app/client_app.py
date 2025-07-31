"""HighlyVariableGenes: A Flower for highly variable genes detection."""

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
import json
from app.task import load_partitioned_anndata
import numpy as np
import scanpy as sc
import os
import random

SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

class FlowerClient(NumPyClient):

    def __init__(self, timeout, data, partition_id: int):
        self.timeout = timeout
        self.data = data
        self.partition_id = partition_id
        
    # PRENDERE VARIANZA DEL GENE PER OGNI CLIENT POI AGGREGARE
    def fit(self, parameters, config):
        # Compute HVGs but do NOT subset, keep dispersions_norm
        sc.pp.highly_variable_genes(self.data, n_top_genes=2000, subset=False)
        print(f"Client {self.partition_id}")

        # Use dispersions_norm as continuous variability score
        hvg_scores_local_unsorted = self.data.var["dispersions_norm"].values.astype(np.float32)

        gene_to_score = dict(zip(self.data.var_names.tolist(), hvg_scores_local_unsorted))

        gene_list_str = config.get("global_gene_list", "[]")
        global_gene_list = json.loads(gene_list_str)

        aligned_scores = np.zeros(len(global_gene_list), dtype=np.float32)
        for i, gene in enumerate(global_gene_list):
            aligned_scores[i] = gene_to_score.get(gene, 0.0)

        # Save for debugging
        selected_genes_dict = {gene: float(score) for gene, score in zip(global_gene_list, aligned_scores) if score > 0}
        with open(f"HighlyVariableGenes/check_hvg_fl/selected_genes_client{self.partition_id}.json", "w") as f:
            json.dump(selected_genes_dict, f)

        return [aligned_scores], len(self.data), {}

    def get_properties(self, config):
        return {"gene_names": self.data.var_names.tolist()}

def client_fn(context: Context):
    timeout = context.run_config["timeout"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_partitioned_anndata(partition_id, num_partitions)
    return FlowerClient(timeout, data, partition_id).to_client()

app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
