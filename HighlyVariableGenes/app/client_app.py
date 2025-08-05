"""HighlyVariableGenes: A Flower for highly variable genes detection."""

import os
import json
import random
import numpy as np
import scanpy as sc

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from app.task import load_partitioned_anndata

# --- CONFIGURATION ---
SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class FlowerClient(NumPyClient):
    """Flower client for calculating local gene statistics."""

    def __init__(self, timeout: float, data: sc.AnnData, partition_id: int):
        self.timeout = timeout
        self.data = data
        self.partition_id = partition_id
    
    def fit(self, parameters, config):
        """
        Computes and returns local statistics (mean and mean of squares)
        for each gene. These are aligned to a global gene list.
        """
        # N_i: number of samples (cells) for this client
        num_examples = self.data.n_obs

        # Get the global gene list from the server configuration
        gene_list_str = config.get("global_gene_list", "[]")
        global_gene_list = json.loads(gene_list_str)
        
        # Initialize arrays for local means and local means of squares
        aligned_mean_scores = np.zeros(len(global_gene_list), dtype=np.float32)
        aligned_mean_of_squares_scores = np.zeros(len(global_gene_list), dtype=np.float32)

        # Get the expression matrix, handling potential sparse format
        X_matrix = self.data.X.A if hasattr(self.data.X, 'A') else self.data.X
        gene_to_index = {gene: i for i, gene in enumerate(self.data.var_names)}

        # Iterate through the global gene list to compute and align statistics
        for i, global_gene in enumerate(global_gene_list):
            if global_gene in gene_to_index:
                local_index = gene_to_index[global_gene]
                gene_expression = X_matrix[:, local_index]
                
                # Calculate local mean and local mean of squares for the gene
                local_mean_gene = np.mean(gene_expression)
                local_mean_of_squares_gene = np.mean(gene_expression ** 2)
                
                aligned_mean_scores[i] = local_mean_gene
                aligned_mean_of_squares_scores[i] = local_mean_of_squares_gene

        # Return the computed statistics as parameters and the number of examples
        return [aligned_mean_scores, aligned_mean_of_squares_scores], num_examples, {}

    def get_properties(self, config):
        """Returns the list of gene names available on this client."""
        return {"gene_names": self.data.var_names.tolist()}

def client_fn(context: Context):
    """Loads data and creates a FlowerClient instance."""
    timeout = context.run_config["timeout"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_partitioned_anndata(partition_id, num_partitions)
    return FlowerClient(timeout, data, partition_id).to_client()

app = ClientApp(
    client_fn=client_fn,
)