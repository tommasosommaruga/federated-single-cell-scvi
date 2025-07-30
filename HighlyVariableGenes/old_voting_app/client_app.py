"""HighlyVariableGenes: A Flower for highly variable genes detection."""

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
import json
from app.task import load_partitioned_anndata
import numpy as np
import scanpy as sc

class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, timeout, data, partition_id: int,):
        self.timeout = timeout
        self.data = data
        self.partition_id = partition_id
        
    def fit(self, parameters, config):
        sc.pp.highly_variable_genes(self.data, n_top_genes=2000, subset=False)
        print(f"Client {self.partition_id}")

        # Sort client gene list exactly the same as global gene list (alphabetically)
        hvg_mask_local_unsorted = self.data.var["highly_variable"].values.astype(np.float32)

        # Create mapping from unsorted client genes to their HVG mask
        gene_to_hvg_flag = dict(zip(self.data.var_names.tolist(), hvg_mask_local_unsorted))

        gene_list_str = config.get("global_gene_list", "[]")
        global_gene_list = json.loads(gene_list_str)

        aligned_mask = np.zeros(len(global_gene_list), dtype=np.float32)
        for i, gene in enumerate(global_gene_list):
            if gene in gene_to_hvg_flag:
                aligned_mask[i] = gene_to_hvg_flag[gene]

        # Saved to check the alignment
        selected_genes_dict = {gene: i for i, (gene, flag) in enumerate(zip(global_gene_list, aligned_mask)) if flag == 1.0}
        with open(f"selected_genes_client{self.partition_id}.json", "w") as f: json.dump(selected_genes_dict, f)
        return [aligned_mask], len(self.data), {}
    
    def get_properties(self, config):
        # Send gene list to the server
        return {
            "gene_names": self.data.var_names.tolist()
        }
def client_fn(context: Context):
    # Retrieve timeout (necessary for SecAggPlus)
    timeout = context.run_config["timeout"]

    # Retrieve simulation dataset parameters
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_partitioned_anndata(partition_id, num_partitions)

    return FlowerClient(timeout, data, partition_id).to_client() 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod, # Enable secure aggregation through SecAggPlus
    ],
)
