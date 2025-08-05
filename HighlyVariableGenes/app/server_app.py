"""HighlyVariableGenes: A Flower for highly variable genes detection."""

from logging import INFO
import numpy as np
import json
import os
import random
from typing import List, Tuple
from flwr.common import Context, log, parameters_to_ndarrays
from flwr.common import ndarrays_to_parameters, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import anndata as ad
from app.task import get_initial_gene_list, get_gene_index_name_dict
import atexit

# --- CONFIGURATION ---
SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Placeholder for the strategy object, which will be populated by server_fn
strategy = None

def aggregate_variances(server_round, results: List[Tuple], failures):
    """
    Custom aggregation function to compute global variance from local means and
    mean of squares.
    """
    if not results:
        return None, {}

    # Extract local means, local means of squares, and number of examples from each client
    local_means_list = []
    local_mean_of_squares_list = []
    num_examples_list = []

    for _, res in results:
        local_means, local_mean_of_squares = parameters_to_ndarrays(res.parameters)
        
        local_means_list.append(local_means)
        local_mean_of_squares_list.append(local_mean_of_squares)
        num_examples_list.append(res.num_examples)

    total_examples = sum(num_examples_list)
    if total_examples == 0:
        return None, {}

    num_genes = len(local_means_list[0])
    global_mean = np.zeros(num_genes, dtype=np.float32)
    global_mean_of_squares = np.zeros(num_genes, dtype=np.float32)

    for i, (local_means, local_mean_of_squares) in enumerate(zip(local_means_list, local_mean_of_squares_list)):
        weight = num_examples_list[i] / total_examples
        global_mean += local_means * weight
        global_mean_of_squares += local_mean_of_squares * weight
    
    global_variance = global_mean_of_squares - (global_mean ** 2)

    return ndarrays_to_parameters([global_variance]), {}

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round: int, results: List[Tuple], failures: List):
        aggregated_parameters, metrics = aggregate_variances(server_round, results, failures)
        
        self.final_parameters = aggregated_parameters
        
        return aggregated_parameters, metrics

def server_fn(context: Context) -> ServerAppComponents:
    """
    The main server function to execute the federated learning workflow.
    """
    global strategy
    num_clients = int(context.run_config.get("num_clients", 2))
    num_rounds = int(context.run_config.get("num_rounds", 1))
    
    # --- SERVER SETUP ---
    global_gene_list = get_initial_gene_list()
    log(INFO, f"Server global gene list first 10 genes: {global_gene_list[:10]}")
    log(INFO, f"Global gene list length: {len(global_gene_list)}")
    
    def fit_config_fn(rnd: int):
        """Returns a configuration dictionary for the client fit round."""
        return {"global_gene_list": json.dumps(global_gene_list)}

    # Initialize with a single dummy array of zeros for the initial parameters.
    # The dimensions should match the expected output.
    initial_params_ndarrays = [np.zeros(len(global_gene_list), dtype=np.float32)]
    initial_params = ndarrays_to_parameters(initial_params_ndarrays)

    # --- STRATEGY SETUP ---
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        accept_failures=False,
        fraction_evaluate=0.0,
        initial_parameters=initial_params,
        on_fit_config_fn=fit_config_fn,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
    )
    
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

app = ServerApp(server_fn=server_fn)

def _save_on_exit():
    """Saves the final aggregated HVG list and indices to JSON files."""
    global strategy
    
    if strategy and strategy.final_parameters:
        try:
            aggregated_variances = parameters_to_ndarrays(strategy.final_parameters)[0]
            
            # Select top 2000 genes by aggregated variance
            top_2000_indices = np.argsort(aggregated_variances)[-2000:][::-1]
            sorted_indices = sorted([int(i) for i in top_2000_indices])
            
            os.makedirs("HighlyVariableGenes", exist_ok=True)
            
            with open("HighlyVariableGenes/hvg_indices.json", "w") as f:
                json.dump(sorted_indices, f)
            
            # Assuming get_gene_index_name_dict can read the necessary data
            idx_to_name = get_gene_index_name_dict("data/pancreas_train.h5ad")
            hvg_list = sorted([idx_to_name[i] for i in sorted_indices])

            with open("HighlyVariableGenes/hvg_list.json", "w") as f:
                json.dump(hvg_list, f)
            with open("data/hvg_list_from_fl.json", "w") as f:
                json.dump(hvg_list, f)
            log(INFO, "Successfully saved final HVG lists to 'HighlyVariableGenes/hvg_indices.json' and 'HighlyVariableGenes/hvg_list.json'")

        except Exception as e:
            log(INFO, f"An error occurred while saving the final results: {e}")
    else:
        log(INFO, "No final parameters found in the strategy. Skipping save operation.")

atexit.register(_save_on_exit)