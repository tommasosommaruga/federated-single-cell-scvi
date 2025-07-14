import flwr as fl
import scvi
import torch
import pandas as pd
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

# --- Server State to hold the dynamic configuration ---
class DynamicServerState:
    def __init__(self):
        self.hvg_list: Optional[List[str]] = None
        self.cat_mappings: Optional[Dict[str, Dict[str, int]]] = None
        self.model_parameters: Optional[Parameters] = None

server_state = DynamicServerState()

# --- Custom Strategy for Coordination ---
class CoordinatedStrategy(FedAvg):
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        config = {}
        if server_round == 1:
            # Round 1 is for discovery
            config["round"] = "discovery"
        else:
            # Subsequent rounds are for training
            config["round"] = "train"
            config["hvg_list"] = server_state.hvg_list
            config["cat_mappings"] = server_state.cat_mappings

        # Use the standard FedAvg mechanism to create FitIns
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Get all available clients
        clients = client_manager.sample(
            num_clients=self.min_available_clients, min_num_clients=self.min_available_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results from clients."""
        if server_round == 1:
            # --- ROUND 1: AGGREGATE METADATA ---
            print("Server: Round 1 - Aggregating metadata from clients...")
            
            all_cats = {"batch": set(), "cell_type": set()}
            all_gene_stats = []

            for _, fit_res in results:
                client_metadata = fit_res.metrics
                all_cats["batch"].update(client_metadata["batch_cats"])
                all_cats["cell_type"].update(client_metadata["cell_type_cats"])
                # Deserialize gene stats from JSON string
                gene_stats_df = pd.read_json(client_metadata["gene_stats_json"])
                all_gene_stats.append(gene_stats_df.set_index("gene"))

            # --- CREATE GLOBAL CONFIGURATION ---
            # 1. Create global categorical mappings
            server_state.cat_mappings = {
                key: {label: i for i, label in enumerate(sorted(list(labels)))}
                for key, labels in all_cats.items()
            }
            print(f"Server: Created global mappings: {server_state.cat_mappings}")

            # 2. Create global HVG list
            combined_stats = pd.concat(all_gene_stats).groupby(level=0).mean()
            combined_stats.sort_values("dispersions_norm", ascending=False, inplace=True)
            server_state.hvg_list = combined_stats.head(2000).index.tolist()
            print(f"Server: Created HVG list of length {len(server_state.hvg_list)}")

            # 3. Initialize the global model with the new dimensions
            n_input = len(server_state.hvg_list)
            n_batch = len(server_state.cat_mappings["batch"])
            n_labels = len(server_state.cat_mappings["cell_type"])
            
            model = scvi.model.SCVI(
                n_input=n_input, n_batch=n_batch, n_labels=n_labels,
                use_layer_norm="both", use_batch_norm="none",
                encode_covariates=True, dropout_rate=0.2, n_layers=2,
            )
            server_state.model_parameters = ndarrays_to_parameters(
                [val.cpu().numpy() for val in model.module.state_dict().values()]
            )
            print("Server: Global model initialized.")
            
            # Return None for parameters, FedAvg will use the one from the previous round
            return None, {}

        else:
            # --- SUBSEQUENT ROUNDS: AGGREGATE WEIGHTS ---
            print(f"Server: Round {server_round} - Aggregating model weights...")
            aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_params:
                server_state.model_parameters = aggregated_params
            
            return server_state.model_parameters, aggregated_metrics

# --- Define the Flower App using a server_fn ---
def server_fn(context: Context) -> ServerAppComponents:
    """Define the server components."""
    num_clients = int(context.run_config.get("num_clients", 2))
    num_rounds = int(context.run_config.get("num_rounds", 4))

    # Define the strategy
    strategy = CoordinatedStrategy(
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        # Pass empty initial parameters, they will be created after discovery
        initial_parameters=ndarrays_to_parameters([]),
    )

    # Return the server components
    return ServerAppComponents(
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

# Define the ServerApp
app = ServerApp(
    server_fn=server_fn,
)
