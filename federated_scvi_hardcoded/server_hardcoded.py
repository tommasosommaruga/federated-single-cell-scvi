import flwr as fl
import scvi
import torch
import scanpy as sc
from flwr.common import Context, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

# 1. PRE-AGREED GLOBAL CONFIGURATION (HARCODED)
# This information must be known before the federation starts.
# In a real scenario, this would be derived from a pilot study or public data.

# To make this realistic, we derive the HVG list and categories once from the full dataset.
# This simulates the outcome of a pre-federation analysis.
adata_full = scvi.data.pancreas()
sc.pp.highly_variable_genes(
    adata_full, n_top_genes=2000, flavor="seurat_v3", batch_key="batch", subset=True
)
HVG_LIST = adata_full.var_names.tolist()

# A pre-defined global mapping for all categorical variables.
GLOBAL_CAT_MAPPINGS = {
    "batch": {
        label: i for i, label in enumerate(sorted(adata_full.obs.batch.unique()))
    },
    "cell_type": {
        label: i for i, label in enumerate(sorted(adata_full.obs.cell_type.unique()))
    },
}

# Pre-defined model architecture dimensions
N_INPUT = len(HVG_LIST)
N_BATCH = len(GLOBAL_CAT_MAPPINGS["batch"])
N_LABELS = len(GLOBAL_CAT_MAPPINGS["cell_type"])


# 2. SERVER-SIDE MODEL INITIALIZATION
def get_initial_parameters() -> Parameters:
    """Create a dummy model instance to get the initial weights."""
    model = scvi.model.SCVI(
        n_input=N_INPUT,
        n_batch=N_BATCH,
        n_labels=N_LABELS,
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
    )
    model_weights = [val.cpu().numpy() for val in model.module.state_dict().values()]
    return ndarrays_to_parameters(model_weights)


# 3. FLOWER STRATEGY DEFINITION
def fit_config(server_round: int):
    """Pass the hardcoded configuration to the clients."""
    return {
        "hvg_list": HVG_LIST,
        "cat_mappings": GLOBAL_CAT_MAPPINGS,
    }


# 4. DEFINE THE FLOWER APP using a server_fn
def server_fn(context: Context) -> ServerAppComponents:
    """Define the server components."""
    # Get configuration from pyproject.toml
    num_rounds = int(context.run_config.get("num_rounds", 3))
    num_clients = int(context.run_config.get("num_clients", 2))

    # Define the strategy
    strategy = FedAvg(
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        initial_parameters=get_initial_parameters(),
    )

    # Return the server components
    return ServerAppComponents(
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

app = ServerApp(
    server_fn=server_fn,
)
