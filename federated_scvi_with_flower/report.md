### Federated `scvi-tools` with Flower

This report outlines the steps to use `scvi-tools` models in a federated manner. The core idea is to perform data loading and local model training on distributed clients, while a central server aggregates model updates.

---

### 1. Client-Side Operations

Each client is responsible for loading its local data, performing preprocessing, and training the global model on its data.

#### 1.1. Data Loading and Preprocessing

Clients start by loading their single-cell data into an `AnnData` object. `scvi-tools` provides several utility functions for this, documented in the Data loading API docs.

For general preprocessing, `scvi-tools` integrates with [Scanpy]. A typical first step is to preserve the raw counts before normalization.

```python
import scanpy as sc
import scvi

# Client loads its local data
adata = scvi.data.read_h5ad("local_data.h5ad")

# Basic preprocessing using Scanpy
# It's crucial to keep the raw counts for scvi-tools models
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
)
```

#### 1.2. Model Setup

Before training, the model must be set up with the client's `AnnData` object. This step registers the data schema with the model's `AnnDataManager`. This configuration must be consistent across all clients in the federation.

```python
# ... existing code ...
# This setup must be consistent across all clients (e.g., same keys)
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="tech", 
    labels_key="celltype" 
)
```

#### 1.3. Local Training (Flower Client)

Within the Flower framework, a client defines methods to get, set, and fit the model parameters.

*   **`set_parameters`**: Receives global model weights from the server and updates the local model.
*   **`fit`**: Trains the local model for a specified number of epochs using its local data.
*   **`get_parameters`**: Returns the updated local model weights to the server.

Here is a conceptual implementation of a Flower `NumPyClient`:

```python
import flwr as fl
import scvi
import torch

class ScviClient(fl.client.NumPyClient):
    def __init__(self, adata):
        self.adata = adata
        # Initialize the model but don't train yet
        self.model = scvi.model.SCVI(self.adata)

    def get_parameters(self, config):
        # Return local model weights
        return [val.cpu().numpy() for val in self.model.module.state_dict().values()]

    def set_parameters(self, parameters):
        # Update local model with parameters from the server
        params_dict = zip(self.model.module.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.module.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # Set model parameters, train locally, and return updated parameters
        self.set_parameters(parameters)
        self.model.train(max_epochs=1, batch_size=128) # Train for one epoch
        return self.get_parameters(config={}), self.adata.n_obs, {}

# Start the Flower client
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ScviClient(adata))
```

---

### 2. Server-Side Operations

The server orchestrates the federated learning process. Its main roles are to initialize the global model and aggregate the parameters received from clients.

#### 2.1. Global Preprocessing and Configuration

In a real-world federated scenario, the server does not have access to the clients' data. Therefore, "global" preprocessing is not directly possible. Instead, the federation must agree on a common data schema *a priori*.

This includes:
*   **A shared feature space**: All clients must use the same set of genes. This can be achieved by agreeing on a common list of highly variable genes beforehand.
*   **Consistent `setup_anndata` arguments**: All clients must use the same keys for `layer`, `batch_key`, `labels_key`, etc.
*   **Consistent categorical mappings**: The integer encodings for categorical data like batches or cell types must be globally consistent. This may require a pre-training coordination step where clients report their unique categories to the server, which then creates and distributes a global mapping.

#### 2.2. Global Model Initialization and Aggregation (Flower Server)

The server initializes the `scvi-tools` model. The model architecture (e.g., `n_input`, `n_batch`) must match the clients' data schema. The server then runs the federated averaging strategy.

```python
import flwr as fl
import scvi

# (Optional) Define a function to initialize model parameters on the server
def get_initial_parameters(n_input, n_batch):
    # Create a dummy model instance to get initial weights
    # This requires knowing the dimensions beforehand
    model = scvi.model.SCVI.from_rna_unsupervised(
        n_input=n_input,
        n_batch=n_batch
    )
    return [val.cpu().numpy() for val in model.module.state_dict().values()]

# Define the federated averaging strategy
strategy = fl.server.strategy.FedAvg(
    # initial_parameters=fl.common.ndarrays_to_parameters(
    #     get_initial_parameters(n_input=2000, n_batch=5)
    # ),
    # min_fit_clients=2,
    # min_available_clients=2,
)

# Start the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

---
## 3. SUMMARY
#### 3.1 What Can Be Done Individually by Each Client

* Data Loading: Each client is responsible for loading its own raw dataset. The actual data (the single-cell measurements) never leaves the client's machine. In client.py, this is handled within the load_data function, which would be adapted to load a client's specific file.

* Local Model Training: The core training process happens on the client. The fit method in the ScviClient class calls self.model.train(), which uses the client's local data to update the model weights. The computational work is entirely decentralized.

#### 3.2 What Must Be Set Up and Agreed Upon Together

* Preprocessing Pipeline: While each client preprocesses its data locally, the steps and their parameters must be the same for everyone.

    - Highly Variable Genes: In client.py, we select 2000 highly variable genes (n_top_genes=2000). This number must be the same for all clients to ensure the final model has a consistent input feature space.
    - Normalization: The method and target sum (sc.pp.normalize_total(adata, target_sum=1e4)) should be consistent.
* Data Schema with setup_anndata: This is the most critical part. The arguments passed to scvi.model.SCVI.setup_anndata define the model's structure and must be identical across all clients.

    - layer="counts": All clients must have the raw counts stored in this layer.
    - batch_key="cell_source": All clients must use the same column name in their adata.obs to denote the data's origin or batch.
    - labels_key="cell_type": If you are training a supervised or semi-supervised model, the key for cell type labels must be consistent.
* Categorical Mappings: The integer codes for categorical data (like cell types or batches) must be globally consistent. For example, if "T-cell" is encoded as 0 for one client, it must be 0 for all clients. This often requires a pre-coordination step where the server gathers all unique categories and distributes a global mapping.

* Model Architecture: The server initializes the global model with a specific architecture. The parameters defining this architecture (n_input, n_batch, n_labels in server.py) must match the data properties of all clients after their preprocessing.