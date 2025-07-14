# Federated Learning Coordination Strategies for scvi-tools

This document outlines strategies for coordinating the setup and preprocessing steps in a federated learning environment using `scvi-tools` and Flower. These steps are critical to ensure that the global model is trained on consistent and compatible data from all clients.

### 1. Preprocessing Pipeline Coordination

A consistent preprocessing pipeline is essential for the federated model to learn meaningful biological patterns.

#### Highly Variable Genes (HVGs)

A consistent feature space (i.e., the same set of genes) is crucial for model compatibility. Here are two ways to achieve this:

1.  **Pre-defined Gene List (Centralized Curation):**
    *   **How it works:** A list of HVGs is created beforehand, either from a relevant public dataset or by a trusted curator with access to a representative sample. This static list is then hardcoded or distributed to all clients, who use it to subset their data.
    *   **Pros:** Simple to implement and requires no extra communication rounds.
    *   **Cons:** The gene list might not be optimal for every client's specific data distribution, potentially excluding important genes or including non-informative ones.

2.  **Federated HVG Selection (Iterative Aggregation):**
    *   **How it works:** This involves a multi-step coordination protocol before the main training begins.
        1.  **Local Stats:** Each client calculates gene dispersion statistics locally using a function like `scanpy.pp.highly_variable_genes`.
        2.  **Aggregation:** Clients send these summary statistics (e.g., mean, normalized dispersion per gene) to the server. **No raw data is shared.**
        3.  **Global Selection:** The server aggregates the statistics from all clients to identify a globally representative set of HVGs.
        4.  **Distribution:** The server sends the final list of HVGs back to all clients.
    *   **Pros:** More robust and data-driven, creating a feature space tailored to the actual federated dataset.
    *   **Cons:** Adds complexity and requires extra communication rounds, increasing the overhead of the federated setup.

#### Normalization: Handled Internally by SCVI

External normalization (e.g., `scanpy.pp.normalize_total`) is **not required and should be avoided** for `scvi.model.SCVI`.

The model is designed to work directly on raw count data. It automatically accounts for differences in sequencing depth (library size) as part of its internal architecture. This is a core feature that makes it robust to technical variations between datasets.

**Federation Rule:** The only requirement is for all clients to provide the raw, unnormalized integer counts to the model. This is typically done by saving the raw counts in `adata.layers["counts"]` and pointing to it with `scvi.model.SCVI.setup_anndata(adata, layer="counts")`. This approach simplifies the entire federated process by removing a major coordination step.

<hr>
<br>
<br>

### 2. Data Schema (`setup_anndata`) Coordination

The `scvi.model.SCVI.setup_anndata()` function acts as a **blueprint**. It doesn't process data, but instead tells the model *where* to find specific information within each client's `AnnData` object (e.g., which `.obs` column contains the batch information). For the global model to work, **every client must use the exact same blueprint**. This means the keys passed to this function (`layer`, `batch_key`, `labels_key`, etc.) must be identical across the entire federation.

Here are two ways to enforce this:

1.  **Shared Convention (Hardcoding):**
    *   **How it works:** All participants agree on standard names (e.g., `batch_key="cell_source"`) and hardcode them into their scripts.
    *   **Pros:** Simple to implement.
    *   **Cons:** Rigid and relies on perfect manual compliance, which can be error-prone.

2.  **Server-Sent Configuration (Dynamic & Recommended):**
    *   **How it works:** The server defines the blueprint and sends it to clients as a configuration dictionary. Client code is written to use these keys dynamically.
    *   **Pros:** Programmatically ensures consistency, is flexible, and centrally controlled. This is the most robust approach.
    *   **Cons:** Requires slightly more complex client-side code to handle the dynamic configuration.

<hr>
<br>
<br>

### 3. Categorical Mappings Coordination

To ensure a category like "T-cell" is encoded as `0` for all clients, a global mapping is needed.
```python
scvi.model.SCVI.setup_anndata(pancreas_ref, batch_key="tech", layer="counts")
```

1.  **Pre-defined Mapping:** If all possible categories (cell types, batches) are known beforehand, a global mapping can be created centrally and distributed to all clients. Each client is then responsible for re-mapping its local categories according to this global dictionary. This is suitable for controlled environments where the data schema is stable.
2.  **Federated Mapping Creation:** If categories are not known in advance, a coordination step is required:
    1.  **Local Categories:** Each client inspects its data and sends a list of its unique category labels (e.g., `["T-cell", "B-cell"]`) to the server.
    2.  **Global Aggregation:** The server collects the lists from all clients, finds the unique set of all categories across the federation, creates a single mapping (e.g., `{"T-cell": 0, "B-cell": 1, "Macrophage": 2}`), and sorts it for consistency.
    3.  **Distribution:** The server sends this final global mapping back to all clients to use for encoding their local data. This is a robust solution for heterogeneous data sources.

### 4. Model Architecture Coordination

The server must initialize a model with the correct architecture (e.g., `n_input`, `n_batch`, `n_labels`) that matches the clients' preprocessed data.

1.  **Manual Configuration:** The simplest way is to manually provide these dimensions to the server script. `n_input` would be the number of HVGs (e.g., 2000), and `n_batch` and `n_labels` would be the total number of unique batches and labels across the entire federated dataset. This requires knowing these values in advance and can be error-prone if the dataset changes.
2.  **Dynamic Configuration via a "Discovery" Round:**
    *   **How it works:** The server can initiate a "discovery" round before training.
    *   In this round, clients would preprocess their data and report back the resulting dimensions (`adata.n_vars`, number of unique batches, number of unique labels) to the server.
    *   The server then uses this information to initialize the global model with the correct architecture before starting the first training round. This automates the setup, reduces the chance of manual error, and makes the system more adaptable.
