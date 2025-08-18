# 3 Experiments and Results

This section details the experimental setup, including data characteristics, model training dynamics, and the evaluation of model performance through various metrics and visualizations.

## 3.1 Data Explanation

The primary dataset used for these experiments is the `pancreas.h5ad` AnnData object, containing single-cell RNA sequencing data. Prior to model training, the dataset underwent a specific splitting procedure to create training and testing sets.

The splitting methodology, implemented in `0_data/split_data.py`, designates cells originating from "smartseq2" and "celseq2" technologies as the dedicated test set. The remaining cells, encompassing data from other technologies, constitute the training set. This approach ensures that the model's generalization capabilities are evaluated on technically distinct data, which is crucial for assessing performance in a federated learning context where different clients might contribute data generated with varying technologies.

The raw counts layer of the AnnData object is converted to `int32` for consistency. The data is saved into `pancreas_train.h5ad` and `pancreas_test.h5ad` files.

## 3.2 Loss

The training and testing loss curves provide crucial insights into the learning dynamics and convergence behavior of the different scVI models (centralized, independent, and federated). These curves, typically stored in CSV files within the `loss_logs/` directory (e.g., `loss_curve_centralised.csv`, `loss_curve_client_X.csv`, and `loss_curve_federated_X_clients_Y_rounds_Z_epochs.csv`), illustrate the model's performance over epochs or communication rounds.

Generally, a well-performing model exhibits a decreasing trend in both training and testing loss, indicating successful learning and generalization to unseen data. Comparisons across centralized, independent, and federated models, often visualized in plots generated in `compare_fed_plot/`, `compare_models_plot/`, and `loss_plots/` directories, help in understanding the efficacy of federated learning in achieving comparable or superior performance while preserving data privacy. Specifically, the federated loss curves demonstrate the impact of different configurations of clients, communication rounds, and local epochs on the overall model convergence.

## 3.3 UMAP and Metrics

To visually assess the quality of the learned latent spaces from the scVI models, Uniform Manifold Approximation and Projection (UMAP) plots are generated. These visualizations, typically found in the `umap/centralised/`, `umap/independent_client_X/`, and `umap/federated/` directories, provide a two-dimensional representation of the high-dimensional single-cell data, allowing for qualitative evaluation of cell type separation and batch effect integration.

Beyond visual inspection, quantitative metrics are employed to rigorously evaluate the clustering performance and the removal of technical variations (batch effects) in the latent space. The `umap/latent_space_clustering_metrics.ipynb` notebook details the calculation of these metrics, which notably include:

*   **Adjusted Rand Index (ARI):** Measures the similarity between true and predicted cluster assignments, adjusted for chance. A higher ARI indicates better agreement between the clustering derived from the latent space and the ground truth cell type annotations.
*   **Normalized Mutual Information (NMI):** Quantifies the mutual dependence between the true and predicted clusterings, normalized to values between 0 and 1. Similar to ARI, a higher NMI signifies more accurate clustering.
*   **Silhouette Score:** Evaluates the compactness and separation of clusters. A higher silhouette score indicates that objects are well-matched to their own cluster and poorly matched to neighboring clusters.

## 3.4 Comparison with 1 Batch Per Client

In the context of single-cell data analysis, technical variations, often referred to as "batch effects," are a common challenge. The experimental setup implicitly addresses a scenario akin to "one batch per client" by distributing data across clients based on underlying technical variations, as observed in the `0_data/split_data.py` where specific technologies (`smartseq2`, `celseq2`) are separated.

This section focuses on evaluating how the federated scVI model performs when faced with data from distinct technical batches across different clients. The comparison will highlight the model's ability to integrate these diverse datasets while preserving biological signals, contrasting it with centralized and independently trained models. Key aspects of this comparison include:

*   **Batch Effect Removal:** Assessing how effectively the federated model harmonizes data from different technical batches in the latent space, as visualized through UMAP plots (e.g., in `umap/comparison/`, `umap/federated/` directories).
*   **Cell Type Resolution:** Examining whether the integration of multiple batches compromises the ability to distinguish between different cell types, which can be quantitatively evaluated using metrics like ARI and NMI, as described in Section 3.3.
*   **Model Robustness:** Analyzing the federated model's resilience to inter-client data heterogeneity, a critical factor for real-world distributed learning scenarios. This involves observing if the model can learn a unified representation despite varying batch compositions across clients, potentially leveraging mechanisms within the federated learning framework to mitigate batch-specific biases.