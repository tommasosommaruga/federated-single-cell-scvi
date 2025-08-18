# 0 Introduction
    Brief overview of the thesis, problem statement, and main contributions.

# 1 Background

## 1.1 Variational Autoencoders (VAEs)
    ### 1.1.1 Fundamental Principles of VAEs
        Explanation of encoder-decoder architecture, latent space, probabilistic modeling, and the reparameterization trick.
    ### 1.1.2 VAEs in Single-Cell Transcriptomics
        How VAEs are adapted for scRNA-seq data, addressing sparsity and high dimensionality.

## 1.2 Federated Learning (FL)
    ### 1.2.1 Core Concepts and Principles of Federated Learning
        Definition, collaborative learning, data privacy, and iterative model aggregation.
    ### 1.2.2 Advantages and Challenges of Federated Learning
        Benefits (privacy, data ownership) and hurdles (data heterogeneity, communication overhead, security).
    ### 1.2.3 Federated Learning Frameworks (e.g., Flower)
        Brief mention of the framework used and its role in the project.

## 1.3 Single-Cell RNA Sequencing (scRNA-seq)
    ### 1.3.1 Overview of scRNA-seq Data Characteristics
        Discussion of data types, inherent noise, sparsity, and batch effects.
    ### 1.3.2 Preprocessing Steps for scRNA-seq Data
        Common steps like normalization, log-transformation, and feature selection (e.g., Highly Variable Genes).

## 1.4 Single-Cell Variational Inference (scVI)
    ### 1.4.1 Introduction to scVI
        What scVI is, its purpose in single-cell data integration and latent space learning.
    ### 1.4.2 scVI Model Architecture and Components
        Detailed breakdown of the scVI neural network, including conditional parameters.
    ### 1.4.3 Training Objective and Loss Functions in scVI
        Explanation of the ELBO, reconstruction loss, and KL divergence components.

# 2 Federation of Single-Cell Variational Inference (scVI)

## 2.1 Federated Highly Variable Gene (HVG) Selection Application
    ### 2.1.1 Motivation for Federated HVG Selection
        Why HVG selection is critical and why a federated approach is beneficial for privacy and distributed data.
    ### 2.1.2 Architecture of the Federated HVG Application
        Description of the client-server interaction for HVG computation (referencing `HighlyVariableGenes/app/`).
    ### 2.1.3 Client-Side HVG Calculation Methodology
        How clients compute local gene variances or statistics without sharing raw data (referencing `HighlyVariableGenes/app/client_app.py`).
    ### 2.1.4 Server-Side Aggregation and Global HVG Selection
        Details on how the server aggregates client contributions to determine global HVGs (referencing `HighlyVariableGenes/app/server_app.py`).

## 2.2 Federated scVI Model Implementation
    ### 2.2.1 Federated Learning Setup for scVI
        This subsection describes the overall architecture and configuration of the federated scVI training process. The project leverages the Flower federated learning framework. Key configurable parameters for the federated training include the total `num_clients` participating in each round, the `num_rounds` of global communication, and the `local-epochs` each client trains for before sending updates to the server. The setup ensures that all clients are available for both fitting and evaluation in each round.

    ### 2.2.2 Client-Side scVI Model Training and Updates
        The client-side implementation, primarily found in `federated_scvi_flower/app/client_scvi.py`, defines the `ScviClient` class which inherits from `flwr.client.NumPyClient`. Each client is responsible for:
        *   **Data Loading and Preparation:** Loading its specific data partition (`load_partitioned_anndata`) for training and the full `pancreas_test.h5ad` for local evaluation. The `ensure_hvg_genes` utility ensures that only highly variable genes are used, and the 'tech' observation is explicitly assigned to the 'batch' key (`adata.obs['batch'] = adata.obs['tech']`) for scVI model setup, enabling proper batch correction.
        *   **Model Initialization:** Initializing a local scVI model (`get_scvi_model`) using its `adata` and the global highly variable gene list.
        *   **Parameter Exchange:** Implementing `get_parameters` to send its current model weights to the server and `set_parameters` to receive the aggregated global model weights.
        *   **Local Training (Fit):** The `fit` method orchestrates the local training loop, where the client's scVI model is trained on its private `adata` for a specified number of `epochs` (controlled by the server's configuration). The `train_scvi` utility handles the actual training process, and the updated local model parameters are returned along with the number of observed cells and training loss.
        *   **Local Evaluation:** The `evaluate` method assesses the performance of the client's current model on the held-out test set (`adata_test`), returning the test loss and the number of test observations.

    ### 2.2.3 Server-Side Model Aggregation Strategy
        The server-side logic, implemented in `federated_scvi_flower/app/server_scvi.py`, defines the `server_fn` that sets up the federated learning server. It manages the global model and coordinates client interactions. Key components and processes include:
        *   **Parameter Initialization:** A dummy AnnData object (`create_dummy_adata`) is used to initialize a global scVI model (`get_scvi_model`) on the server. The initial weights of this model are then used to set the `initial_parameters` for the aggregation strategy.
        *   **Custom Aggregation Strategy (`FedAvgWithEval`):** The server utilizes a custom `FedAvgWithEval` strategy, which extends Flower's `FedAvg` strategy. This custom strategy not only aggregates the model updates received from clients but also performs a global evaluation of the aggregated model on the full `pancreas_test.h5ad` dataset after each communication round.
        *   **Global Model Evaluation and Logging:** Within the `aggregate_fit` method of `FedAvgWithEval`, the aggregated model parameters are applied to the server's global scVI model. This model is then evaluated on the `adata_test` using `evaluate_scvi`, and the resulting test loss for each round is logged to a CSV file (e.g., `loss_logs/loss_curve_federated_...csv`) for subsequent analysis.
        *   **Model Saving:** An `atexit` hook ensures that the `save_final_model_and_adata` function is called upon server shutdown. This function saves the final aggregated global model's state dictionary to a `.pt` file, typically located in `federated_scvi_flower/app/models/`, using a naming convention that reflects the number of clients, rounds, and epochs.

    ### 2.2.4 Data Characteristics and Distribution Across Clients in FL-scVI
        The data used in the federated scVI experiments is distributed among clients based on a partitioning strategy. While the main dataset is `pancreas.h5ad`, the splitting process described in `0_data/split_data.py` ensures that the test set includes specific technologies (`smartseq2`, `celseq2`). For federated training, each client receives a partition of the training data. Crucially, the `tech` column in the AnnData objects is explicitly used as the `batch` key during the scVI model setup (`adata.obs['batch'] = adata.obs['tech']`). This means that each client may implicitly hold data primarily from one or more technical batches, making the federated learning setup relevant for integrating data from diverse technological origins without centralizing the raw data.

# 3 Experiments and Results

## 3.1 Data Description and Experimental Setup
    ### 3.1.1 Pancreas Dataset Overview
        Detailed description of the pancreas dataset (`pancreas.h5ad`), its origin, cell types, and technological batches.
    ### 3.1.2 Data Preprocessing and Highly Variable Gene Selection
        Specific steps for preparing the data, including normalization and the use of the federated HVG selection method.
    ### 3.1.3 Data Splitting Strategy for Training and Evaluation
        Elaboration on the `0_data/split_data.py` methodology, particularly the `train_val_test` split and the designation of `smartseq2` and `celseq2` for the test set.
    ### 3.1.4 Experimental Configurations for Model Training
        Overview of the different models trained:
        #### 3.1.4.1 Centralized scVI Model Configuration
            Description of the centralized training approach (`1_centralised/centralised.py`).
        #### 3.1.4.2 Independent Client scVI Models Configuration
            Description of the independent training approach for each client (`2_independent/independent_client_scvi.py`).
        #### 3.1.4.3 Federated scVI Model Configurations
            Details on the various federated learning setups, including different combinations of communication rounds and local epochs (referencing model directories like `federated_scvi_flower/app/models/`).

## 3.2 Model Training Dynamics and Loss Analysis
    ### 3.2.1 Metrics for Loss Evaluation
        Explanation of what the training and testing loss values represent in the context of scVI.
    ### 3.2.2 Centralized Model Loss Curves
        Analysis of `loss_curve_centralised.csv` and corresponding plots in `loss_logs/compare_models_plot/`.
    ### 3.2.3 Independent Client Model Loss Curves
        Analysis of `loss_curve_client_0.csv`, `loss_curve_client_1.csv`, etc., and associated plots.
    ### 3.2.4 Federated Model Loss Curves Across Configurations
        Detailed analysis of various `loss_curve_federated_X_clients_Y_rounds_Z_epochs.csv` files, demonstrating the impact of different FL parameters on convergence.
    ### 3.2.5 Comparative Analysis of Loss Trends
        Comparison of loss trajectories between centralized, independent, and federated models, highlighting convergence and stability.

## 3.3 Latent Space Visualization and Quality Metrics
    ### 3.3.1 UMAP Visualization Methodology
        Explanation of UMAP as a dimensionality reduction technique and its application in visualizing scRNA-seq latent spaces.
    ### 3.3.2 UMAP Visualizations: Batch Effect Correction
        Presentation and discussion of UMAP plots (from `umap/centralised/`, `umap/independent_client_X/`, `umap/federated/`, `umap/comparison/`) demonstrating batch effect integration across models.
    ### 3.3.3 UMAP Visualizations: Cell Type Preservation
        Analysis of UMAP plots showing the distinct clustering of cell types within the integrated latent spaces.
    ### 3.3.4 Quantitative Evaluation Metrics for Latent Space Quality
        Definition of the chosen metrics (ARI, NMI, Silhouette Score) and their significance in evaluating clustering and batch correction (referencing `umap/latent_space_clustering_metrics.ipynb`).
    ### 3.3.5 Performance Comparison: Adjusted Rand Index (ARI)
        Comparison of ARI values across centralized, independent, and federated models, showing accuracy of cell type clustering.
    ### 3.3.6 Performance Comparison: Normalized Mutual Information (NMI)
        Comparison of NMI values, further assessing the agreement between true and predicted clusters.
    ### 3.3.7 Performance Comparison: Silhouette Score
        Comparison of Silhouette Scores, indicating the compactness and separation of clusters.

## 3.4 Specific Experimental Scenarios and Robustness Analysis
    ### 3.4.1 Comparison with "One Batch Per Client" Scenario
        In-depth discussion on how the federated setup addresses the challenge of clients having data primarily from a single technological batch, and the implications for generalization.
    ### 3.4.2 Impact of Communication Rounds and Local Epochs on Federated Performance
        Analysis of how varying these federated learning parameters affects model quality and convergence speed.
    ### 3.4.3 Analysis of Specific Federated Configurations
        Highlighting results from key federated runs (e.g., `3_clients_20_rounds_5_epochs`) and their significance.
    ### 3.4.4 Robustness to Data Heterogeneity and Imbalance
        Discussion on how the federated scVI model handles variations in data size, cell type composition, or technological batches across clients.

# 5 Conclusion

## 5.1 Future Works and Challenges
    ### 5.1.1 Addressing Computational Scalability and Efficiency
        Potential improvements for handling larger datasets or more clients, including memory optimization techniques (e.g., reducing reliance on explicit garbage collection, as seen with `gc.collect()`), and exploring more efficient data loading and processing pipelines. Further investigation into addressing any non-deterministic operations for enhanced reproducibility (as indicated by `torch.use_deterministic_algorithms(True, warn_only=True)`).

    ### 5.1.2 Dynamic Client Participation and Robustness
        Currently, the federated setup likely assumes a fixed number of available clients. A significant improvement would be to implement and test strategies for dynamic client participation, where clients can join or leave the federation throughout the training process. This would require enhancing the robustness of the aggregation process to handle partial client responses and network instabilities more gracefully, mirroring real-world federated deployment scenarios.

    ### 5.1.3 Exploring Diverse Federated Averaging Strategies
        While FedAvg is a strong baseline, further research could focus on implementing and evaluating other advanced federated averaging algorithms. Strategies like FedProx, SCAFFOLD, or personalized federated learning approaches could be explored to better address data heterogeneity (non-IID data) among clients, potentially leading to faster convergence, improved model performance, or more robust generalization across diverse client datasets.

    ### 5.1.4 Applications to Diverse scRNA-seq Datasets
        Future work on testing the federated scVI model on different biological contexts 
        and data types.
    ### 5.1.5 Optimized On-Device/Client-Side Computation
        A key area for enhancement is optimizing the computational efficiency of the scVI training process on individual client devices. This could involve investigating techniques such as model quantization, pruning, or knowledge distillation to reduce the computational resources required per client, thereby making the federated solution more viable for deployment on edge devices or less powerful computing infrastructures.

    ### 5.1.6 Enhanced Framework Understanding for Granular Optimization and Parallelization
        A critical future work involves a deeper dive into the internal mechanisms of the Flower framework and its interaction with the scVI library. This understanding is essential for enabling more granular optimization of communication patterns, custom strategy development, and precisely identifying which components of the scVI library's computations can be effectively parallelized (e.g., using PyTorch's distributed data parallel or other multiprocessing techniques). This involves careful consideration of the trade-offs between local computation and communication overhead, aiming to maximize computational gains while ensuring data integrity and consistency across distributed processes.

    ### 5.1.5 Challenges in Framework Understanding and Parallelization
        A significant challenge lies in a deeper understanding of the Flower framework's internal mechanisms to fine-tune its behavior for optimal performance in distributed environments. This extends to the complex task of identifying which components of the scVI library's computations can be effectively parallelized across clients or within client-side training (e.g., using PyTorch's distributed data parallel or other multiprocessing techniques), and which operations are inherently sequential or require careful synchronization and aggregation. This involves balancing communication overhead with computational gains and ensuring data integrity across distributed processes. Enhancing the robustness of critical operations like final model saving (`_save_on_exit` in `server_scvi.py`) would also be a valuable improvement.

## 5.2 Critical Evaluation and Summary
    ### 5.2.1 Strengths and Key Advantages of the Federated scVI Approach
        Summarizing the benefits, especially regarding data privacy, integration of distributed data, and performance.
    ### 5.2.2 Limitations and Potential Improvements of the Current Implementation
        Acknowledging shortcomings and proposing ways to enhance the model or framework.
    ### 5.2.3 Overall Contributions and Impact of the Project
        Restating the main achievements and their significance in the field of single-cell analysis and federated learning.