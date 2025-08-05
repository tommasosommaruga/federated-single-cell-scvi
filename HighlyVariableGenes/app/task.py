import numpy as np
import pandas as pd
import anndata
import logging
import os

# SAME FUNCTION AS IN federated_scvi_flower/data_utils_scvi.py
def load_partitioned_anndata(partition_id, num_partitions, adata_path="data/pancreas_train.h5ad"):
    adata = anndata.read_h5ad(adata_path)
    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Loaded filtered AnnData: shape={adata.shape}, genes={list(adata.var_names[:10])}")
    
    if "tech" in adata.obs:
        techs = sorted(adata.obs["tech"].unique().tolist())
        tech_groups = np.array_split(techs, num_partitions)
        client_techs = tech_groups[partition_id]
        mask = adata.obs["tech"].isin(client_techs)
        adata_part = adata[mask].copy()
    else:
        idxs = np.array_split(np.arange(adata.n_obs), num_partitions)
        adata_part = adata[idxs[partition_id]].copy()

    logging.info(f"[load_partitioned_anndata][partition {partition_id}] Partitioned AnnData: shape={adata_part.shape}, genes={list(adata_part.var_names[:10])}")
    return adata_part

### Initial conditions for FL process

def get_dummy_start(global_gene_list):
    # Create a float32 mask of zeros
    dummy = np.zeros(len(global_gene_list), dtype=np.float32)
    print(f"Dummy mask with {len(global_gene_list)} genes initialized.")
    return dummy

def get_initial_gene_list(path="data/pancreas_train.h5ad"):
    adata = anndata.read_h5ad(path)
    global_gene_list = sorted(adata.var_names.tolist())
    print(f"Global gene list: {global_gene_list[:10]}... Total genes: {len(global_gene_list)}")
    return global_gene_list

def get_gene_index_name_dict(path) -> dict[int, str]:
    adata = anndata.read_h5ad(path)
    global_gene_list = sorted(adata.var_names.tolist())
    index_to_name = {i: gene for i, gene in enumerate(global_gene_list)}
    return index_to_name

# ### Results of federated process
    
# def get_output_df(expr_array: np.array):
#     """
#     Converts a NumPy expression array into a formatted DataFrame with gene names as columns.

#     Args:
#         expr_array (np.array): A 2D NumPy array where the first row contains gene expression values.

#     Returns:
#         pd.DataFrame: A DataFrame with gene names as columns and a single row of percentage-based values.
#     """
#     # Extract and round the first row of expression values, converting them to percentages
#     values = expr_array[0].round(2)*100

#     # Retrieve gene names based on the number of genes in the expression array
#     gene_names = get_gene_names(values.shape[1]) 

#     # Create a DataFrame with gene names as column headers
#     df_expr = pd.DataFrame(values, columns=gene_names)

#     return df_expr
    
# def get_output_list(dataset: pd.DataFrame, threshold: int):
#     """
#     Extracts a list of gene names where the low-expressed percentage is greater than the given threshold.

#     Args:
#         dataset (pd.DataFrame): A DataFrame with gene names as columns and a single row of expression values.
#         threshold (int): The threshold percentage for filtering genes.

#     Returns:
#         list: A list of gene names that are weakly expressed.
#     """
#     # Select the first row (assuming only one row in the dataset)
#     values = dataset.iloc[0]

#     # Filter genes where the value exceeds the threshold
#     return values[values > threshold].index.tolist()


# ### Create and partition simulation dataset

# def get_gene_names(num_genes: int):

#     return [f"Gene_{i+1}" for i in range(num_genes)]    

# def generate_gene_expression_data(num_individuals: int, num_genes: int, seed_value: int):
#     """
#     Generates a synthetic gene expression dataset with given parameters.
    
#     Parameters:
#         num_individuals (int): Number of individuals (samples).
#         num_genes (int): Number of genes (features).
#         seed_value (int): Random seed for reproducibility.
    
#     Returns:
#         pd.DataFrame: A dataframe with gene expression levels.
#     """
#     np.random.seed(seed_value)

#     # Define gene names
#     gene_names = get_gene_names(num_genes)  

#     # Random mean (10-50) and std (2-10) for each gene
#     mean_values = np.random.uniform(5, 25, num_genes)
#     std_dev_values = np.random.uniform(2, 5, num_genes)

#     # Generate gene expression levels
#     data = np.zeros((num_individuals, num_genes))
#     for i in range(num_genes):
#         gene_data = np.abs(np.random.normal(mean_values[i], std_dev_values[i], num_individuals))
#         data[:, i] = np.clip(gene_data, 0, 100)  # Clip to range [0, 100]

#     # Create DataFrame
#     dataset = pd.DataFrame(data, columns=gene_names)

#     return dataset

# partitioner = None

# def load_data(partition_id: int, num_partitions: int, num_individuals: int, num_genes: int, seed_value: int):
#     """
#     Loads a partition of the synthetic gene expression dataset.
    
#     Parameters:
#         partition_id (int): The ID of the partition to load.
#         num_partitions (int): Total number of partitions.
#         num_individuals (int): Number of individuals in the dataset.
#         num_genes (int): Number of genes in the dataset.
#         seed_value (int): Random seed for reproducibility.

#     Returns:
#         pd.DataFrame: The selected partition of the dataset.
#     """
#     global partitioner
    
#     if partitioner is None:  # Create partitioner only once
        
#         dataset = generate_gene_expression_data(num_individuals, num_genes, seed_value)
#         dataset = Dataset.from_pandas(dataset)

#         partitioner = IidPartitioner(num_partitions)
#         partitioner.dataset = dataset
        
#     partition = partitioner.load_partition(partition_id).with_format("pandas").to_pandas()
    
#     return partition
