import scanpy as sc
import anndata
import numpy as np
import os

adata_path = 'data/pancreas.h5ad'
train_path = 'data/pancreas_train.h5ad'
val_path = 'data/pancreas_val.h5ad'
test_path = 'data/pancreas_test.h5ad'

split_mode = "train_val_test"  # "simple" or "train_val_test"

print(f'Loading AnnData from {adata_path}...')
adata = sc.read_h5ad(adata_path)
print(f'Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.')

if hasattr(adata, 'layers') and 'counts' in adata.layers:
    adata.layers['counts'] = adata.layers['counts'].astype('int32')

np.random.seed(55)

# def stratified_split(adata, ratios):
#     """Stratified split by 'tech' or random if no 'tech'."""
#     train_idx, val_idx, test_idx = [], [], []
#     if "tech" in adata.obs:
#         print("Using the stratified_split function with 'tech' stratification.")
#         for tech in adata.obs["tech"].unique():
#             idx = np.where(adata.obs["tech"] == tech)[0]
#             n = len(idx)
#             n_train = int(ratios[0]*n)
#             n_val = int(ratios[1]*n) if len(ratios) > 1 else 0
#             perm = np.random.permutation(idx)
#             train_idx.extend(perm[:n_train])
#             if n_val:
#                 val_idx.extend(perm[n_train:n_train+n_val])
#             test_idx.extend(perm[n_train+n_val:] if n_val else perm[n_train:])
#     else:
#         print("Using the random split")
#         n = adata.n_obs
#         n_train = int(ratios[0]*n)
#         n_val = int(ratios[1]*n) if len(ratios) > 1 else 0
#         perm = np.random.permutation(n)
#         train_idx.extend(perm[:n_train])
#         if n_val:
#             val_idx.extend(perm[n_train:n_train+n_val])
#         test_idx.extend(perm[n_train+n_val:] if n_val else perm[n_train:])
#     return np.array(train_idx), np.array(val_idx), np.array(test_idx)

# if split_mode == "simple":
#     ref_mask = (~adata.obs["tech"].isin(["smartseq2", "celseq2"])) if "tech" in adata.obs else np.ones(adata.n_obs, bool)
#     adata_ref = adata[ref_mask].copy()
#     train_idx, _, test_idx = stratified_split(adata_ref, [0.8])
#     adata_train = adata_ref[train_idx].copy()
#     adata_test = adata_ref[test_idx].copy()
#     print(f'Train: {adata_train.n_obs} cells, Test: {adata_test.n_obs} cells')
#     adata_train.write(train_path)
#     adata_test.write(test_path)

if split_mode == "train_val_test":
    test_mask_tech = adata.obs["tech"].isin(["smartseq2", "celseq2"]) if "tech" in adata.obs else np.zeros(adata.n_obs, bool)
    test_data_tech = adata[test_mask_tech].copy()
    adata_remaining = adata[~test_mask_tech].copy()
    print(f'Train: {adata_remaining.n_obs}, Test: {test_data_tech.n_obs}')

    test_data_tech.write(test_path)
    adata_remaining.write(train_path)
    # commented the lines below, and splitting just by tech
    # train_idx, val_idx, test_idx = stratified_split(adata_remaining, [0.7, 0.1])
    # adata_train = adata_remaining[train_idx].copy()
    # adata_val = adata_remaining[val_idx].copy()
    # adata_test_split = adata_remaining[test_idx].copy()
    # adata_test = anndata.concat([test_data_tech, adata_test_split], join='inner')
    # print(f'Train: {adata_train.n_obs}, Val: {adata_val.n_obs}, Test: {adata_test.n_obs}')
    # adata_train.write(train_path)
    # # adata_val.write(val_path)
    # adata_test.write(test_path)

else:
    raise ValueError("split_mode must be 'simple' or 'train_val_test'")

print("Done.")
