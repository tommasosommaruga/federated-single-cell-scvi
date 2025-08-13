import scanpy as sc
import anndata
import json
import os

# Use only the train set for HVG selection
adata_path = os.path.join('..','0_data', 'pancreas_train.h5ad')
hvg_path = os.path.join('..','0_data', 'hvg_list.json')

print(f'Loading train AnnData from {adata_path}...')
adata = sc.read_h5ad(adata_path)
# Convert counts layer to int32 if present
if hasattr(adata, 'layers') and 'counts' in adata.layers:
    adata.layers['counts'] = adata.layers['counts'].astype('int32')
print(f'Loaded train AnnData with {adata.n_obs} cells and {adata.n_vars} genes.')

# Filter reference data as in server_scvi.py
ref_mask = (~adata.obs["tech"].isin(["smartseq2", "celseq2"])).values
adata_ref = adata[ref_mask].copy()

print(f'Filtered reference AnnData: {adata_ref.n_obs} cells, {adata_ref.n_vars} genes.')

# Compute highly variable genes (HVGs) on reference data
print('Computing highly variable genes (HVGs) on reference data...')
sc.pp.highly_variable_genes(adata_ref, n_top_genes=2000, batch_key='tech', subset=False)
hvg_list = adata_ref.var[adata_ref.var['highly_variable']].index.tolist()
print(f'Number of HVGs: {len(hvg_list)}')

# Save HVG list to JSON
print(f'Saving HVG list to {hvg_path}...')
with open(hvg_path, 'w') as f:
    json.dump(hvg_list, f)
print('Done.') 