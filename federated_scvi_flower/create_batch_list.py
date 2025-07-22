import anndata
import json
import os
from typing import List

def extract_all_batches(h5ad_paths: List[str], batch_column: str = "tech") -> List[str]:
    all_batches = set()

    for path in h5ad_paths:
        print(f"Reading {path}...")
        adata = anndata.read_h5ad(path)
        if batch_column not in adata.obs.columns:
            raise ValueError(f"'{batch_column}' column not found in {path}")
        batches = adata.obs[batch_column].unique().tolist()
        all_batches.update(batches)

    return sorted(list(all_batches))

def save_batch_list(batches: List[str], out_path: str):
    with open(out_path, "w") as f:
        json.dump(batches, f, indent=2)
    print(f"Saved batch list to {out_path}")

if __name__ == "__main__":
    # Example usage
    train_path = "data/pancreas_train.h5ad"
    test_path = "data/pancreas_test.h5ad"
    output_path = "data/batch_list.json"

    batch_list = extract_all_batches([train_path, test_path], batch_column="tech")
    save_batch_list(batch_list, output_path)
