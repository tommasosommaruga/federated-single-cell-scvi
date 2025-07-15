import os
import sys
import logging
import anndata
import json

# Ensure the project root is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federated_scvi_flower.data_utils_scvi import load_partitioned_anndata, ensure_hvg_genes
from federated_scvi_flower.model_utils_scvi import get_scvi_model, train_scvi, evaluate_scvi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hvg_list(hvg_list_path):
    with open(hvg_list_path) as f:
        return json.load(f)

def main():
    partition_id = 0  # Change this to use a different partition
    num_partitions = 1  # Only one client/partition
    hvg_list_path = os.path.join("data", "hvg_list.json")
    adata = load_partitioned_anndata(partition_id, num_partitions)
    hvg_list = load_hvg_list(hvg_list_path)
    adata = ensure_hvg_genes(adata, hvg_list, partition_id=partition_id)
    logger.info(f"[independent_client] AnnData shape before SCVI init: {adata.shape}, genes: {list(adata.var_names[:10])}")
    model = get_scvi_model(adata, hvg_list, partition_id=partition_id)
    logger.info("[independent_client] Training SCVI model...")
    train_loss = train_scvi(model, adata, max_epochs=10)
    logger.info(f"[independent_client] Training complete. Final train loss: {train_loss}")
    test_loss = evaluate_scvi(model, adata)
    logger.info(f"[independent_client] Evaluation complete. Test loss: {test_loss}")

if __name__ == "__main__":
    main() 