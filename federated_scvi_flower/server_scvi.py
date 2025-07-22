from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from federated_scvi_flower.model_utils_scvi import get_scvi_model, get_weights, set_weights, setup_scvi_anndata, evaluate_scvi
import os
import csv
import torch
from federated_scvi_flower.data_utils_scvi import ensure_hvg_genes, create_dummy_adata, load_batch_list, load_hvg_list
import atexit
import anndata as ad

model = None
adata_ref = None
global_strategy = None
adata_test = None

class FedAvgWithEval(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        global model, adata_test
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_result is not None:
            self.final_parameters = aggregated_result[0]
            print(f"[Server] Round {rnd} aggregated parameters successfully.", adata_test)
            # Evaluate and log after each round
            if model is not None and adata_test is not None:
                weights = parameters_to_ndarrays(self.final_parameters)
                set_weights(model, weights)
                model.is_trained = True
                loss = evaluate_scvi(model, adata_test)
                print(f"[Server] Round {rnd} test loss: {loss:.4f}")
                self.log_to_csv(rnd, loss)

        return aggregated_result

    def log_to_csv(self, round_number, loss_value):
        log_path = os.path.join("loss_logs","loss_curve_federated.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.isfile(log_path)

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["round", "test_loss"])
            writer.writerow([round_number, loss_value])

def server_fn(context: Context) -> ServerAppComponents:
    global model, adata_ref, global_strategy, adata_test

    num_clients = int(context.run_config.get("num_clients", 2))
    num_rounds = int(context.run_config.get("num_rounds", 3))
    epochs = int(context.run_config.get("local-epochs", 1))

    print(f"[server_scvi] num_clients: {num_clients}, context.run_config: {context.run_config}")

    # Load HVG list and batch list
    hvg_list_path = context.run_config.get("hvg_list_path", "data/hvg_list.json")
    batch_list_path = context.run_config.get("batch_list_path", "data/batch_list.json")
    hvg_list = load_hvg_list(hvg_list_path)
    batch_list = load_batch_list(batch_list_path)

    # Create dummy AnnData for initialization
    adata_ref = create_dummy_adata(num_cells=10, num_genes=len(hvg_list), batches=batch_list)

    # Make sure the gene names match the HVG list
    adata_ref = ensure_hvg_genes(adata_ref, hvg_list)

    # Set up AnnData for scVI
    setup_scvi_anndata(adata_ref, all_batches=batch_list)

    # Load test data
    adata_test_path = context.run_config.get("adata_test_path", "data/pancreas_test.h5ad")
    adata_test = ad.read_h5ad(adata_test_path)
    adata_test = ensure_hvg_genes(adata_test, hvg_list)
    setup_scvi_anndata(adata_test, all_batches=batch_list)

    # Create model using dummy data
    model = get_scvi_model(adata_ref, hvg_list)

    # Get initial weights
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    # Define strategy
    strategy = FedAvgWithEval(
        initial_parameters=initial_parameters,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"epochs": epochs}
    )
    global_strategy = strategy

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

app = ServerApp(server_fn=server_fn)

# Save final model weights
def save_final_model_and_adata(model, path_prefix="federated_scvi_flower/final_server_model"):
    model_path = f"{path_prefix}.pt"
    # Use model.state_dict() unless your model is wrapped in DataParallel
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Saved final model to {model_path}")

def _save_on_exit():
    global model, adata_ref, global_strategy
    try:
        if model is not None and adata_ref is not None and global_strategy is not None:
            final_parameters = global_strategy.final_parameters
            if final_parameters is not None:
                final_weights = parameters_to_ndarrays(final_parameters)
                set_weights(model, final_weights)
                model.is_trained = True
                save_final_model_and_adata(model)
            else:
                print("[WARN] final_parameters is None, cannot save model weights")
        else:
            print("[WARN] Model, AnnData, or strategy not defined at exit, not saving.")
    except Exception as e:
        print(f"[WARN] Could not save final model: {e}")

atexit.register(_save_on_exit)
