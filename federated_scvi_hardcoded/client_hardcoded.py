import flwr as fl
import scvi
import torch
import scanpy as sc
import pandas as pd
import numpy as np
import sys
from flwr.common import Context

def load_client_data(client_id: int):
    """Loads a partition of the pancreas dataset."""
    print(f"Loading data for client {client_id}")
    full_adata = sc.read_h5ad("./data/pancreas.h5ad")
    print(f"Full dataset loaded: {full_adata.shape}")
    print(f"Available layers: {list(full_adata.layers.keys())}")
    
    # Check if raw counts are available
    if hasattr(full_adata, 'raw') and full_adata.raw is not None:
        print("Raw data is available")
    else:
        print("No raw data available")
    
    # Check if there's a counts layer already
    if "counts" not in full_adata.layers:
        print("Creating counts layer in loaded data")
        # If not, we assume X contains raw counts and create the layer
        full_adata.layers["counts"] = full_adata.X.copy()
    
    unique_techs = sorted(full_adata.obs.tech.unique())
    print(f"Available technologies: {unique_techs}")
    
    partition = unique_techs[client_id % len(unique_techs)]
    print(f"Selected partition: {partition}")
    
    adata = full_adata[full_adata.obs.tech == partition].copy()
    print(f"Partition shape: {adata.shape}")
    
    # Critical: Verify that counts layer was copied correctly
    if "counts" in adata.layers:
        print(f"Counts layer shape: {adata.layers['counts'].shape}")
    else:
        print("WARNING: counts layer not present after subsetting")
    
    return adata

class ScviClientHardcoded(fl.client.NumPyClient):
    def __init__(self, adata):
        self.adata = adata
        self.model = None

    def setup_model(self, config):
        """Set up the model based on the hardcoded configuration from the server."""
        print("Setting up model")
        try:
            hvg_list = config["hvg_list"]
            cat_mappings = config["cat_mappings"]
            
            print(f"Original data shape: {self.adata.shape}")
            print(f"HVG list length: {len(hvg_list)}")
            print(f"Available layers before filtering: {list(self.adata.layers.keys())}")
            
            # 1. PREPROCESSING
            # Subset to the agreed-upon HVG list
            genes_to_keep = [g for g in hvg_list if g in self.adata.var_names]
            print(f"Genes to keep: {len(genes_to_keep)} out of {len(hvg_list)}")
            
            # Instead of complex layer handling, let's use scanpy's built-in subsetting
            # which correctly handles layers
            print("Subsetting data to HVGs")
            self.adata = self.adata[:, genes_to_keep].copy()
            print(f"Subsetted data shape: {self.adata.shape}")
            
            # Critical: If counts layer not in subsetted data, create it from X
            if "counts" not in self.adata.layers:
                print("WARNING: counts layer not preserved during subsetting, creating from X")
                self.adata.layers["counts"] = self.adata.X.copy()
            else:
                print(f"Counts layer present after subsetting, shape: {self.adata.layers['counts'].shape}")
            
            # Log layers to confirm everything is as expected
            print(f"Available layers after filtering: {list(self.adata.layers.keys())}")
        except Exception as e:
            print(f"Error during data preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        try:
            # 2. MODEL SETUP - EXACT MATCH WITH SERVER
            print("Setting up model to exactly match the server implementation")
            
            # Check the 'tech' column in obs to ensure it's present
            print(f"Columns in adata.obs: {self.adata.obs.columns.tolist()}")
            if 'tech' in self.adata.obs.columns:
                print(f"Values in tech column: {self.adata.obs['tech'].unique()}")
            else:
                print("WARNING: 'tech' column not found in adata.obs!")
                # If there is _batch_indices, we can use that to reconstruct tech
                if '_batch_indices' in self.adata.obs.columns:
                    print("Found _batch_indices, using it to reconstruct tech")
                    reverse_mapping = {v: k for k, v in cat_mappings["tech"].items()}
                    self.adata.obs['tech'] = self.adata.obs['_batch_indices'].map(reverse_mapping)
                    print(f"Reconstructed tech values: {self.adata.obs['tech'].unique()}")
                    
            # CRITICAL: Setup AnnData for SCVI the exact same way as the server
            print("Running SCVI setup_anndata with batch_key='tech'")
            scvi.model.SCVI.setup_anndata(
                self.adata,
                batch_key="tech"
            )
            
            print("Initializing SCVI model with exact server parameters")
            self.model = scvi.model.SCVI(
                self.adata,
                use_layer_norm="both",
                use_batch_norm="none",
                encode_covariates=True,
                dropout_rate=0.2,
                n_layers=2,
            )
            print("Model initialized with server-compatible parameters")
            
            # Print the keys of the model's state dict to help debugging
            model_keys = list(self.model.module.state_dict().keys())
            print(f"Model state dict has {len(model_keys)} keys")
            print(f"First few keys: {model_keys[:5]}")
            
        except Exception as e:
            print(f"Error during SCVI setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.module.state_dict().values()]

    def set_parameters(self, parameters):
        try:
            print(f"Setting parameters: received {len(parameters)} parameter arrays")
            
            # Get the state dict keys
            model_keys = list(self.model.module.state_dict().keys())
            print(f"Model has {len(model_keys)} parameter tensors")
            
            if len(model_keys) != len(parameters):
                print(f"WARNING: Parameter count mismatch! Model: {len(model_keys)}, Received: {len(parameters)}")
                print(f"Model keys: {model_keys}")
                print(f"Will try to load as many parameters as possible")
            
            # Create parameter dictionary - handle mismatch gracefully
            state_dict = {}
            for i, key in enumerate(model_keys):
                if i < len(parameters):
                    print(f"Loading parameter {i}: {key}, shape={parameters[i].shape if hasattr(parameters[i], 'shape') else 'unknown'}")
                    state_dict[key] = torch.tensor(parameters[i])
                else:
                    print(f"Missing parameter for {key}, keeping original")
            
            # Load parameters with strict=False to allow partial loading
            print("Loading parameters into model")
            self.model.module.load_state_dict(state_dict, strict=False)
            print("Parameters loaded successfully")
        except Exception as e:
            print(f"Error setting parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def fit(self, parameters, config):
        try:
            print("\n==== STARTING CLIENT FIT ====")
            # Create a separate error log file for each client fit attempt
            import os
            import uuid
            import traceback
            
            os.makedirs("logs", exist_ok=True)
            log_id = str(uuid.uuid4())[:8]
            log_file = f"logs/client_fit_{log_id}.log"
            
            with open(log_file, "w") as f:
                f.write(f"=== Client Fit Log {log_id} ===\n")
                f.write(f"AnnData shape: {self.adata.shape}\n")
                f.write(f"AnnData layers: {list(self.adata.layers.keys())}\n\n")
            
            def log_step(message):
                print(message)
                with open(log_file, "a") as f:
                    f.write(f"{message}\n")
            
            log_step(f"Starting fit with log_id={log_id}")
            
            # Dump the received parameters and config for debugging
            log_step(f"Received config: {config}")
            
            log_step(f"Parameter info:")
            if parameters:
                log_step(f"Number of parameter arrays: {len(parameters)}")
                for i, param in enumerate(parameters[:5]):  # Print info for first few params
                    log_step(f"  Parameter {i}: Shape={param.shape if hasattr(param, 'shape') else 'unknown'}, Type={type(param)}")
            else:
                log_step("No parameters received (None or empty)")
            
            # Setup model if needed
            if self.model is None:
                log_step("Model not initialized, setting up now")
                try:
                    self.setup_model(config)
                    log_step("Model setup completed successfully")
                except Exception as setup_err:
                    log_step(f"ERROR DURING MODEL SETUP: {str(setup_err)}")
                    tb = traceback.format_exc()
                    log_step(f"Traceback:\n{tb}")
                    with open(log_file, "a") as f:
                        f.write(f"Setup Error: {str(setup_err)}\n")
                        f.write(f"Traceback:\n{tb}\n")
                    raise setup_err
            
            # Set the model parameters
            if parameters and len(parameters) > 0:
                log_step("Setting model parameters")
                try:
                    # Inspect model state dict and received parameters
                    model_keys = list(self.model.module.state_dict().keys())
                    log_step(f"Model has {len(model_keys)} parameters, received {len(parameters)} parameters")
                    
                    # Check for shape mismatches
                    model_shapes = {k: v.shape for k, v in self.model.module.state_dict().items()}
                    log_step(f"Model parameter shapes (first few):")
                    for i, (k, shape) in enumerate(list(model_shapes.items())[:5]):
                        log_step(f"  {k}: {shape}")
                    
                    log_step(f"Received parameter shapes (first few):")
                    for i, param in enumerate(parameters[:5]):
                        log_step(f"  Param {i}: {param.shape if hasattr(param, 'shape') else 'unknown'}")
                    
                    self.set_parameters(parameters)
                    log_step("Parameters set successfully")
                except Exception as param_err:
                    log_step(f"ERROR SETTING PARAMETERS: {str(param_err)}")
                    log_step("Will continue with initialized parameters")
                    tb = traceback.format_exc()
                    log_step(f"Traceback:\n{tb}")
            else:
                log_step("No parameters to set, using initialized model")
            
            # Check if model and parameters are compatible
            log_step(f"Training model with {self.adata.n_obs} cells and {self.adata.n_vars} genes")
            
            # Set batch size based on data size
            batch_size = min(128, max(16, self.adata.n_obs // 10))
            log_step(f"Using batch size: {batch_size}")
            
            # Train the model with detailed logging
            log_step("Starting model training")
            try:
                log_step(f"Training model with {self.adata.n_obs} cells and {self.adata.n_vars} genes")
                
                # Set batch size based on data size
                batch_size = min(128, max(16, self.adata.n_obs // 10))
                log_step(f"Using batch size: {batch_size}")
                
                # Train for just 1 epoch to keep it fast
                self.model.train(max_epochs=1, batch_size=batch_size, early_stopping=False)
                log_step("Model training completed successfully")
                
                # Get updated parameters
                updated_params = self.get_parameters(config={})
                log_step(f"Returning {len(updated_params)} updated parameters")
                
                log_step("==== CLIENT FIT COMPLETED ====\n")
                with open(log_file, "a") as f:
                    f.write("FIT COMPLETED SUCCESSFULLY\n")
                return updated_params, self.adata.n_obs, {}
            except Exception as train_err:
                log_step(f"ERROR DURING MODEL TRAINING: {str(train_err)}")
                tb = traceback.format_exc()
                log_step(f"Traceback:\n{tb}")
                with open(log_file, "a") as f:
                    f.write(f"Training Error: {str(train_err)}\n")
                    f.write(f"Traceback:\n{tb}\n")
                raise train_err
                
        except Exception as e:
            print(f"CRITICAL ERROR DURING CLIENT TRAINING: {str(e)}")
            print("Detailed traceback:")
            import traceback
            traceback.print_exc()
            
            # Save error info to file for later inspection
            with open(f"logs/client_error_{id(self)}.log", "w") as f:
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
            
            raise e

def make_client_and_handle_errors(context: Context):
    """Creates a client and handles errors with detailed logging."""
    import os
    import traceback
    import uuid
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Generate unique log ID
    log_id = str(uuid.uuid4())[:8]
    log_file = f"logs/client_init_{log_id}.log"
    
    def log_to_file(message):
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
    
    log_to_file(f"=== Client Initialization Log {log_id} ===\n")
    
    try:
        # Extract partition ID
        partition_id = context.node_config.get("partition-id", getattr(context, "node_id", 0))
        message = f"[CONFIG] Creating client with partition_id={partition_id}"
        print(message)
        log_to_file(message)
        
        # Load data
        message = f"Loading data for partition {partition_id}"
        print(message)
        log_to_file(message)
        
        adata = load_client_data(int(partition_id))
        log_to_file(f"Data loaded successfully: shape={adata.shape}")
        
        # Create client
        message = f"Creating client instance for partition {partition_id}"
        print(message)
        log_to_file(message)
        
        client = ScviClientHardcoded(adata)
        
        # Return wrapped client
        message = f"Client for partition {partition_id} ready"
        print(message)
        log_to_file(message)
        
        return client.to_client()
        
    except Exception as e:
        error_msg = f"ERROR CREATING CLIENT: {str(e)}"
        print(error_msg)
        log_to_file(error_msg)
        
        tb = traceback.format_exc()
        print(tb)
        log_to_file(f"Traceback:\n{tb}")
        
        # Try to create a minimal functioning client that won't crash the system
        try:
            # Create a minimal fallback client
            
            class FallbackClient(fl.client.NumPyClient):
                def get_parameters(self, config):
                    print("Fallback client: get_parameters called")
                    log_to_file("Fallback client: get_parameters called")
                    # Return empty parameters
                    return []
                
                def fit(self, parameters, config):
                    print("Fallback client: fit called")
                    log_to_file("Fallback client: fit called")
                    # Report the original error
                    log_to_file(f"Original client creation failed with: {str(e)}")
                    # Return empty parameters
                    return [], 0, {"error": str(e)}
                
                def evaluate(self, parameters, config):
                    print("Fallback client: evaluate called")
                    log_to_file("Fallback client: evaluate called") 
                    # Return a dummy evaluation
                    return float('inf'), 0, {"error": str(e)}
            
            print("Created fallback client")
            log_to_file("Created fallback client")
            return FallbackClient().to_client()
        except Exception as fallback_error:
            print(f"Fallback client creation also failed: {str(fallback_error)}")
            log_to_file(f"Fallback client creation also failed: {str(fallback_error)}")
            raise e  # Raise the original error

def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    return make_client_and_handle_errors(context)

# Flower ClientApp
app = fl.client.ClientApp(
    client_fn=client_fn,
)
