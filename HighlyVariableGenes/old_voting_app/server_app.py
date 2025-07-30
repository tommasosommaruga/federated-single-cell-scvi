"""HighlyVariableGenes: A Flower for highly variable genes detection."""

from logging import DEBUG, INFO
import numpy as np
import json
import os
import random
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler
from flwr.common.record import ParametersRecord
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.strategy import FedAvg

from app.task import get_dummy_start, get_initial_gene_list
SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# Flower ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Round 0: Collect all gene names # NOT WORKING IN SIMULATED APPROACH AS CLIENTS ARE NOT REGISTERED
    # all_gene_names = set()
    # clients = grid.clients()
    # for cid in clients:
    #     props = clients[cid].get_properties({})
    #     all_gene_names.update(props["gene_names"])
    # global_gene_list = sorted(all_gene_names)  # Sorting ensures order consistency

    global_gene_list = get_initial_gene_list()
    print(f"Server global gene list first 10 genes: {global_gene_list[:10]}")
    print(f"Global gene list length: {len(global_gene_list)}")

    def fit_config_fn(rnd: int):
        return {"global_gene_list": json.dumps(global_gene_list)}
    
    # Define strategy (using dummy boolean mask as initial parameters)
    strategy = FedAvg(
        fraction_fit=1.0,
        accept_failures=False,
        fraction_evaluate=0.0,
        initial_parameters=ndarrays_to_parameters([get_dummy_start(global_gene_list)]),
        on_fit_config_fn=fit_config_fn,
    )

    # Setup context and config
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    # Setup workflow with secure aggregation
    update_console_handler(DEBUG, True, True)
    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
    )

    # Start training
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, context)

    # Extract aggregated results
    paramsrecord = context.state[MAIN_PARAMS_RECORD]
    masks = ParametersRecord.to_numpy_ndarrays(paramsrecord)[0]  # Aggregated HVG mask counts
    gene_votes = masks.astype(int)
    # Get indices of top 2000 genes by number of votes
    top_2000_indices = np.argsort(gene_votes)[-2000:][::-1]

    # Adjust indices to be 1-based to match the original gene list
    adjusted_indices = (top_2000_indices - 1).tolist()
    sorted_indices = sorted([int(i) for i in adjusted_indices])

    with open("hvg_indices.json", "w") as f:
        json.dump(sorted_indices, f)

    # Print summary
    log(INFO, "")
    log(INFO, "############################## Final HVG Output ##############################")
    log(INFO, "")
    log(INFO, "Top 2000 Highly Variable Genes (indices): %s", sorted_indices)
    log(INFO, "")
