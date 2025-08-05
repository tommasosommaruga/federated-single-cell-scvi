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

from old_wrong_app.task import get_initial_gene_list, get_dummy_start,get_gene_index_name_dict

SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    global_gene_list = get_initial_gene_list()
    print(f"Server global gene list first 10 genes: {global_gene_list[:10]}")
    print(f"Global gene list length: {len(global_gene_list)}")

    def fit_config_fn(rnd: int):
        return {"global_gene_list": json.dumps(global_gene_list)}

    # Initialize with zeros float mask (scores)
    initial_params = ndarrays_to_parameters([get_dummy_start(global_gene_list)])

    strategy = FedAvg(
        fraction_fit=1.0,
        accept_failures=False,
        fraction_evaluate=0.0,
        initial_parameters=initial_params,
        on_fit_config_fn=fit_config_fn,
    )

    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    update_console_handler(DEBUG, True, True)
    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
    )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, context)

    paramsrecord = context.state[MAIN_PARAMS_RECORD]
    aggregated_scores = ParametersRecord.to_numpy_ndarrays(paramsrecord)[0]  # float sums

    # Select top 2000 genes by aggregated scores
    top_2000_indices = np.argsort(aggregated_scores)[-2000:][::-1]
    sorted_indices = sorted([int(i) for i in top_2000_indices])

    with open("hvg_indices.json", "w") as f:
        json.dump(sorted_indices, f)

    idx_to_name = get_gene_index_name_dict("data/pancreas_train.h5ad")
    hvg_list = sorted([idx_to_name[i] for i in sorted_indices])

    with open("data/hvg_list.json", "w") as f:
        json.dump(hvg_list, f)
    
    log(INFO, "")
    log(INFO, "############################## Final HVG Output ##############################")
    log(INFO, "")
    log(INFO, f"Top 2000 Highly Variable Genes (indices): {top_2000_indices}")
    log(INFO, "")
