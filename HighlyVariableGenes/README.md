# Highly Variable Genes: A Flower for highly variable genes detection.

## Introduction

This example revises the previous weakly expressed genes analysis, adapting it to detect highly variable genes (HVGs). The resulting dataset, filtered for HVGs, serves as a base for subsequent federated analyses. The workflow demonstrates how to identify HVGs in a federated setting using Flower, ensuring that the filtered dataset is suitable for further collaborative studies.

We recommend running the federated simulation first. After completion, results can be analyzed using the provided Jupyter notebook, which compares federated and centralized analyses to confirm consistency. The notebook also contrasts federated results with local analyses, illustrating what each client would obtain independently, without data sharing.

Federation is simulated using Flower’s engine, based on the **FedAvg** aggregation strategy and secure aggregation via **SecAgg+**. For more details, see [Flower's documentation](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).

## Run the federated simulation with the Simulation Engine

In the `HighlyVariableGenes` directory, use `flwr run` to start a local simulation of the federated process:

```bash
flwr run .
```
