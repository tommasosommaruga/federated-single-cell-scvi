# Federated scVI (Flower)

Thesis project on **federated learning for single-cell variational inference (scVI)** using [Flower](https://flower.ai/) and [scvi-tools](https://scvi-tools.org/).

The codebase compares three training setups on pancreas scRNA-seq data:

1. **Centralized** — one model trained on the full training set  
2. **Independent clients** — separate local models, no weight sharing  
3. **Federated (FedAvg)** — clients train locally and aggregate on a server via Flower  

It also includes a **federated highly variable gene (HVG)** selection app so clients can agree on a shared gene list without sharing raw counts.

**Author:** Tommaso Sommaruga (`tommy.sommaruga@gmail.com`)  
**License:** [CC BY 4.0](LICENSE) — free to use; **attribution / citation required** (see [Citation](#citation)).

---

## Overview

| Component | Path | Role |
|-----------|------|------|
| Data prep | `0_data/` | Split pancreas data, HVG & batch lists |
| Centralized scVI | `1_centralised/centralised.py` | Train / evaluate one global model |
| Independent clients | `2_independent/independent_client_scvi.py` | Train one local model per partition |
| Federated scVI | `federated_scvi_flower/` | Flower FedAvg simulation |
| Federated HVG | `HighlyVariableGenes/` | Federated HVG selection (SecAgg+) |
| Plots & metrics | `plotting_functions/`, `umap/` | Loss curves, UMAP, clustering metrics |
| Logs / models | `loss_logs/`, `models/` | Saved losses and trained models |

**Dataset:** pancreas AnnData from Figshare  
[https://figshare.com/ndownloader/files/24539828](https://figshare.com/ndownloader/files/24539828)

**Train / test split:** cells from `smartseq2` and `celseq2` form the held-out **test** set; remaining technologies form **train**. This probes generalization across sequencing technologies.

---

## Requirements

- Python **3.11** (tested with `3.11.13`)
- Conda / Mamba recommended
- Flower simulation (`flwr`), `scvi-tools`, `scanpy`, PyTorch (CPU or CUDA)

### Option A — Conda environment (recommended)

```bash
conda env create -f environment.yaml
conda activate flwr-env-cpu
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "flwr[simulation]"
```

For CUDA builds, follow comments in `requirements.txt` (PyTorch cu124 index, etc.). Scripts currently force CPU via `CUDA_VISIBLE_DEVICES=""`.

---

## Project layout

```
SCVI/
├── 0_data/                    # Data scripts & JSON metadata (h5ad files local / gitignored)
├── 1_centralised/             # Centralized scVI baseline
├── 2_independent/             # Per-client independent scVI
├── federated_scvi_flower/     # Federated scVI (Flower app)
│   └── app/
│       ├── client_scvi.py
│       ├── server_scvi.py
│       ├── helper_py_files/   # create_hvg_list, create_batch_list, …
│       └── utils/             # data_utils_scvi, model_utils_scvi
├── HighlyVariableGenes/       # Federated HVG Flower app
├── plotting_functions/        # Loss & UMAP helpers
├── loss_logs/                 # CSV loss curves
├── models/                    # Saved scVI models
├── umap/                      # UMAP figures & clustering notebook
├── environment.yaml
├── requirements.txt
├── LICENSE                    # CC BY 4.0
└── CITATION.cff
```

Large `.h5ad` files are typically not committed; download and place them under `0_data/` as described below.

---

## Quick start

All commands below assume the **repository root** as the working directory, unless noted.

### 1. Download and split data

```bash
# Download into 0_data/ (or let Scanpy backup_url fetch it where used)
mkdir -p 0_data
# Save as 0_data/pancreas.h5ad from:
# https://figshare.com/ndownloader/files/24539828

python 0_data/split_data.py
```

This writes `pancreas_train.h5ad` / `pancreas_test.h5ad` (and related splits depending on `split_mode` in the script).

### 2. Build shared HVG and batch lists

```bash
cd federated_scvi_flower/app/helper_py_files
python create_hvg_list.py      # → 0_data/hvg_list.json (top 2000 HVGs, batch_key=tech)
python create_batch_list.py    # → 0_data/batch_list.json (all tech categories)
cd ../../..
```

Optional: run the federated HVG app (see [Federated HVG](#federated-hvg-selection)) and use `0_data/hvg_list_from_fl.json` if you prefer a federated gene list.

### 3. Centralized baseline

```bash
python 1_centralised/centralised.py
```

- Trains (or loads) `models/centralised_model/`
- Logs losses to `loss_logs/loss_curve_centralised.csv`
- Writes UMAPs under `umap/centralised/`

### 4. Independent client baselines

Edit `partition_id` / `num_partitions` at the top of `2_independent/independent_client_scvi.py`, then:

```bash
python 2_independent/independent_client_scvi.py
```

Repeat for each client id (e.g. `0 … 6` for 7 partitions). Models land in `models/{N}_cl_independent_client_{id}/`.

### 5. Federated scVI (Flower simulation)

Configure clients, rounds, and local epochs in `federated_scvi_flower/pyproject.toml`:

```toml
[tool.flwr.app.config]
num_clients = 7
num_rounds = 20
local-epochs = 5
hvg_list_path = "0_data/hvg_list.json"
batch_list_path = "0_data/batch_list.json"
adata_test_path = "0_data/pancreas_test.h5ad"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 7   # must match num_clients
```

Run from the **app directory**:

```bash
cd federated_scvi_flower
flwr run .
```

Or override config on the CLI, for example:

```bash
flwr run . --run-config "num_clients=3 num_rounds=10 local-epochs=10"
```

(Ensure `options.num-supernodes` matches `num_clients`.)

Federated test loss is appended under `loss_logs/` as:

`loss_curve_federated_{clients}_clients_{rounds}_rounds_{epochs}_epochs.csv`

### 6. Federated HVG selection

```bash
cd HighlyVariableGenes
flwr run .
```

See `HighlyVariableGenes/README.md` and `comparison.ipynb` for federated vs centralized HVG comparison. Aggregation uses **FedAvg** with **SecAgg+**.

### 7. Plots and evaluation

```bash
# Loss curves
python plotting_functions/create_curve_plot.py

# UMAP helpers
python plotting_functions/plot_umap.py
python plotting_functions/plot_umap_fed.py
python plotting_functions/umap_comparison_report.py
```

Clustering metrics (ARI, NMI, silhouette, etc.): open  
`umap/latent_space_clustering_metrics.ipynb`.

---

## How federation works (short)

1. **Shared feature space** — all clients subset to the same HVG list and use a common batch category list (`tech` → `batch`).
2. **Raw counts** — scVI expects unnormalized counts (e.g. `layers["counts"]`); do not normalize externally for the model.
3. **Clients** — each partition loads local train data, trains for `local-epochs`, returns scVI weights.
4. **Server** — FedAvg aggregates weights; evaluates the global model on the held-out test set each round.

Design notes on HVG / `setup_anndata` coordination: `federated_proposal.md`.

---

## Reproducing thesis-style experiments

Typical comparisons in the write-up:

| Experiment | What to run |
|------------|-------------|
| Centralized vs federated loss | `1_centralised` + several `flwr run` configs (vary rounds × local epochs) |
| Federated vs independent clients | Federated run + all `2_independent` partitions |
| Cross-tech generalization | Test set = `smartseq2` + `celseq2` (default split) |
| HVG privacy-preserving selection | `HighlyVariableGenes` + `comparison.ipynb` |
| Latent space quality | UMAP folders + clustering metrics notebook |

Existing CSVs in `loss_logs/` and models under `models/` correspond to prior runs (3/7 clients, various round/epoch budgets).

Outline / notes: `thesis.md`, `thesis_part_3.md`.

---

## Configuration tips

- Keep **`num_clients` == `options.num-supernodes`** in Flower apps.
- Use the **same** `hvg_list.json` and `batch_list.json` for centralized, independent, and federated runs so comparisons are fair.
- Seeds: scVI / scripts use fixed seeds (`scvi.settings.seed = 0`, federated apps use `SEED = 55`) for reproducibility.
- Paths in Flower config are relative to where `flwr run` is launched; prefer running from `federated_scvi_flower/` with paths like `../0_data/...` if needed, or run from repo root if your Flower version resolves project root paths as in `pyproject.toml`.

---

## Citation

This project is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
You may use, modify, and redistribute the work **as long as you credit the author**.

### Preferred citation

> Tommaso Sommaruga. *Federated Single-Cell Variational Inference (scVI) with Flower*. Thesis code repository, 2025.  
> https://github.com/tommasosommaruga/SCVI

### BibTeX

```bibtex
@software{sommaruga_federated_scvi_2025,
  author       = {Sommaruga, Tommaso},
  title        = {Federated Single-Cell Variational Inference (scVI) with Flower},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/tommasosommaruga/SCVI},
  note         = {Thesis code; licensed under CC BY 4.0}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff). Full license terms: [`LICENSE`](LICENSE).

### Third-party software

This work builds on open-source tools including **Flower**, **scvi-tools**, **Scanpy**, and **PyTorch**. Cite those projects as well when appropriate. The pancreas dataset should be credited according to its original publication / Figshare record.

---

## Acknowledgments

Federated HVG Flower app publisher credit in `HighlyVariableGenes/pyproject.toml` includes co-author **dmalpetti**; federated scVI app publisher: **tommaso-sommaruga**.

---

## Disclaimer

Research / thesis code: useful for reproduction and extension, not a production FL deployment. Paths, GPU settings, and Flower config may need adjustment for your machine.
