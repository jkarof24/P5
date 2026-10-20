# GNN Benchmark — Heterogeneous Graphs

Trains and evaluates the **structural-only Heterogeneous GNN** benchmark across
three datasets (PyPI, DataCo, SupplyGraph) on the **full** graphs.

Trains five structural HGNN variants over several seeds and epochs, then
consolidates the results into one benchmark table:

```
Dataset, Model, Precision, Recall, False Positives, False Negatives, F1-Score, ROC-AUC
```

## Layout
```
Gnnpipeline/
├── experiment.yaml                # single source of truth: models, datasets, seeds, hyper-params
├── run_sweep.py                   # grid sweep: models x datasets x seeds x epochs
├── consolidate_results.py         # best ROC-AUC per (Dataset, Model) -> benchmark CSV
├── README.md                      # this file
└── pipeline/                      # shared training harness
    ├── config.py                  # dataset registry + model list
    ├── driver.py                  # graph load + build/train callbacks
    ├── runner.py                  # uniform train+eval+log wrapper
    ├── training.py                # per-epoch trainer (cosine LR + warmup)
    ├── metrics.py                 # standardized metrics + CSV logging
    └── models/hgnn.py             # the 5 structural HGNN models
```

## Design notes

* **Structural-only message passing.** Every model carries the *source (neighbour)*
  feature through message passing (the GCN message is the neighbour's
  representation, not the destination's own). Honest structural-only baseline.
* **GPU-aware.** `pipeline/models/hgnn.py` moves all forward-pass tensors onto
  the parameters' device, so the same code runs on CPU or CUDA.
* **Full graphs by default.** Training runs on the full graphs built by
  `code/Builtgraph/`. Mini-batch SGD is used automatically (see below) so even
  the full PyPI graph trains in memory while touching every node/edge. Exact
  node/edge sizes depend on the shipped raw data — see `stats_report.txt`.

## What to run (full graphs)

```bash
cd code/Gnnpipeline
python run_sweep.py --build-full --datasets pypi dataco supplygraph \
    --seeds 0 1 2 --epochs 2 --hidden 64 --logdir results
python consolidate_results.py results/
```
`--build-full` builds each dataset's full graph if missing (cached in
`../../full_graphs/`). `--batch-size N` overrides the auto mini-batch size
(0 = full-batch; only used for small graphs).

## Config & reproducibility

* **`experiment.yaml`** — one source of truth for the grid and hyper-params
  (models, datasets, seeds, epochs, lr, warmup, weight-decay, hidden, self-loop,
  global seed, logdir). `run_sweep.py` loads it; CLI flags win over YAML. It is
  optional: if missing, `run_sweep.py` falls back to built-in defaults.
* **Deps** — all pinned Python dependencies live in the **root**
  `../requirements.txt` (numpy, scikit-learn, PyYAML, torch, torch_geometric).
  One `requirements.txt` for the whole project; install it before training:

    pip install -r ../requirements.txt

## Mini-batch training (why it fits in memory)

Full-batch training must hold a message per edge in memory — infeasible on the
full PyPI graph (tens of millions of edges) even on a large GPU. The trainer
therefore uses
**mini-batch SGD**:

* `pipeline/training.py` builds a column-sorted CSR map once
  (`precompute_edge_csrs`) and induces a small `HeteroData` subgraph per batch
  (`build_mini_batch`), dropping dangling edges and reindexing every node type
  locally — the same discipline `keep_connected` applies.
* Every node is sampled each epoch, so **all** data is used; only a batch's
  incident edges are materialised per step.
