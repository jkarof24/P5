# Full Heterogeneous Graphs + GNN Benchmark

Build the **full-size** PyPI / DataCo / SupplyGraph heterogeneous graphs from the
raw inputs, then train and evaluate the HGNN benchmark models on them.

## Run it all (one command)

```bash
bash run_all.sh
```

This does everything end to end:

1. **Build** the three full graphs from raw data into `full_graphs/`.
2. **Stats** — print nodes / edges-per-type / feature dims to `stats_report.txt`.
3. **Train / test** — run all 5 models over several seeds on the full graphs
   (mini-batch SGD, so even the huge PyPI graph trains in memory).
4. **Consolidate** — merge per-seed CSVs into one benchmark table
   (`model_benchmarks_results.csv`).

To tune the training step:

```bash
cd code/Gnnpipeline
python run_sweep.py --datasets pypi dataco supplygraph \
    --seeds 0 1 2 --epochs 2 --hidden 64 --logdir ../results   # train only
python consolidate_results.py ../results                      # merge into table
```

## Layout

```
.
├── requirements.txt                 # all dependencies (one place)
├── README.md                        # this file
├── run_all.sh                       # one-command: build → stats → train → consolidate
├── full_graphs/                     # built HeteroData graphs (~4 GB, git-ignored)
├── data/                            # raw inputs, one subfolder per dataset
│   ├── pypi/                        #   features/, edges/, versions/, projects/, advisories/
│   ├── dataco/                      #   DataCoSupplyChainDataset.csv (UCI Retail)
│   └── supplygraph/                 #   Homogenoeus/ (Nodes/, Edges/, Temporal Data/)
├── code/
│   ├── Builtgraph/                  #   build_pypi_hetero_graph.py, build_dataco_graph.py,
│   │                                  #   build_supplygraph_graph.py   (the graph builders)
│   ├── Gnnpipeline/                 #   pipeline/ (config, runner, training, metrics, models)
│   │                                    #   run_sweep.py, consolidate_results.py   (the trainer)
│   └── ETA/                         #   generate_stats.py               (dataset-size stats)
└── results/                         # per-dataset benchmark CSVs + consolidated table
```

`run_all.sh` is the only entry point; the scripts under `code/` are the actual
build / train / stats implementations it calls.

## Datasets

- **PyPI** — 3 node types: `PackageVersion` (feat 6, **target**), `Package` (feat 3),
  `Repository` (feat 3). Edges: `version_of`, `depends_on` (PackageVersion→Package),
  `hosted_on` (PackageVersion→Repository) plus their reverses. Target is a binary
  advisory/vulnerability label on `PackageVersion`, split temporally (train/val/test).
  The full build does its joins in DuckDB and is the heavy one (30–50 GB RAM).
- **DataCo** (UCI Retail) — 3 node types: `Customer` (feat 5), `Order` (feat 4,
  **target**), `Product` (feat 6). Edges: `places` (Customer→Order), `contains`
  (Order→Product). Target: `Order` `SUSPECTED_FRAUD` (minority class) vs other status.
- **SupplyGraph** — facility nodes typed by product **sub-group** (`AT`, `ATN`, … `SOS`,
  19 types, feat 16 each). Edges: `connected_in_plant`, `same_product_group`,
  `same_product_subgroup`, `connected_in_storage`. Target: binary by product category.

Exact node/edge counts are printed by step 2 (`generate_stats.py`) — they depend on
what raw data is shipped, so read `stats_report.txt` rather than hardcoding them.

## Models

All five are **structural-only** 2-round gated heterogeneous GNNs that carry the
*neighbour's* feature through message passing; they differ only in the per-edge
transform. Training is shared: class-balanced BCE + cosine LR with warmup, best
checkpoint logged per run.

- **BaselineHeteroGNN** — gated GCN / GATv2-style attention (reference).
- **SpikingHAN** — gated message passing + leaky-integrator + threshold spike.
- **CausalHGNN** — residual causal GCN, no attention reweighting.
- **RelGT** — one learnable projection per edge type.
- **MiniTransformer** — softmax-normalised attention over neighbour messages.

## Requirements

- Python 3.11+, ≥64 GB RAM (full PyPI build peaks at 30–50 GB during the edge join).
- A GPU is optional — feature tensors are placed on CUDA when built on one, so the
  cached graphs drop straight into the trainer; CPU-only torch also works.
- Disk: ~4 GB inputs + a few GB for the cached graphs.
