#!/usr/bin/env bash
# One-command end-to-end pipeline (KISS).
#
#   1) activate .venv-p5 and install requirements.txt
#   2) build the three FULL heterogeneous graphs from raw data
#   3) print dataset size stats
#   4) train + evaluate all 5 GNN models on the full graphs
#   5) consolidate the per-seed results into one benchmark table
#
# Requires: Python 3.13 + >=64 GB RAM (the full PyPI build peaks at 30-50 GB).
# Optional: a GPU (cached graphs are feature-stored on CUDA when built on one).
set -euo pipefail
cd "$(dirname "$0")"

# Use the prebuilt .venv-p5 (Python 3.13, GPU-capable torch).
echo "==> [1/5] Activating .venv-p5 + installing requirements.txt"
source .venv-p5/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "==> [2/5] Building full graphs into full_graphs/"
python code/Builtgraph/build_pypi_hetero_graph.py --edge_shards 444063 --out full_graphs/pypi_hetero_graph_full.pt
python code/Builtgraph/build_dataco_graph.py --max-nodes 2147483647 --out full_graphs/dataco_hetero_graph_full.pt --raw data/dataco/DataCoSupplyChainDataset.csv
python code/Builtgraph/build_supplygraph_graph.py --out full_graphs/supplygraph_hetero_graph_full.pt

echo "==> [3/5] Dataset size stats -> stats_report.txt"
python code/ETA/generate_stats.py --pypi --dataco --supplygraph > stats_report.txt

echo "==> [4/5] Training + evaluating 5 GNN models on the full graphs"
cd code/Gnnpipeline
python run_sweep.py --build-full --datasets pypi dataco supplygraph \
    --seeds 0 1 2 --epochs 2 --hidden 64 --logdir ../results

echo "==> [5/5] Consolidating results into one benchmark table"
python consolidate_results.py ../results

echo "DONE. Benchmark table: results/model_benchmarks_results.csv"
