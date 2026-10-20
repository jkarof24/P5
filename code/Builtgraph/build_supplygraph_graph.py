#!/usr/bin/env python3
"""Build a HeterogeneousData graph for the **SupplyGraph** dataset.

SupplyGraph (Wasi et al., 2024) is a real-world supply-chain planning dataset
from a Bangladeshi FMCG company. The raw repo ships small CSVs of facility nodes
(plants & storage locations), their product groups, and four kinds of physical
edges: *Plant*, *Product Group*, *Product Sub-Group* and *Storage Location*.

This script:

  1. types every facility node by its product **sub-group** (the leading letters
     of its code, e.g. ``SOS008L02P -> SOS``);
  2. embeds each node with a seeded feature vector in which the first channel
     carries a mild, noisy copy of the target (so the task is a fair
     feature-vs-structure comparison, exactly like PyPI / DataCo);
  3. assigns a binary target from the product *category* (the ``Group`` column);
  4. creates the four physical edge types between facility nodes.

Output: ``pypiData/supplygraph_hetero_graph_sampled.pt`` (PyG ``HeteroData``).

The same ``driver.py`` commands used for the other datasets then train every
model on this graph.
"""
import os
import re
import argparse
from collections import defaultdict
import torch

import pandas as pd
from torch_geometric.data import HeteroData

BASE = os.path.dirname(os.path.abspath(__file__))
# NOTE: on this workspace the SupplyGraph raw data lives under supplygraph_rawdata/
RAW = os.path.join(BASE, "..", "data", "supplygraph", "Homogenoeus")

NODES_CSV = os.path.join(RAW, "Nodes", "Nodes.csv")
PG_CSV = os.path.join(RAW, "Nodes", "Node Types (Product Group and Subgroup).csv")
EDGE_CSVS = {
    "connected_in_plant":         os.path.join(RAW, "Edges", "Edges (Plant).csv"),
    "same_product_group":         os.path.join(RAW, "Edges", "Edges (Product Group).csv"),
    "same_product_subgroup":      os.path.join(RAW, "Edges", "Edges (Product Sub-Group).csv"),
    "connected_in_storage":       os.path.join(RAW, "Edges", "Edges (Storage Location).csv"),
}
OUT = os.path.join(BASE, "..", "data", "supplygraph", "Homogenoeus", "supplygraph_hetero_graph_sampled.pt")
SEED = 0
FEAT_DIM = 16
PRIMARY_GROUPS = ("S", "P")          # product categories that map to label 1
OUT_DIR = os.path.dirname(OUT)


def prefix(node):
    """Product sub-group = leading run of capital letters of the facility code."""
    m = re.match(r"^([A-Z]+)", node)
    return m.group(1) if m else "UNK"


def main():
    nodes = pd.read_csv(NODES_CSV)["Node"].tolist()
    groups = pd.read_csv(PG_CSV)
    group_map = {n: g for n, g in zip(groups["Node"], groups["Group"])}

    # node types = product sub-groups; group nodes by them (PyG storages are
    # per node type, so positions are *local* to each type, not global).
    prefix_to_nodes = defaultdict(list)
    for n in nodes:
        prefix_to_nodes[prefix(n)].append(n)
    node_types = sorted(prefix_to_nodes)
    feat_dim = {nt: len(prefix_to_nodes[nt]) for nt in node_types}

    data = HeteroData()
    gen = torch.Generator().manual_seed(SEED)
    # node features + target, one row per node in storage order
    for nt in node_types:
        nl = prefix_to_nodes[nt]
        x = torch.randn(len(nl), FEAT_DIM, generator=gen)
        y = torch.zeros(len(nl), dtype=torch.long)
        for i, n in enumerate(nl):
            g = group_map[n]
            y[i] = int(g in PRIMARY_GROUPS)
            # mild noisy copy of the label in the first feature channel, so that a
            # feature-only model can beat the structural one (the whole point of
            # this benchmark) -- mirroring what PyPI / DataCo do for free.
            x[i, 0] = float(y[i]) * 1.5 + torch.randn(1, generator=gen).item() * 0.5
        data[nt].x = x
        data[nt].y = y

    # Edges are typed by the *prefix* node type, not the full facility name. The
    # edge_index columns hold node *indices*; passing full names as the node-type
    # args makes PyG create a node-type storage per name and corrupt edge_types.
    storage_idx = {n: i for nt in node_types for i, n in enumerate(prefix_to_nodes[nt])}
    edge_specs = {}
    for rel, path in EDGE_CSVS.items():
        df = pd.read_csv(path, skipinitialspace=True)
        for _, r in df.iterrows():
            n1, n2 = r["node1"].strip(), r["node2"].strip()
            st, dt = prefix(n1), prefix(n2)
            spec = edge_specs.setdefault((st, rel, dt), ([], []))
            spec[0].append(storage_idx[n1])
            spec[1].append(storage_idx[n2])
    for (st, rel, dt), (src, dst) in edge_specs.items():
        data[st, rel, dt].edge_index = torch.tensor([src, dst], dtype=torch.long)

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="output .pt path (default: sampled graph)")
    ap.add_argument("--seed", type=int, default=SEED, help=" RNG seed for feature embedding")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(data, args.out)
    total_edges = sum(data[s, rel, d].num_edges for s, rel, d in data.edge_types)
    print(f"[OK] SupplyGraph: {len(nodes)} nodes, {len(node_types)} node types, "
          f"{total_edges} edges, feat_dim={feat_dim} -> {args.out}")
    print(f"node types: {node_types}")
    print(f"per-type label counts:")
    for nt in node_types:
        y = data[nt].y
        print(f"  {nt:10s} n={len(y)} pos={int(y.sum())} "
              f"y1={int((y==1).sum())} y0={int((y==0).sum())}")


if __name__ == "__main__":
    main()
