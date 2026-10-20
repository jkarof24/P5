#!/usr/bin/env python3
"""Generic HGNN training driver.

Runs any (graph, model) pair through the shared harness:

    python driver.py <graph.pt> --model <kind> --dataset <alias> \
        --target <nt> --epochs N --lr L --hidden H [--seed S]

The result is logged to ``results/*.csv`` and collapsed to the single best
result per (Dataset, Model) by ``consolidate_results.py``.
"""
import argparse
import sys

import torch

from pipeline import runner, training


def keep_connected(data):
    """Subsample every node type down to nodes that appear in some edge.

    Some graphs store millions of isolated nodes (e.g. PyPI's Repository type)
    that message passing never touches -- subsampling them keeps training fast.
    Each node type is reindexed *locally* (0..k-1 over its own kept nodes);
    edge endpoints are remapped with the matching per-type index map.  Note that
    PyG reshapes a 1-D assignment into ``(2, E)`` so we must write a proper 2-D
    ``edge_index`` (src row / dst row) instead.
    """
    keep_set = set()
    for et in data.edge_types:
        ei = torch.as_tensor(data[et[0], et[1], et[2]].edge_index)
        keep_set.update(ei[0].tolist())
        keep_set.update(ei[1].tolist())

    # per-node-type local reindex: old position -> new local position
    node_reindex = {}
    for nt in data.node_types:
        n = data[nt].num_nodes
        local = [i for i in range(n) if i in keep_set]
        node_reindex[nt] = {old: i for i, old in enumerate(local)}

    for nt in data.node_types:
        storage = data[nt]
        n = storage.num_nodes
        keep_idx = torch.tensor(list(node_reindex[nt].values()), dtype=torch.long)
        storage.x = storage.x[keep_idx]
        if storage.get("y", None) is not None:
            storage.y = torch.as_tensor(storage.y)[keep_idx]
        storage.num_nodes = len(keep_idx)

    for et in data.edge_types:
        s, r, d = et
        ei = torch.as_tensor(data[s, r, d].edge_index)
        if ei.numel() == 0:
            continue
        # Drop dangling edges whose endpoints are not valid node indices for
        # their type (some builders emit edges that reference node ids which
        # were later dropped -- e.g. PyPI's depends_on edges referencing package
        # ids beyond the retained set). This is a no-op for clean graphs.
        src_valid = torch.tensor([o in node_reindex[s] for o in ei[0].tolist()], dtype=torch.bool)
        dst_valid = torch.tensor([o in node_reindex[d] for o in ei[1].tolist()], dtype=torch.bool)
        keep = src_valid & dst_valid
        if not keep.any():
            continue
        src = torch.tensor([node_reindex[s][o] for o in ei[0][keep].tolist()], dtype=torch.long)
        dst = torch.tensor([node_reindex[d][o] for o in ei[1][keep].tolist()], dtype=torch.long)
        data[s, r, d].edge_index = torch.stack([src, dst], dim=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph")
    p.add_argument("--model", default="BaselineHeteroGNN")
    p.add_argument("--dataset", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=0, help="linear warmup epochs before cosine decay")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-loop", action=argparse.BooleanOptionalAction,
                   default=True, help="fold each node's own features into the "
                   "message (self-loop); True by default")
    a = p.parse_args()

    cfg = None
    if a.dataset:
        from pipeline.config import dataset as ds_cfg
        cfg = ds_cfg(a.dataset)
    target = (a.target or cfg["target_node"]) if cfg else None
    if target is None:
        print("error: --target or --dataset required", file=sys.stderr)
        sys.exit(1)

    data = torch.load(a.graph, map_location="cpu", weights_only=False)
    keep_connected(data)
    feat_dim = {nt: data[nt].x.shape[-1] for nt in data.node_types}

    args = type("Args", (), {
        "model": a.model, "target": target, "epochs": a.epochs, "lr": a.lr,
        "warmup": a.warmup, "weight_decay": a.weight_decay,
        "hidden": a.hidden, "seed": a.seed,
    })()

    def build_fn(d):
        from pipeline.models.hgnn import build_fn as hf
        fwd, seen = [], set()
        for et in d.edge_types:
            # skip edges with no endpoints -- they yield empty idx_src/idx_dst
            if d[et[0], et[1], et[2]].num_edges == 0:
                continue
            key = (et[0], et[1], et[2])
            if key not in seen:
                fwd.append(key); seen.add(key)
        return hf(fwd, feat_dim, a.hidden, 2, kind=a.model, use_self_loop=a.self_loop)

    def train_eval_fn(d, m):
        return training.train_eval_fn(d, m, args)

    res = runner.run(
        dataset_name=a.dataset or a.model,
        model_name=a.model,
        data=data,
        build_fn=build_fn,
        train_eval_fn=train_eval_fn,
        verbose=True,
    )
    print(f"[RESULT] {a.dataset or a.model} | {a.model} | AUC {res.roc_auc:.3f} "
          f"F1 {res.f1:.3f} (test pos {res.n_support})", file=sys.stderr)
    return res.roc_auc


if __name__ == "__main__":
    main()
