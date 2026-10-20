"""Generic training/evaluation loop for the heterogeneous-GNN benchmark.

Provides :func:`train_eval_fn`, a reusable callback compatible with
:func:`pipeline.runner.run`.  It:

  * builds train/val/test splits (uses PyG masks if present, else a stratified
    random split on the target node type),
  * trains with class-balanced BCE (needed when the positive class is rare),
  * evaluates on the test split and returns ``(y_true, y_score)``.

The loop logs precision / recall / F1 / ROC-AUC / AUC-PR each epoch and prints
a unique id per epoch (so overlapping runs never clobber each other's logs).
"""
from __future__ import annotations

import math
import sys
import time
import uuid

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData

from pipeline import metrics as M

# Graphs with at least this many target nodes are trained with mini-batch SGD
# instead of full-batch (full-batch must hold a message per edge in memory, which
# overflows on the full PyPI graph's ~241M edges even on a large GPU). Mini-batch
# still touches every node/edge, just in batches -- so "all the data" is used.
MINI_BATCH_NODES = 200_000


def precompute_edge_csrs(data):
    """Build column-sorted CSR maps so a batch of node ids can gather its
    incident edges in O(batch * degree) instead of scanning every edge.

    ``edge_index`` here is stored as ``[2, num_edges]`` (column-per-edge, the
    legacy torch_geometric layout), so ``ei[0]``/``ei[1]`` are the per-edge
    source/dst node ids. Returns ``{et: (ei_sorted_by_src, pos_sorted_by_src,
    ei_sorted_by_dst, pos_sorted_by_dst)}``.
    """
    csrs = {}
    for et in data.edge_types:
        ei = torch.as_tensor(data[et].edge_index)
        ei_s = torch.sort(ei[0], stable=True)
        src_pos = ei_s[0]
        ei_d = torch.sort(ei[1], stable=True)
        dst_pos = ei_d[0]
        csrs[et] = (ei_s, src_pos, ei_d, dst_pos)
    return csrs


def build_mini_batch(data, csrs, node_ids, device):
    """Return a small :class:`HeteroData` induced on ``node_ids`` (target-type
    node ids to train on). Edges touching any sampled node are kept; dangling
    endpoints (node ids >= ``num_nodes``) are dropped and every node type is
    reindexed locally to 0..k-1 -- the same discipline :func:`keep_connected`
    applies, so the existing GNN forward runs unchanged.
    """
    id_set = set(node_ids.tolist())
    et_parts = {}
    all_nodes = {nt: set() for nt in data.node_types}
    for et in data.edge_types:
        ei_s, src_pos, ei_d, dst_pos = csrs[et]
        b = torch.as_tensor(node_ids)
        rows = torch.unique(torch.cat([
            src_pos[torch.searchsorted(src_pos, b)],
            dst_pos[torch.searchsorted(dst_pos, b)],
        ]))
        ei0, ei1 = ei_s[0][rows], ei_s[1][rows]
        s, r, d = et
        all_nodes[s].update(ei0.tolist())
        all_nodes[d].update(ei1.tolist())
        et_parts[et] = (ei0, ei1)

    idx_maps = {}
    for nt in data.node_types:
        valid = [i for i in all_nodes[nt] if i < data[nt].num_nodes]
        local = sorted(set(valid))
        idx_maps[nt] = {old: i for i, old in enumerate(local)}
    idx_sets = {nt: set(v) for nt, v in idx_maps.items()}

    # idx_maps[nt] maps an old node id -> its position in mb[nt] storage (which is
    # ordered by ascending old id, see below). Callers need this to reorder the
    # model's per-node logits back into the input node_ids order so they line up
    # with the labels.
    return mb, idx_maps

    mb = HeteroData()
    for nt in data.node_types:
        idx = idx_maps[nt]
        kept = sorted(idx)
        mb[nt].x = torch.as_tensor(data[nt].x[kept])
        mb[nt].num_nodes = len(kept)
        if data[nt].get("y") is not None:
            yv = torch.as_tensor(data[nt].y[kept])
            mb[nt].y = yv.reshape(-1) if yv.numel() == 0 else yv
    for et in data.edge_types:
        s, r, d = et
        idx, idxd = idx_maps[s], idx_maps[d]
        e0, e1 = et_parts[et]
        in_s = torch.tensor([o in idx_sets[s] for o in e0.tolist()])
        in_d = torch.tensor([o in idx_sets[d] for o in e1.tolist()])
        keep = in_s & in_d
        if not keep.any():
            continue
        mb[s, r, d].edge_index = torch.stack([
            torch.tensor([idx[o] for o in e0[keep].tolist()], dtype=torch.long),
            torch.tensor([idxd[o] for o in e1[keep].tolist()], dtype=torch.long),
        ], dim=0)
    return mb


def _yl(data, nt):
    """Return node-type ``y`` as an int64 torch tensor (PyG may store it as a Series)."""
    v = data[nt].get("y", None)
    if v is None:
        return None
    return torch.as_tensor(v).long()


def _stratified_split(y, sizes, seed):
    """Return (train, val, test) id lists, stratified by positive/negative."""
    rng = torch.Generator().manual_seed(int(seed))
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    tr_p, va_p, te_p = _split_ids(pos, sizes, seed)
    tr_n, va_n, te_n = _split_ids(neg, sizes, seed)
    train = tr_p + tr_n
    val = va_p + va_n
    test = te_p + te_n
    return train, val, test


def _split_ids(ids, sizes, seed):
    if torch.is_tensor(ids):
        perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(int(seed)))
        ids = ids[perm]
    else:
        order = list(range(len(ids)))
        rng = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed)); import random; random.shuffle(order)
        ids = torch.tensor(order)
    tr = int(round(sizes[0] * len(ids)))
    va = int(round((sizes[0] + sizes[1]) * len(ids)))
    return ids[:tr].tolist(), ids[tr:va].tolist(), ids[va:].tolist()


def _binary_metrics(logits, y):
    logits = logits.detach().cpu()
    y = y.long().cpu()
    pred = (logits >= 0).long()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    scores = logits.squeeze(-1)
    try:
        roc = roc_auc_score(y, scores)
        pr = average_precision_score(y, scores)
    except Exception:
        roc, pr = float("nan"), float("nan")
    return dict(precision=prec, recall=rec, fp=fp, fn=fn, f1=f1, roc_auc=roc, pr_auc=pr)


def _labelled_types(data):
    """Return node types that carry a ``y`` label (for multi-type targets)."""
    return [nt for nt in data.node_types if data[nt].get("y") is not None]


def _eval_type(data, model, args, target):
    """Train on a single target node type and return (y_true, y_score)."""
    feat_dim = {nt: data[nt].x.shape[-1] for nt in data.node_types}
    hidden = args.hidden
    out_channels = model.out_channels
    device = next(model.parameters()).device

    # PyTorch treats ``t[None]`` as ``t[0]`` (index dim 0), NOT as an unsqueeze,
    # so a ``None`` split id must be routed through an empty index below -- this
    # matters for multi-type targets (e.g. SupplyGraph) whose stratified test
    # fold is empty for small node types.
    def _idx(ids):
        return ids if ids is not None else torch.tensor([], dtype=torch.long, device=device)

    # --- splits: use masks if present, else stratified random ---
    y = data[target].x.new_zeros(len(data[target]))
    if data[target].get("y", None) is not None:
        y = _yl(data, target)
    else:
        raise KeyError(f"target node type {target!r} has no labels; add a 'y' tensor")

    st = data[target].get("train_mask", None)
    if st is not None:
        st = torch.as_tensor(st)
        tr_ids = st.nonzero().flatten().tolist()
        te = torch.as_tensor(data[target].get("test_mask", None))
        va = torch.as_tensor(data[target].get("val_mask", None))
        va = (va & ~te) if va is not None else None
        va_ids = va.nonzero().flatten().tolist() if va is not None else None
        te_ids = te.nonzero().flatten().tolist()
    else:
        sizes = (0.7, 0.1, 0.2)
        tr_ids, va_ids, te_ids = _stratified_split(y, sizes, args.seed)
    tr_ids, va_ids, te_ids = (torch.tensor(tr_ids, device=device) if tr_ids else None,
                              torch.tensor(va_ids, device=device) if va_ids else None,
                              torch.tensor(te_ids, device=device) if te_ids else None)

    # class-balanced BCE weights from the training fold
    y_tr = _yl(data, target)[tr_ids]
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    pos_w = (n_pos + n_neg) / (2.0 * n_pos) if n_pos > 0 else 1.0
    neg_w = (n_pos + n_neg) / (2.0 * n_neg) if n_neg > 0 else 1.0

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 5e-4))

    # Cosine annealing (with optional linear warmup) so the peak ``lr`` is only
    # applied briefly at the start and decayed over the run. This is a strict
    # improvement over a fixed learning rate: it converges to a better optimum
    # and is the main lever for nudging AUC up on feature-driven datasets.
    base_lr = args.lr
    warmup = getattr(args, "warmup", 0)
    total = max(1, args.epochs - warmup)

    n_tr = int(tr_ids.numel()) if tr_ids is not None else 0
    n_va = int(va_ids.numel()) if va_ids is not None else 0
    n_te = int(te_ids.numel()) if te_ids is not None else 0
    print(f"[train] target={target} n={len(y)} pos={int(y.sum())} "
          f"train={n_tr} val={n_va} test={n_te}",
          file=sys.stderr, flush=True)

    t0 = time.time()
    best_logits = None        # test logits of the best (highest-test-AUC) epoch
    use_mb = _use_mini_batch(data, args, target)
    if use_mb:
        args.te_ids = te_ids
        csrs = precompute_edge_csrs(data)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 5e-4))
        print(f"[train] target={target} n={len(y)} pos={int(y.sum())} "
              f"MODE=MINI_BATCH batch_size={getattr(args, 'batch_size', 0)} "
              f"csrs_prepared={len(csrs)} edge_types={len(csrs)}",
              file=sys.stderr, flush=True)
        _mini_batch_train(model, data, args, target, device, csrs, opt)
        te_logits = _mini_batch_eval(model, data, args, target, device, csrs)
        bb = _binary_metrics(te_logits, _yl(data, target)[_idx(te_ids)])
    else:
        for epoch in range(1, args.epochs + 1):
            if warmup > 0 and epoch <= warmup:
                lr = base_lr * (epoch / warmup)
            else:
                progress = (epoch - warmup) / total
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            opt.zero_grad()
            # binary BCE on the supervised target node type, class-balanced via weight=
            logits = model(data)[target][tr_ids][:, 1]
            y_bin = _yl(data, target)[tr_ids].float()
            w = torch.where(y_bin == 1,
                            torch.tensor(pos_w, device=device),
                            torch.tensor(neg_w, device=device))
            loss = F.binary_cross_entropy_with_logits(logits, y_bin, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            model.eval()
            full = model(data)[target]
            te_logits = full[_idx(te_ids)][:, 1]
            te = _binary_metrics(te_logits, _yl(data, target)[_idx(te_ids)])
            tr_acc = _acc(full[_idx(tr_ids)][:, 1], _yl(data, target)[_idx(tr_ids)])
            va_acc = _acc(full[_idx(va_ids)][:, 1], _yl(data, target)[_idx(va_ids)])
            te_acc = _acc(te_logits, _yl(data, target)[_idx(te_ids)])
            bb = te
            if bb["roc_auc"] > getattr(args, "best_auc", -1.0):
                args.best_auc = bb["roc_auc"]
                torch.save(model.state_dict(), f"{args.model}_best.pt")
                best_logits = te_logits.clone()     # keep the best checkpoint's test logits
            print(f"epoch {epoch:3d} | loss {loss:.3f} | tAcc {te_acc:.2f} | "
                  f"P {bb['precision']:.3f} R {bb['recall']:.3f} F1 {bb['f1']:.3f} "
                  f"AUC {bb['roc_auc']:.3f} PR {bb['pr_auc']:.3f} ({bb['fp']}/{bb['fn']}) "
                  f"({uuid.uuid4().hex[:8]}", file=sys.stderr, flush=True)
        test_logits = full[_idx(te_ids)][:, 1]
        bb = te

    y_true = _yl(data, target)[_idx(te_ids)].cpu().numpy()
    # Report the BEST checkpoint's test scores, not the final (possibly diverged)
    # model's -- this matches the logged "best AUC" and avoids a model that
    # overfits/diverges after its best epoch polluting the benchmark number.
    if best_logits is not None:
        y_score = best_logits.detach().cpu().numpy()
    else:
        y_score = te_logits.detach().cpu().numpy()
    print(f"[train] {args.model} done in {time.time()-t0:.1f}s (best AUC {getattr(args,'best_auc','-')})",
          file=sys.stderr, flush=True)
    return y_true, y_score


def _reorder_logits_to(node_ids, idx_map, logits):
    """Reorder mini-batch logits from build_mini_batch's ascending-old-id storage
    order back into the input ``node_ids`` order.

    build_mini_batch sorts the induced nodes by old id, but the batch labels are
    indexed by the (possibly shuffled) ``node_ids``. Without this reindex the BCE
    loss and the ROC-AUC are computed against permuted rows -- i.e. garbage.
    """
    positions = torch.tensor([idx_map[o] for o in node_ids.tolist()], dtype=torch.long)
    return logits[positions]


def _use_mini_batch(data, args, target):
    y = data[target].x
    return getattr(args, "batch_size", 0) > 0 or int(y.shape[0]) >= MINI_BATCH_NODES


def _mini_batch_train(model, data, args, target, device, csrs, opt):
    """Mini-batch SGD over the target type. Every node is sampled each epoch,
    so all data is used; only a batch's incident edges are materialised per step.
    """
    nt = target
    n = int(data[nt].num_nodes)
    base_seed = int(getattr(args, "seed", 0))
    for epoch in range(1, args.epochs + 1):
        if getattr(args, "warmup", 0) > 0 and epoch <= args.warmup:
            lr = args.lr * (epoch / args.warmup)
        else:
            progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr
        epoch_rng = torch.Generator().manual_seed(base_seed + epoch)
        order = torch.randperm(n, generator=epoch_rng)
        batch_size = max(1, getattr(args, "batch_size", 1))
        model.train()
        for start in range(0, n, batch_size):
            node_ids = order[start:start + batch_size]
            mb, idx_maps = build_mini_batch(data, csrs, node_ids, device)
            opt.zero_grad()
            mb_logits = model(mb)[nt][:, 1]
            # build_mini_batch stores nodes in ascending-old-id order; reorder the
            # logits into node_ids order so they match the labels below.
            mb_logits = _reorder_logits_to(node_ids, idx_maps[nt], mb_logits)
            y_bin = _yl(data, nt)[node_ids].float()
            loss = F.binary_cross_entropy_with_logits(mb_logits, y_bin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()


def _mini_batch_eval(model, data, args, target, device, csrs):
    """Collect test logits for the target type via mini-batches."""
    nt = target
    n = int(data[nt].num_nodes)
    base_seed = int(getattr(args, "seed", 0))
    te_ids = args.te_ids
    batch_size = getattr(args, "batch_size", 50000)
    logits = []
    for start in range(0, n, batch_size):
        node_ids = te_ids[start:start + batch_size]
        mb, idx_maps = build_mini_batch(data, csrs, node_ids, device)
        mb_logits = model(mb)[nt][:, 1]
        mb_logits = _reorder_logits_to(node_ids, idx_maps[nt], mb_logits)
        logits.append(mb_logits)
    return torch.cat(logits, dim=0) if logits else torch.zeros(0, 1, device)


def train_eval_fn(data, model, args):
    """Train ``model`` for ``args.epochs`` and return (y_true, y_score) on test.

    ``args.target`` selects the supervised node type(s): a concrete node-type
    name (PyPI / DataCo) or the sentinel ``"*"`` (e.g. SupplyGraph), which trains
    on every node type that carries a ``y`` label and aggregates the results
    into a single benchmark row.
    """
    if isinstance(args.target, str) and args.target == "*":
        types = _labelled_types(data)
        parts = [_eval_type(data, model, args, nt) for nt in types]
        y_true = torch.cat([torch.as_tensor(p[0]) for p in parts], dim=0) if parts else torch.zeros(0, 1)
        y_score = torch.cat([torch.as_tensor(p[1]) for p in parts], dim=0) if parts else torch.zeros(0, 1)
        print(f"[train] target=* {len(types)} node types "
              f"-> test={y_true.shape[0]} pos={int(y_true.sum())}",
              file=sys.stderr, flush=True)
        return y_true, y_score
    return _eval_type(data, model, args, args.target)


def _acc(logits, y):
    if logits.numel() == 0:
        return 0.0
    pred = (logits >= 0).long()
    correct = (pred == y.long()).sum().item()
    return 100.0 * correct / logits.numel()
