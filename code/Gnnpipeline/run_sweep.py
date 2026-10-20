#!/usr/bin/env python3
"""Full benchmark sweep for the p5ailab heterogeneous-GNN experiment.

Runs the complete matrix of every model on every dataset, across several random
seeds and a configurable number of epochs, and appends one CSV row per
(run) to ``results/<dataset>_benchmark.csv``.  A consolidated table of the best
result per (Dataset, Model) is printed at the end.

The sweep reuses the shared harness (pipeline.runner / pipeline.training) so the
code path is identical to ``pipeline/driver.py`` -- only the loop over the
experimental grid is added here.  This is what powers a stable, reproducible
"full experiment": many epochs and seeds so the reported AUC/F1 are estimates,
not single-run noise.

Usage
-----
    python run_sweep.py [--datasets pypi,dataco,supplygraph] \
        [--models SpikingHAN,CausalHGNN,RelGT,MiniTransformer,BaselineHeteroGNN] \
        [--seeds 1,2,3] [--epochs 200] [--lr 0.01] [--hidden 64] \
        [--logdir results] [--seed-arg 0]

Everything runs on GPU when one is present (see pipeline/models/hgnn.py -- the
forward pass moves its inputs onto the parameters' device) and falls back to CPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import HeteroData

# Make the bundled ``pipeline`` package importable regardless of the caller's CWD.
_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from pipeline import runner, training            # noqa: E402
from pipeline.config import DATASETS, MODELS, ROOT  # noqa: E402

# Directory holding the shipped, GPU-friendly HeteroData graphs
# ("<alias>_hetero_graph_<variant>.pt"). Referenced by both main() and run_grid()
# so graph selection works from anywhere (it is a module-level constant).
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Directory for on-demand FULL-scale HeteroData graphs (built when missing).
# The self-contained package ships GPU-friendly SAMPLES here (datasets/); this
# dir holds the full graphs when --build-full is used on a machine with enough
# RAM/VRAM (e.g. AI-LAB's 96 GB GPUs).
FULL_GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "full_graphs")  # workspace-root full_graphs/

# Build specs: alias -> (build_script relative to cfg["raw_root"], full-scale args).
# A full build reads the raw inputs from cfg["raw_root"] (or the script's own
# BASE_DIR) and writes the HeteroData to FULL_GRAPHS_DIR/<alias>_hetero_graph_full.pt.
# Full graphs are only built when missing; any failure falls back to the shipped
# sample so the sweep always completes.
# Build scripts live in <SCRIPTS_DIR>/build_*.py. They are invoked with
# cwd=<SCRIPTS_DIR> so PyPI's in-process DuckDB reads its raw parquet shards
# from its own BASE_DIR; DataCo/SupplyGraph resolve their raw inputs from their
# own (absolute) BASE_DIR/RAW, so cwd is irrelevant for them.
# SCRIPTS_DIR defaults to the real location of the build scripts + PyPI raw data;
# override with the P5A_SCRIPTS_DIR env var (e.g. on a different workspace).
SCRIPTS_DIR = os.environ.get("P5A_SCRIPTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Builtgraph"))

# Build specs: alias -> (build_script name, full-scale args).
BUILD_SPECS = {
    "pypi":       ("build_pypi_hetero_graph.py",       ["--edge_shards", "444063"]),
    "dataco":     ("build_dataco_graph.py",            []),
    "supplygraph":("build_supplygraph_graph.py",       []),
}


def load_config(path=None):
    """Read experiment defaults from ``experiment.yaml`` (optional).

    PyYAML is a *soft* dependency: if it is missing (or the file is absent /
    malformed) we silently fall back to the CLI defaults so the sweep still runs.
    Keeping the YAML in sync with these defaults is the single source of truth
    for the experiment grid (see p5ailab/experiment.yaml).
    """
    try:
        import yaml
    except ImportError:
        return {}
    if not path or not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - never fail the sweep over config
        print(f"[config] could not read {path}: {e}; using CLI defaults", file=sys.stderr)
        return {}
    return cfg


@dataclass
class Grid:
    datasets: list = field(default_factory=list)
    models: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    epochs: int = 200
    lr: float = 0.01
    warmup: int = 0
    weight_decay: float = 5e-4
    hidden: int = 64
    seed_arg: int = 0
    self_loop: bool = True
    logdir: str = "results"


def build_run_fns(dataset_alias: str, graph_path: str, model_kind: str, seed, g, target):
    """Return (build_fn, train_eval_fn) for a single (dataset, model, seed) run.

    Mirrors ``pipeline/driver.py`` so the sweep shares the exact same graph
    subsampling (``keep_connected``) and model construction. A fresh ``Args`` is
    built *inside* ``train_eval_fn`` so each seed gets its own RNG seed (the
    stratified splits in ``training.py`` are seeded from ``args.seed``).
    """
    import torch, sys as _sys
    from pipeline import driver as drv

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    drv.keep_connected(data)
    if os.environ.get("P5A_DBG") == "1":
        _sys.stderr.write(f"[DBG {dataset_alias}] post-keep nodes={ {nt:int(data[nt].x.shape[0]) for nt in data.node_types} }\n")
        for et in data.edge_types:
            ei=torch.as_tensor(data[et].edge_index); s,e=ei[0],ei[1]
            _sys.stderr.write(f"  edge {et} n={int(ei.shape[1])} src_max={int(s.max()) if ei.numel()>0 else -1} dst_max={int(e.max()) if ei.numel()>0 else -1}\n")
    feat_dim = {nt: data[nt].x.shape[-1] for nt in data.node_types}

    hidden, self_loop, epochs, lr, warmup, weight_decay, batch_size = (
        g.hidden, g.self_loop, g.epochs, g.lr, g.warmup, g.weight_decay, getattr(g, "batch_size", 0))

    def build_fn(d):
        from pipeline.models.hgnn import build_fn as hf
        fwd, seen = [], set()
        for et in d.edge_types:
            if d[et[0], et[1], et[2]].num_edges == 0:
                continue
            key = (et[0], et[1], et[2])
            if key not in seen:
                fwd.append(key); seen.add(key)
        return hf(fwd, feat_dim, hidden, 2, kind=model_kind, use_self_loop=self_loop)

    def train_eval_fn(d, m):
        args = type("Args", (), {
            "model": model_kind, "target": target, "epochs": epochs, "lr": lr,
            "warmup": warmup, "weight_decay": weight_decay,
            "hidden": hidden, "seed": seed,
            "batch_size": batch_size,
        })()
        return training.train_eval_fn(d, m, args)

    return build_fn, train_eval_fn


def _build_full_graph(alias: str, cfg):
    """Return the FULL-scale graph path for ``alias``, building it if missing.

    Priority: (1) an existing full graph in ``FULL_GRAPHS_DIR``; (2) build it by
    running the dataset's build script at full scale (raw inputs from
    ``cfg["raw_root"]``); (3) return ``None`` if the build is not possible. This
    NEVER raises -- callers fall back to the shipped GPU-friendly sample.
    """
    full_path = os.path.join(FULL_GRAPHS_DIR, f"{alias}_hetero_graph_full.pt")
    if os.path.exists(full_path):
        return full_path
    os.makedirs(FULL_GRAPHS_DIR, exist_ok=True)  # torch.save needs the dir to exist
    if alias not in BUILD_SPECS:
        return None
    import subprocess
    script = os.path.join(SCRIPTS_DIR, BUILD_SPECS[alias][0])
    if not os.path.exists(script):
        print(f"  [{alias}] no build script at {script}; using shipped sample",
              file=sys.stderr)
        return None
    print(f"  [{alias}] building FULL graph -> {full_path} "
          f"(this is heavy; needs the raw inputs present)", file=sys.stderr)
    proc = subprocess.run([sys.executable, script] + list(BUILD_SPECS[alias][1])
                          + ["--out", full_path],
                          cwd=SCRIPTS_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [{alias}] full build FAILED (rc={proc.returncode}); "
              f"falling back to shipped sample",
              file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[-2000:], file=sys.stderr)
        return None
    if not os.path.exists(full_path):
        print(f"  [{alias}] full build produced no output; using shipped sample",
              file=sys.stderr)
        return None
    print(f"  [{alias}] full graph built -> {full_path}", file=sys.stderr)
    return full_path


def resolve_graph_path(alias: str, cfg, build_full: bool):
    """Resolve which graph to train on for ``alias``.

    Resolution priority (so a 96 GB GPU can use full graphs when available):
      1. FULL-scale graph -- shipped full, or auto-built via _build_full_graph.
      2. Smallest GPU-friendly SAMPLE shipped in datasets/ (the historical
         fallback if the full graph is absent or its build is impossible).
    """
    graph_path = None
    if build_full and alias in BUILD_SPECS:
        graph_path = _build_full_graph(alias, cfg)
    if graph_path is None:
        for variant in ("_10k_sample", "_sampled", "_benchmark", ""):
            matches = sorted(glob.glob(
                os.path.join(DATASETS_DIR, f"{alias}_hetero_graph{variant}.pt")))
            if matches:
                graph_path = matches[0]
                break
    return graph_path


def parse_list(s):
    # Accept both YAML lists (seeds/models/datasets) and CLI CSV strings.
    if isinstance(s, list):
        return [str(x) for x in s]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    # Tunable flags default to argparse.SUPPRESS so we can tell whether the user
    # passed them explicitly (CLI wins over experiment.yaml) or should use the
    # YAML / built-in defaults instead.
    SUPPRESS = argparse.SUPPRESS
    p = argparse.ArgumentParser(description="Full HGNN benchmark sweep.")
    p.add_argument("--datasets", default=SUPPRESS,
                   help="comma-separated dataset aliases (default: all in experiment.yaml)")
    p.add_argument("--models", default=SUPPRESS, help="comma-separated model kinds")
    p.add_argument("--seeds", default=SUPPRESS, help="comma-separated seeds")
    p.add_argument("--graph", default=SUPPRESS,
                   help="override the .pt graph per dataset (default: smallest GPU-friendly sample)")
    p.add_argument("--self-loop", action=argparse.BooleanOptionalAction,
                   default=SUPPRESS, help="fold each node's own features into the "
                   "message (self-loop); default from experiment.yaml")
    p.add_argument("--epochs", type=int, default=SUPPRESS)
    p.add_argument("--lr", type=float, default=SUPPRESS)
    p.add_argument("--warmup", type=int, default=SUPPRESS, help="linear warmup epochs before cosine decay")
    p.add_argument("--weight-decay", type=float, default=SUPPRESS)
    p.add_argument("--hidden", type=int, default=SUPPRESS)
    p.add_argument("--batch-size", type=int, default=SUPPRESS,
                   help="mini-batch SGD size (0 = full-batch). Graphs with >= "
                        "MINI_BATCH_NODES target nodes auto-switch to mini-batch SGD.")
    p.add_argument("--seed-arg", type=int, default=SUPPRESS)
    p.add_argument("--logdir", default=SUPPRESS)
    p.add_argument("--build-full", action=argparse.BooleanOptionalAction,
                   default=SUPPRESS, help="auto-build each dataset's FULL-scale graph "
                   "from its raw data (heavy; needs the raw inputs present and lots of "
                   "RAM) when no full graph is shipped. Off by default: the sweep then "
                   "uses the shipped GPU-friendly sample. On AI-LAB (>=96 GB VRAM) run "
                   "with --build-full. Training then uses mini-batch SGD (auto for large "
                   "graphs) so even the full ~241M-edge PyPI graph trains in memory.")
    a = p.parse_args()

    # experiment.yaml is the source of truth; CLI flags above override it.
    cfg = load_config()
    cfg_seeds = parse_list(cfg.get("seeds", "1,2,3"))
    g = Grid()
    g.seeds = [int(s) for s in parse_list(a.seeds)] if hasattr(a, "seeds") else [int(s) for s in cfg_seeds]
    g.datasets = parse_list(a.datasets) if hasattr(a, "datasets") else list(cfg.get("datasets", []))
    g.models = parse_list(a.models) if hasattr(a, "models") \
        else list(cfg.get("models", MODELS + ["BaselineHeteroGNN"]))
    g.epochs = a.epochs if hasattr(a, "epochs") else cfg.get("epochs", 200)
    g.lr = a.lr if hasattr(a, "lr") else cfg.get("lr", 0.01)
    g.warmup = a.warmup if hasattr(a, "warmup") else cfg.get("warmup", 0)
    g.weight_decay = a.weight_decay if hasattr(a, "weight_decay") else cfg.get("weight_decay", 5e-4)
    g.hidden = a.hidden if hasattr(a, "hidden") else cfg.get("hidden", 64)
    g.seed_arg = a.seed_arg if hasattr(a, "seed_arg") else cfg.get("seed", 0)
    g.logdir = a.logdir if hasattr(a, "logdir") else cfg.get("logdir", "results")
    g.self_loop = a.self_loop if hasattr(a, "self_loop") else cfg.get("self_loop", True)
    g.build_full = a.build_full if hasattr(a, "build_full") else False

    # Build the list of datasets that actually have a .pt graph on disk.
    # The self-contained package ships GPU-friendly graphs in ``datasets/``
    # (named "<alias>_hetero_graph_<variant>.pt"); prefer those over the build
    # output dir (cfg["raw_root"]) which may not exist yet. --graph overrides all.
    have_graph = {}
    for alias, cfg in DATASETS.items():
        # shipped, GPU-friendly graphs take priority (the self-contained package
        # stores them in datasets/); the build output dir is only a fallback.
        shipped = sorted(glob.glob(os.path.join(DATASETS_DIR, f"{alias}_hetero_graph*.pt")))
        if shipped:
            have_graph[alias] = DATASETS_DIR
        elif os.path.isdir(cfg["raw_root"]):
            found = sorted(glob.glob(os.path.join(cfg["raw_root"], "*.pt")))
            if found:
                have_graph[alias] = cfg["raw_root"]
    if hasattr(a, "datasets"):
        g.datasets = [d for d in parse_list(a.datasets) if d in have_graph]
    else:
        g.datasets = list(have_graph.keys())

    if not g.models:
        g.models = list(MODELS) + ["BaselineHeteroGNN"]

    g.datasets.sort(); g.models.sort()
    print(f"Grid: datasets={g.datasets} models={g.models} "
          f"seeds={g.seeds} epochs={g.epochs} lr={g.lr} hidden={g.hidden}")

    os.makedirs(g.logdir, exist_ok=True)

    # per-dataset accumulators (ordered by first appearance).
    # ``rows`` / ``order`` are passed to ``run_grid`` by reference so the shared
    # grid runner (and the self-loop ablation, which reuses it) can populate them.
    order = []
    rows = {}

    t0 = time.time()
    rows, order = run_grid(g, g.seed_arg, g.logdir, rows, order)
    print(f"\nRan {sum(len(v) for v in rows.values())} runs in {time.time()-t0:.1f}s. "
          f"Per-dataset CSVs in {g.logdir}/")

    # ---- consolidated table: best result per (Dataset, Model) ----
    print("\n=== CONSOLIDATED (best per Dataset x Model) ===", file=sys.stderr)
    for alias in order:
        best = max(rows[alias], key=lambda r: r.roc_auc)
        print(f"{alias:<12} {best.model:<16} "
              f"prec {best.precision:.3f} rec {best.recall:.3f} "
              f"FP {int(best.false_positives):<4} FN {int(best.false_negatives):<4} "
              f"F1 {best.f1:.3f}  AUC {best.roc_auc:.3f}", file=sys.stderr)


def _seed_all(seed):
    """Central determinism: seed torch / numpy / python's RNG from one value."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_grid(g, seed_arg, logdir, rows, order):
    """Run the full (dataset, model, seed) grid.

    Mutates and returns ``rows`` (per-dataset -> list of Result) and ``order``.
    Shared by ``main`` and ``ablate_self_loop.py`` so the self-loop ablation can
    reuse the exact same training path with a single flag toggle.
    """
    _seed_all(seed_arg)
    t0 = time.time()
    n = 0

    def ensure_ds(alias):
        """Pre-create the per-dataset list + (append-mode) benchmark CSV header."""
        if alias not in rows:
            rows[alias] = []
            order.append(alias)
            path = os.path.join(logdir, f"{alias}_benchmark.csv")
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Dataset", "Model", "Precision",
                    "Recall", "False Positives", "False Negatives", "F1-Score",
                    "ROC-AUC", "N_Support", "Notes"])
                w.writeheader()
        return rows[alias]

    build_full = bool(os.environ.get("P5A_BUILD_FULL")) or g.build_full
    for alias in g.datasets:
        cfg = DATASETS[alias]
        # Resolve which graph to train on. With --build-full the FULL-scale graph
        # is built on demand and preferred; otherwise the smallest GPU-friendly
        # shipped sample is used. --graph overrides this entirely.
        graph_path = resolve_graph_path(alias, cfg, build_full)
        _dbg = os.environ.get("P5A_DBG") == "1"
        if _dbg:
            import sys as _sys, io as _io, torch as _t
            _sys.stderr.write(f"[DBG {alias}] selected={graph_path}\n")
            try:
                d = _t.load(io.BytesIO(open(graph_path, "rb").read()), weights_only=False)
                _sys.stderr.write(f"[DBG {alias}] nodes={ {nt: int(d[nt].x.shape[0]) for nt in d.node_types} }\n")
            except Exception as e:
                _sys.stderr.write(f"[DBG {alias}] load-err={e}\n")
        # The default loop above already chose the smallest GPU-friendly sample;
        # an explicit --graph override replaces it (and is validated for existence).
        if not os.path.exists(graph_path):
            print(f"  [{alias}] NO graph file found -- skipping", file=sys.stderr)
            continue
        for model in g.models:
            for seed in g.seeds:
                build_fn, train_eval_fn = build_run_fns(
                    alias, graph_path, model, seed, g, cfg["target_node"])
                csv_path = os.path.join(logdir, f"{alias}_benchmark.csv")
                ensure_ds(alias)             # pre-create the per-dataset list + CSV header first
                res = runner.run(
                    dataset_name=alias, model_name=model,
                    data=None, graph_path=graph_path,
                    build_fn=build_fn, train_eval_fn=train_eval_fn,
                    device=None, verbose=False, csv_path=csv_path,
                )
                res.notes = f"seed={seed} epochs={g.epochs}"
                rows[alias].append(res)
                n += 1
                print(f"  [{alias}] {model:<16} seed={seed:<3} "
                      f"AUC {res.roc_auc:.3f} F1 {res.f1:.3f} "
                      f"(test pos {res.n_support})", file=sys.stderr)
    return rows, order


if __name__ == "__main__":
    main()
