"""Uniform train + evaluate wrapper.

Every model/dataset pair goes through :func:`run`, which guarantees a single
code path for logging into the benchmark CSV. Model-specific logic lives in
``build_fn`` / ``train_eval_fn`` callbacks.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from pipeline import metrics as M


def run(
    dataset_name,
    model_name,
    graph_path=None,
    data=None,
    build_fn=None,
    train_eval_fn=None,
    device=None,
    notes="",
    csv_path=None,
    verbose=True,
):
    """Load a HeteroData graph, build+train a model, evaluate, log to CSV.

    Args:
        dataset_name: alias from ``pipeline.config``
        model_name: model label written to the CSV
        graph_path: path to a saved ``.pt`` HeteroData (optional, e.g. mock data)
        build_fn: ``fn(data) -> nn.Module`` builds the model from graph metadata
        train_eval_fn: ``fn(data, model) -> (y_true, y_score)`` trains and returns
            integer labels and class-probabilities for the test split
        device: torch device (defaults to cuda if available)
        notes: free-text annotation stored in the CSV
    """
    import torch_geometric
    from torch_geometric.data import HeteroData

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data is None:
        if graph_path is None:
            data = HeteroData()
        else:
            try:
                data = torch.load(graph_path, weights_only=False)
            except RuntimeError as e:
                # The graph was saved with CUDA storage but CUDA is now hidden
                # or unavailable (e.g. an unsupported GPU), so reload on CPU.
                if os.environ.get("RUNNER_DBG"):
                    import sys as _sys
                    _sys.stderr.write(f"[RUNNER] torch.load failed, retrying on CPU: {e}\n")
                data = torch.load(graph_path, weights_only=False, map_location="cpu")
            else:
                # Some storages may have been saved on CUDA (e.g. features on
                # an unsupported GPU) while others are on CPU. If any feature
                # tensor is on CUDA, reload the whole graph on CPU so the
                # training path stays device-consistent.
                if data.is_cuda:
                    if os.environ.get("RUNNER_DBG"):
                        import sys as _sys
                        _sys.stderr.write("[RUNNER] graph loaded with CUDA tensors; reloading on CPU\n")
                    data = torch.load(graph_path, weights_only=False, map_location="cpu")
    # Callers (e.g. the driver) may pre-process the graph here (e.g. subsample
    # isolated nodes) so that both build_fn and train_eval_fn operate on the
    # same, already-subsampled HeteroData instead of reloading from disk.
    elif not isinstance(data, HeteroData):
        raise TypeError(f"data must be a HeteroData, got {type(data).__name__}")

    # Always subsample the graph so build_fn / train_eval_fn operate on a
    # consistent, connected HeteroData -- regardless of whether the caller
    # passed a pre-loaded graph or let us reload it from disk. This drops
    # isolated nodes and dangling edges (some builders emit edges that
    # reference node ids which were later dropped, e.g. PyPI's depends_on
    # edges). Callers that already subsampled get a harmless no-op.
    try:
        from pipeline.driver import keep_connected
        keep_connected(data)
    except Exception as e:
        if os.environ.get("RUNNER_DBG"):
            import sys as _sys
            _sys.stderr.write(f"[RUNNER] keep_connected skipped: {e}\n")

    if build_fn is None or train_eval_fn is None:
        raise ValueError("build_fn and train_eval_fn are required")

    # Some GPUs are detected as available but cannot run this torch build's
    # kernels (e.g. an unsupported card). Fall back to CPU so training still
    # runs instead of aborting on the first CUDA error.
    if device.type == "cuda" and torch.cuda.device_count() >= 1:
        try:
            _ = torch.zeros(1, device=device).float().sum()
        except RuntimeError:
            print(f"[RUNNER] GPU {device} cannot run this torch build; "
                  f"falling back to CPU", file=sys.stderr)
            device = torch.device("cpu")

    t0 = time.time()

    def _run(device):
        model = build_fn(data).to(device)
        return train_eval_fn(data, model)

    # The training kernels may fail on a GPU that torch reports as available
    # but cannot actually execute (e.g. an unsupported card). If a CUDA error
    # occurs, fall back to CPU so the run still completes with full data.
    try:
        y_true, y_score = _run(device)
    except RuntimeError as e:
        if device.type == "cuda" and "CUDA error" in str(e):
            print(f"[RUNNER] CUDA {device} cannot run this build ({e}); "
                  f"falling back to CPU", file=sys.stderr)
            device = torch.device("cpu")
            y_true, y_score = _run(device)
        else:
            raise
    elapsed = time.time() - t0

    if verbose:
        print(
            f"[{dataset_name}] {model_name}: "
            f"test={y_true.shape[0]} pos={int(y_true.sum())} "
            f"train+eval={elapsed:.2f}s"
        )

    result = M.compute(y_true, y_score, pos_label=1)
    result.dataset = dataset_name
    result.model = model_name
    if notes:
        result.notes = notes

    # ``csv_path`` lets a caller (e.g. the sweep's per-dataset loop) direct this
    # run's row at a specific file (e.g. logdir/<dataset>_benchmark.csv) instead
    # of the default consolidated benchmark CSV.
    path = M.write_csv([result], path=csv_path)
    if verbose:
        print(M.summary_table([result]))
        print(f"  -> logged to {path}\n")
    return result
