"""Shared harness for the heterogeneous-GNN / Mini-Transformer benchmark pipeline.

Provides:
  - ``config``   : dataset registry (schemas, paths, split config)
  - ``metrics``  : standardized classification metrics + CSV logging
  - ``runner``   : a uniform train+evaluate wrapper used by every model/dataset
"""
from pipeline import config, metrics, runner

__all__ = ["config", "metrics", "runner"]
