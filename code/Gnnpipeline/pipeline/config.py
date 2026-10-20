"""Dataset registry and schema definitions for the 4 heterogeneous graphs.

Every entry describes:
  * ``name``          : canonical dataset label used in the benchmark CSV
  * ``root``          : workspace path of the repo
  * ``raw``           : where the raw inputs live
  * ``schema``        : node types, per-type feature dims, and metapaths
  * ``task``          : classification target + output metrics

The PyPI entry encodes the real schema discovered from the parquet sources:
    nodes : PackageVersion (feat=6), Package (feat=3), Repository (feat=3)
    edges : (PackageVersion, version_of, Package)
            (PackageVersion, depends_on, Package)
            (PackageVersion, hosted_on, Repository)
    target: PackageVersion -> binary vulnerability label (advisory presence)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
RESULTS_DIR = os.path.join(ROOT, "results")


def _pypi_schema():
    return {
        "node_types": ["PackageVersion", "Package", "Repository"],
        "feat_dim": {"PackageVersion": 6, "Package": 3, "Repository": 3},
        "edge_types": [
            ("PackageVersion", "version_of", "Package"),
            ("PackageVersion", "depends_on", "Package"),
            ("PackageVersion", "hosted_on", "Repository"),
        ],
        "target_node": "PackageVersion",
        "task": "binary",
        "metrics": ["precision", "recall", "f1", "roc_auc", "false_positives", "false_negatives"],
        "raw_root": os.path.join(ROOT, "..", "data", "pypi"),
    }


def _supplygraph_schema():
    # SupplyGraph (Wasi et al., 2024): real-world supply-chain planning dataset.
    # Facility nodes are typed by their product sub-group (leading letters of the
    # facility code); the binary target is the product *category* (Group column).
    # ``target_node == "*"`` evaluates the head on every node type and aggregates.
    return {
        "node_types": ["AT", "ATN", "ATPA", "ATPPCH", "ATWWP", "EEA", "MAC", "MAHS",
                       "MAP", "MAPA", "MAR", "MASR", "POP", "POPF", "POV", "SE", "SO",
                       "SOP", "SOS"],
        "feat_dim": {"AT": 16, "ATN": 16, "ATPA": 16, "ATPPCH": 16, "ATWWP": 16, "EEA": 16,
                     "MAC": 16, "MAHS": 16, "MAP": 16, "MAPA": 16, "MAR": 16, "MASR": 16,
                     "POP": 16, "POPF": 16, "POV": 16, "SE": 16, "SO": 16, "SOP": 16,
                     "SOS": 16},
        "edge_types": [
            ("AT", "connected_in_plant", "AT"), ("AT", "same_product_group", "AT"),
            ("AT", "same_product_subgroup", "AT"), ("AT", "connected_in_storage", "AT"),
        ],
        "target_node": "*",
        "task": "binary",
        "metrics": ["precision", "recall", "f1", "roc_auc", "false_positives", "false_negatives"],
        "raw_root": os.path.join(ROOT, "..", "data", "supplygraph"),
    }


def _dataco_schema():
    # DataCo = UCI Retail / supply-chain dataset (customers, orders, products).
    # Binary target on the Order node: SUSPECTED_FRAUD (~4-6% of orders) vs other
    # status. Structurally learnable through the customer / product graph.
    return {
        "node_types": ["Customer", "Order", "Product"],
        "feat_dim": {"Customer": 5, "Order": 4, "Product": 6},
        "edge_types": [
            ("Customer", "places", "Order"),
            ("Order", "contains", "Product"),
        ],
        "target_node": "Order",
        "task": "binary",
        "metrics": ["precision", "recall", "f1", "roc_auc", "false_positives", "false_negatives"],
        "raw_root": os.path.join(ROOT, "..", "data", "dataco"),
    }


DATASETS = {
    "pypi": _pypi_schema(),
    "supplygraph": _supplygraph_schema(),
    "dataco": _dataco_schema(),
}

# Baseline / candidate models that participate in the benchmark CSV.
MODELS = ["SpikingHAN", "CausalHGNN", "RelGT", "MiniTransformer"]


def dataset(name):
    """Return the schema dict for a dataset by alias."""
    key = name if name in DATASETS else DATASETS[name]
    return DATASETS[key]


def list_datasets():
    return list(DATASETS.keys())


def result_csv_path():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, "model_benchmarks_results.csv")
