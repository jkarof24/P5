#!/usr/bin/env python3
"""Showcase the size of every dataset: nodes (per type), edges (per type) and the
feature dimension of every node type.

This is the "base map" the team will build GNN models on, so it records, for each
dataset, exactly how big the heterogeneous graph is before any training happens.

The numbers are computed directly from the *raw* data in ``data/<dataset>/`` (no
need to load the multi-GB built graphs), so they always reflect the real input.
"""
import os
import glob
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(ROOT, "..", "..", "data"))


def _hr(ch="="):
    return ch * 64


def stats_pypi():
    """PyPI: PackageVersion / Package / Repository + depends_on, version_of, hosted_on."""
    feats = sorted(glob.glob(os.path.join(DATA_ROOT, "pypi", "features", "pypi_features_*.parquet")))
    db = duckdb.connect(":memory:")
    quoted = ", ".join(f"'{f}'" for f in feats)
    db.execute(f"CREATE VIEW features AS SELECT * FROM read_parquet(CAST([{quoted}] AS VARCHAR[]))")
    db.execute(f"CREATE VIEW edges_raw AS SELECT * FROM read_parquet('{DATA_ROOT}/pypi/edges/**/*.parquet')")
    db.execute(f"CREATE VIEW projects_raw AS SELECT * FROM read_parquet('{DATA_ROOT}/pypi/projects/**/*.parquet')")
    db.execute("SET memory_limit='8GB'")

    # node counts
    pv = db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    pkg = db.execute("SELECT COUNT(DISTINCT package_name) FROM features").fetchone()[0]
    # Repository = distinct project_name (the build derives repo_slug from it)
    repo = db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT project_name FROM projects_raw) r").fetchone()[0]

    # edge counts (resolve names -> ids entirely in SQL, count only, never materialise)
    depends_on = db.execute("""
        SELECT COUNT(*)
        FROM edges_raw e
        JOIN features f   ON LOWER(e.package_name) = f.package_name AND LOWER(e.version) = f.version
        JOIN (SELECT DISTINCT package_name FROM features) p
          ON LOWER(e.dependency_name) = p.package_name;
    """).fetchone()[0]
    version_of = pv  # one per PackageVersion
    hosted_on = db.execute("""
        SELECT COUNT(*)
        FROM features pv
        JOIN (SELECT DISTINCT project_name FROM projects_raw WHERE project_name IS NOT NULL) pr
        ON LOWER(pv.package_name) = LOWER(SPLIT_PART(pr.project_name, '/', 2)) AND LENGTH(pr.project_name) > 2;
    """).fetchone()[0]

    feat_dim = {"PackageVersion": 6, "Package": 3, "Repository": 3}
    print(_hr())
    print("  PyPI Heterogeneous Graph  (data/pypi/)")
    print(_hr())
    print(f"  NODES ({len(feat_dim)} types):")
    for nt in ("PackageVersion", "Package", "Repository"):
        print(f"    {nt:14s} {pv if nt == 'PackageVersion' else (pkg if nt == 'Package' else repo):>12,} "
              f"nodes     feat_dim = {feat_dim[nt]}")
    print(f"  EDGES (per type):")
    print(f"    version_of        {version_of:>12,}")
    print(f"    depends_on        {depends_on:>12,}")
    print(f"    hosted_on         {hosted_on:>12,}")
    print(f"    + reverse edges   {(version_of + hosted_on):>12,}")
    print(_hr())
    print(f"  TOTAL edges         {(version_of + depends_on + hosted_on):>12,}")
    print(_hr(" "))


def stats_dataco(raw):
    """DataCo (UCI Retail): Customer / Order / Product + places / contains."""
    df = pd.read_csv(raw, encoding="latin-1", low_memory=False)
    cust = df["Customer Id"].nunique()
    orders = df["Order Id"].nunique()
    prods = df["Product Card Id"].nunique()
    places = int(df.groupby("Order Customer Id")["Order Id"].ngroups
                 if "Order Customer Id" in df else orders)
    contains = int(df.groupby("Product Card Id")["Order Id"].ngroups
                   if "Product Card Id" in df else orders)
    feat_dim = {"Customer": 5, "Order": 4, "Product": 6}
    print(_hr())
    print(f"  DataCo Supply-Chain (UCI Retail)  (data/dataco/)")
    print(_hr())
    print(f"  NODES ({len(feat_dim)} types):")
    for nt in ("Customer", "Order", "Product"):
        n = cust if nt == "Customer" else (orders if nt == "Order" else prods)
        print(f"    {nt:14s} {n:>12,} nodes     feat_dim = {feat_dim[nt]}")
    print(f"  EDGES (per type):")
    print(f"    places (Customer->Order) {places:>12,}")
    print(f"    contains (Order->Product){contains:>12,}")
    print(_hr())
    print(f"  TOTAL edges          {(places + contains):>12,}")
    print(_hr(" "))
    return places + contains


def stats_supplygraph():
    """SupplyGraph: facility nodes typed by product sub-group + 4 physical edges."""
    raw = os.path.join(DATA_ROOT, "supplygraph", "Homogenoeus")
    nodes = pd.read_csv(os.path.join(raw, "Nodes", "Nodes.csv"))["Node"].tolist()
    groups = pd.read_csv(os.path.join(raw, "Nodes", "Node Types (Product Group and Subgroup).csv"))
    group_map = dict(zip(groups["Node"], groups["Group"]))
    import re

    def prefix(n):
        m = re.match(r"^([A-Z]+)", n)
        return m.group(1) if m else "UNK"

    from collections import defaultdict
    by_type = defaultdict(list)
    for n in nodes:
        by_type[prefix(n)].append(n)
    node_types = sorted(by_type)
    feat_dim = {nt: len(by_type[nt]) for nt in node_types}
    edge_specs = {
        "connected_in_plant": os.path.join(raw, "Edges", "Edges (Plant).csv"),
        "same_product_group": os.path.join(raw, "Edges", "Edges (Product Group).csv"),
        "same_product_subgroup": os.path.join(raw, "Edges", "Edges (Product Sub-Group).csv"),
        "connected_in_storage": os.path.join(raw, "Edges", "Edges (Storage Location).csv"),
    }
    counts = {}
    for rel, path in edge_specs.items():
        df = pd.read_csv(path, skipinitialspace=True)
        c = 0
        for _, r in df.iterrows():
            st, dt = prefix(r["node1"].strip()), prefix(r["node2"].strip())
            if st != dt:
                c += 1
        counts[rel] = c
    print(_hr())
    print("  SupplyGraph (Wasi et al., 2024)  (data/supplygraph/)")
    print(_hr())
    print(f"  NODES ({len(node_types)} types):")
    for nt in node_types:
        print(f"    {nt:12s} {len(by_type[nt]):>12,} nodes     feat_dim = {feat_dim[nt]}")
    print(f"  EDGES (per type):")
    for rel in edge_specs:
        print(f"    {rel:22s} {counts[rel]:>12,}")
    print(_hr())
    print(f"  TOTAL edges (distinct, directed) {sum(counts.values()):>12,}")
    print(_hr(" "))
    return sum(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pypi", action="store_true", help="include PyPI stats (heavy)")
    ap.add_argument("--dataco", action="store_true", help="include DataCo stats")
    ap.add_argument("--supplygraph", action="store_true", help="include SupplyGraph stats")
    a = ap.parse_args()
    print(_hr("#"))
    print("  DATASET SIZE SHOWCASE  (nodes / edges per type / feature dims)")
    print(_hr("#"))

    if a.dataco:
        stats_dataco(os.path.join(DATA_ROOT, "dataco", "DataCoSupplyChainDataset.csv"))
    if a.supplygraph:
        stats_supplygraph()
    if a.pypi:
        stats_pypi()

    print(_hr())
    print("  Data lives under data/<dataset>/ ; build scripts under code/ ;")
    print("  built graphs (HeteroData .pt) are git-ignored build outputs.")
    print(_hr())


if __name__ == "__main__":
    main()
