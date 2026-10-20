#!/usr/bin/env python3
"""Build a heterogeneous graph for the DataCo Supply-Chain (UCI Retail) dataset.

Entities (node types):
    Customer  (20k)  demographics + segment
    Order     (65k)  order status, profit, region, priority, dates
    Product   (118)  category, price, quantity, profit ratio

Edges (heterogeneous):
    Customer -> places -> Order
    Order    -> contains -> Product

Target: Order node, binary "SUSPECTED_FRAUD" (a minority class ~6%) vs other
status.  Structurally learnable because fraud orders correlate with customer
segment, department and product category.

The raw CSV is long-format with mixed encodings; only the columns needed for the
schema are loaded.  Output is a HeteroData ``.pt`` ready for the trainer.
"""
import argparse
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def build(args):
    raw = args.raw
    df = pd.read_csv(raw, encoding="latin-1", low_memory=False)

    cust_cols = ["Customer Id", "Customer Segment", "Customer City", "Customer Country",
                 "Customer State", "Customer Fname", "Customer Lname"]
    order_cols = ["Order Id", "Order Status", "Order Profit Per Order", "Order Region",
                  "Order State", "Order Customer Id", "Order Zipcode",
                  "Product Card Id", "order date (DateOrders)", "shipping date (DateOrders)"]
    prod_cols = ["Product Card Id", "Product Name", "Product Category Id",
                 "Order Item Total", "Order Item Quantity", "Order Item Profit Ratio",
                 "Product Price", "Sales"]

    cust = df[[c for c in cust_cols if c in df.columns]].dropna(subset=["Customer Id"])
    orders = df[[c for c in order_cols if c in df.columns]].dropna(subset=["Order Id"])
    prods = df[[c for c in prod_cols if c in df.columns]].dropna(subset=["Product Card Id"])

    # sample to keep training tractable on CPU
    lim = args.max_nodes
    cust = cust.head(lim)
    orders = orders.head(lim)
    prods = prods.head(lim)

    data = HeteroData()
    data.user = "DataCo supply-chain fraud detection"

    _add_customer(data, cust, lim)
    _add_order(data, orders, lim, cust)
    _add_product(data, prods, lim, orders)
    return data


def _zscore(vec):
    """Standardise a numeric column (per-column z-score) for stable training."""
    v = np.asarray(vec, dtype="float64")
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    m = v.mean(); s = v.std()
    return ((v - m) / s) if s > 1e-8 else np.zeros_like(v)


def _encode_categorical(vec):
    """Map a categorical column to small non-negative integers.

    The UCI Retail CSV keeps some object columns as ``str`` dtype but still
    stores missing values as ``float('nan')`` (e.g. ``Customer Lname``). Sorting
    a mixed float/str set raises ``TypeError`` (``'<'`` not supported between
    float and str), so we stringify every unique value before comparing.
    """
    uniques = sorted({str(u) for u in pd.unique(vec)})
    code = {u: i for i, u in enumerate(uniques)}
    return np.array([code.get(str(x)) for x in vec], dtype="float32")


def _feats(df, cols, out_dim, categorical=None):
    """Build a feature matrix, z-scoring numerics and ordinal-encoding categoricals."""
    categorical = categorical or set()
    mat = np.zeros((df.shape[0], max(out_dim, len(cols))), dtype="float32")
    for j, col in enumerate(cols):
        series = df[col]
        if col in categorical:
            mat[:, j] = _encode_categorical(series)
        else:
            mat[:, j] = _zscore(series.map(_num).to_numpy(dtype="float64"))
    mat = np.nan_to_num(mat, nan=0.0)
    if mat.shape[1] > out_dim:
        mat = mat[:, :out_dim]
    return torch.tensor(mat)


def _add_customer(data, cust, lim):
    feats = _feats(cust,
                  ["Customer City", "Customer Country", "Customer State", "Customer Fname",
                   "Customer Lname"], 5, categorical={"Customer City", "Customer Country",
                                                      "Customer State", "Customer Fname",
                                                      "Customer Lname"})
    y = torch.tensor((cust["Customer Segment"] != "Consumer").astype("int64"))  # Consumer vs other
    data["Customer"].x = feats
    data["Customer"].y = y
    data["Customer"].num_nodes = len(cust)


def _add_order(data, orders, lim, cust):
    feats = _feats(orders,
                  ["Order Profit Per Order", "Order Region", "Order State", "Order Zipcode",
                   "order date (DateOrders)", "shipping date (DateOrders)"], 4,
                  categorical={"Order Region", "Order State"})
    data["Order"].x = feats
    y = torch.tensor((orders["Order Status"] == "SUSPECTED_FRAUD").astype("int64"))
    data["Order"].y = y
    data["Order"].num_nodes = len(orders)

    # Customer -> places -> Order
    ci = {int(r): i for i, r in enumerate(cust["Customer Id"].values)}
    oi = {int(r): i for i, r in enumerate(orders["Order Id"].values)}
    src, dst = [], []
    for _, row in orders.iterrows():
        c = row["Order Customer Id"]
        if c in ci and oi.get(int(row["Order Id"])) is not None:
            src.append(ci[c]); dst.append(oi[int(row["Order Id"])])
    data["Customer", "places", "Order"].edge_index = torch.tensor([src, dst])


def _add_product(data, prods, lim, orders):
    prods = prods.head(lim)
    feats = _feats(prods,
                   ["Product Name", "Product Category Id", "Order Item Total",
                    "Order Item Quantity", "Order Item Profit Ratio", "Product Price", "Sales"], 6,
                   categorical={"Product Name"})
    data["Product"].x = feats
    data["Product"].num_nodes = len(prods)
    data["Product"].y = None

    # Order -> contains -> Product: join on the product's own id column.
    prods = prods.sort_values("Product Card Id")
    prod_id = prods["Product Card Id"].astype("int64").values
    prod_i = {pid: i for i, pid in enumerate(prod_id)}
    src, dst = [], []
    for _, row in orders.iterrows():
        pid = row.get("Product Card Id")
        try:
            pid_i = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_i in prod_i:
            i = prod_i[pid_i]
            src.append(i); dst.append(i)
    data["Order", "contains", "Product"].edge_index = torch.tensor([src, dst])
    print(f"  [product] edges {len(src)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(BASE, "..", "data", "dataco", "DataCo_hetero_graph_sampled.pt"))
    p.add_argument("--raw", default=os.path.join(BASE, "..", "data", "dataco", "DataCoSupplyChainDataset.csv"))
    p.add_argument("--max-nodes", type=int, default=3000)
    a = p.parse_args()
    torch.manual_seed(0)
    data = build(a)
    torch.save(data, a.out)
    nt = data["Order"].y
    print(f"[OK] DataCo -> {a.out}: Cust {data['Customer'].num_nodes} | "
          f"Order {data['Order'].num_nodes} | Product {data['Product'].num_nodes} | "
          f"fraud_pos {int(nt.sum())}/{len(nt)}")


if __name__ == "__main__":
    main()
