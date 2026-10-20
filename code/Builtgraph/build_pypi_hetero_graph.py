#!/usr/bin/env python3
"""Build a PyG ``HeteroData`` graph from the raw PyPI parquet sources.

The full PyPI graph is ~15.9M nodes / ~212M edges, so this script is
**parameterized by shard count** to make it runnable in dry-run mode:

    python build_pypi_hetero_graph.py --edge_shards 1 --max_versions 20000

For a full-scale build, set ``--edge_shards 444063`` (all edge shards) and
remove the ``--max_versions`` cap. The heavy lifting (joins, feature
engineering, temporal splits) is done in DuckDB.

Schema (verified against the parquet sources):
  PackageVersion : package_name, version, published_at, stargazers_count,
                   forks_count, commit_activity_count, first_seen_event,
                   last_seen_event
  edges/deps     : package_name, version, dependency_name, dependency_version,
                   SnapshotAt
  versions       : system, package_name, version, Licenses, SnapshotAt
  projects       : project_type, project_name, StarsCount, ForksCount,
                   OpenIssuesCount, SnapshotAt
  advisories     : Source, SourceID, SourceURL, Title, Description, SnapshotAt

Node types / feature dims:
  PackageVersion (feat=6), Package (feat=3), Repository (feat=3)
Edge types: version_of, depends_on (PackageVersion->Package), hosted_on
  (PackageVersion->Repository)
Target: PackageVersion.y = binary advisory/vulnerability label.
"""
import argparse
import os
import resource
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "pypi")
SHARDS_PER_DIR = {
    "features": "pypi_features_*.parquet",
    "edges": "edges/**/*.parquet",
    "versions": "versions/**/*.parquet",
    "projects": "projects/**/*.parquet",
    "advisories": "advisories/**/*.parquet",
}

TRAIN_CUTOFF = np.datetime64("2023-06-01")
VAL_CUTOFF = np.datetime64("2024-01-01")


def _glob(path, n):
    """Take the first ``n`` files matching ``path`` glob."""
    import glob
    files = sorted(glob.glob(path, recursive=True))
    return files[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_shards", type=int, default=1,
                    help="number of edge (deps_*.parquet) shards to read")
    ap.add_argument("--version_shards", type=int, default=None,
                    help="cap on pypi_features shards (default: all)")
    ap.add_argument("--project_shards", type=int, default=None)
    ap.add_argument("--adv_shards", type=int, default=None)
    ap.add_argument("--max_versions", type=int, default=0,
                    help="hard cap on PackageVersion rows (0 = unlimited)")
    ap.add_argument("--out", type=str, default=os.path.join(DATA_DIR, "pypi_hetero_graph_sampled.pt"))
    args = ap.parse_args()

    t0 = time.time()
    con = duckdb.connect(database=":memory:")
    con.execute("SET memory_limit='32GB';")
    con.execute("SET threads TO 4;")

    print("[1/7] Registering parquet views...")
    con.execute(f"CREATE VIEW pypi_features AS SELECT * FROM read_parquet('{DATA_DIR}/features/pypi_features_*.parquet')")
    con.execute(f"CREATE VIEW projects_raw AS SELECT * FROM read_parquet('{DATA_DIR}/projects/**/*.parquet')")
    con.execute(f"CREATE VIEW advisories_raw AS SELECT * FROM read_parquet('{DATA_DIR}/advisories/**/*.parquet')")

    n_edge = args.edge_shards
    n_ver = args.version_shards or len(_glob(f"{DATA_DIR}/features/pypi_features_*.parquet", 10 ** 9))
    n_proj = args.project_shards or len(_glob(f"{DATA_DIR}/projects/**/*.parquet", 10 ** 9))
    n_adv = args.adv_shards or len(_glob(f"{DATA_DIR}/advisories/**/*.parquet", 10 ** 9))
    print(f"    features={n_ver} shards, projects={n_proj} shards, "
          f"advisories={n_adv} shards, edges={n_edge} shards")

    print("[2/7] Building PackageVersion / Package / Repository nodes...")
    con.execute("""
        CREATE TABLE pkg_versions AS
        SELECT ROW_NUMBER() OVER () - 1 AS node_id, package_name, version,
               published_at,
               COALESCE(stargazers_count,0) AS stargazers_count,
               COALESCE(forks_count,0) AS forks_count,
               COALESCE(commit_activity_count,0) AS commit_activity_count,
               first_seen_event, last_seen_event
        FROM pypi_features
        WHERE package_name IS NOT NULL AND version IS NOT NULL
        ORDER BY node_id;
    """)
    con.execute("""
        CREATE TABLE packages AS
        SELECT ROW_NUMBER() OVER () - 1 AS node_id, package_name
        FROM (SELECT DISTINCT package_name FROM pkg_versions ORDER BY node_id);
    """)
    con.execute("""
        CREATE TABLE repositories AS
        SELECT ROW_NUMBER() OVER () - 1 AS node_id, project_name,
               LOWER(SPLIT_PART(project_name, '/', 2)) AS repo_slug
        FROM (SELECT DISTINCT project_name FROM projects_raw
              WHERE project_name IS NOT NULL);
    """)
    n_pv = con.execute("SELECT COUNT(*) FROM pkg_versions").fetchone()[0]
    print(f"    PackageVersion={n_pv}, Package={con.execute('SELECT COUNT(*) FROM packages').fetchone()[0]}, "
          f"Repository={con.execute('SELECT COUNT(*) FROM repositories').fetchone()[0]}")

    if args.max_versions:
        con.execute(f"DELETE FROM pkg_versions WHERE node_id >= {args.max_versions};")
        print(f"    capped PackageVersion -> {args.max_versions} (rebuilding derived tables)")
        con.execute("""
            CREATE OR REPLACE TABLE packages AS
            SELECT ROW_NUMBER() OVER () - 1 AS node_id, package_name
            FROM (SELECT DISTINCT package_name FROM pkg_versions);
        """)

    print("[3/7] Feature engineering & labels...")
    pv_df = con.execute("""
        WITH adv_tokens AS (
            SELECT DISTINCT LOWER(UNNEST(STRING_SPLIT_REGEX(
                COALESCE(Title,'')||' '||COALESCE(Description,''), '[^a-zA-Z0-9_-]+'))) AS token
            FROM advisories_raw),
        adv_matches AS (
            SELECT DISTINCT pv.node_id
            FROM pkg_versions pv JOIN adv_tokens t
            ON LOWER(pv.package_name) = t.token
            WHERE LENGTH(pv.package_name) > 2)
        SELECT pv.node_id, pv.published_at,
               LENGTH(pv.version) AS version_len,
               CASE WHEN pv.version SIMILAR TO '.*(a|b|rc|dev|alpha|beta).*' THEN 1.0 ELSE 0.0 END AS is_pre_release,
               LOG2(GREATEST(pv.stargazers_count,0)+1) AS log_stars,
               LOG2(GREATEST(pv.forks_count,0)+1) AS log_forks,
               LOG2(GREATEST(pv.commit_activity_count,0)+1) AS log_commits,
               COALESCE(EXTRACT(EPOCH FROM (pv.last_seen_event - pv.first_seen_event))/86400.0,0.0) AS repo_age_days,
               CASE WHEN adv.node_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM pkg_versions pv LEFT JOIN adv_matches adv
        ON pv.node_id = adv.node_id
        ORDER BY pv.node_id;
    """).fetchdf()
    if args.max_versions:
        pv_df = pv_df.head(args.max_versions)
    p_df = con.execute("""
        SELECT p.node_id, LENGTH(p.package_name) AS name_len,
               COUNT(pv.version) AS total_versions_count,
               COALESCE(EXTRACT(EPOCH FROM (MAX(pv.published_at)-MIN(pv.published_at)))/86400.0,0.0) AS package_age_days
        FROM packages p LEFT JOIN pkg_versions pv ON p.package_name = pv.package_name
        GROUP BY p.node_id, p.package_name ORDER BY p.node_id;
    """).fetchdf()
    r_df = con.execute("""
        SELECT r.node_id,
               LOG2(GREATEST(COALESCE(MAX(pr.StarsCount),0),0)+1) AS log_stars,
               LOG2(GREATEST(COALESCE(MAX(pr.ForksCount),0),0)+1) AS log_forks,
               LOG2(GREATEST(COALESCE(MAX(pr.OpenIssuesCount),0),0)+1) AS log_issues
        FROM repositories r LEFT JOIN projects_raw pr ON r.project_name = pr.project_name
        GROUP BY r.node_id ORDER BY r.node_id;
    """).fetchdf()
    pv_cols = ["version_len", "is_pre_release", "log_stars", "log_forks", "log_commits", "repo_age_days"]
    X_pv = torch.tensor(pv_df[pv_cols].values.astype("float32"))
    Y_pv = torch.tensor(pv_df["label"].values.astype("long"))
    X_p = torch.tensor(p_df[["name_len", "total_versions_count", "package_age_days"]].values.astype("float32"))
    X_r = torch.tensor(r_df[["log_stars", "log_forks", "log_issues"]].values.astype("float32"))
    del p_df, r_df
    _rss = lambda: f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}"
    print(f"    RSS_after_features={_rss()}MB", flush=True)
    print(f"    features: PackageVersion={X_pv.shape[1]}d x {X_pv.shape[0]}, "
          f"Package={X_p.shape[0]}, Repository={X_r.shape[0]}, "
          f"label pos_rate={(Y_pv.float().mean()):.3f}")

    print("[4/7] Constructing relational edges...")
    # depends_on edges: read the ~183M dependency rows through a single DuckDB
    # view (NOT the old 444k-file pandas loop -- holding 183M string rows in a
    # pandas DataFrame would blow past RAM). Resolve (pkg,ver)->pv id and
    # dep_name->package id entirely in SQL, then fetch src/dst one column at a
    # time so the peak is a single ~1.5GB int64 array rather than a 366M-cell
    # DataFrame.
    con.execute(f"CREATE VIEW edges_raw AS SELECT * FROM read_parquet('{DATA_DIR}/edges/**/*.parquet')")
    src = con.execute("""
        SELECT pv.node_id AS src
        FROM edges_raw e
        JOIN pkg_versions pv ON LOWER(e.package_name) = pv.package_name
                               AND LOWER(e.version) = pv.version
        JOIN packages p ON LOWER(e.dependency_name) = p.package_name;
    """).fetchdf()["src"].values.astype(np.int64)
    dst = con.execute("""
        SELECT p.node_id AS dst
        FROM edges_raw e
        JOIN pkg_versions pv ON LOWER(e.package_name) = pv.package_name
                               AND LOWER(e.version) = pv.version
        JOIN packages p ON LOWER(e.dependency_name) = p.package_name;
    """).fetchdf()["dst"].values.astype(np.int64)
    print(f"    depends_on edges={len(src)} RSS={_rss()}MB", flush=True)

    version_of = con.execute("""
        SELECT pv.node_id AS src, p.node_id AS dst
        FROM pkg_versions pv JOIN packages p ON pv.package_name = p.package_name;
    """).fetchdf()
    vo_src = version_of["src"].values.astype(np.int64)
    vo_dst = version_of["dst"].values.astype(np.int64)
    del version_of
    hosted_on = con.execute("""
        SELECT DISTINCT pv.node_id AS src, r.node_id AS dst
        FROM pkg_versions pv JOIN repositories r
        ON LOWER(pv.package_name) = r.repo_slug AND LENGTH(r.repo_slug) > 2;
    """).fetchdf()
    ho_src = hosted_on["src"].values.astype(np.int64)
    ho_dst = hosted_on["dst"].values.astype(np.int64)
    del hosted_on
    print(f"    version_of edges={len(vo_src)}, hosted_on edges={len(ho_src)} RSS={_rss()}MB", flush=True)

    print("[5/7] Assembling HeteroData...")
    data = HeteroData()
    data["PackageVersion"].x = X_pv
    data["PackageVersion"].y = Y_pv
    data["Package"].x = X_p
    data["Repository"].x = X_r
    print(f"    before_version_of_tensor RSS={_rss()}MB", flush=True)
    data["PackageVersion", "version_of", "Package"].edge_index = torch.tensor([vo_src, vo_dst], dtype=torch.long)
    data["PackageVersion", "depends_on", "Package"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    del src, dst
    data["PackageVersion", "hosted_on", "Repository"].edge_index = torch.tensor([ho_src, ho_dst], dtype=torch.long)
    del ho_src, ho_dst
    # reverse (undirected) edges, explicitly oriented as [dst_node, src_node]
    vo = data["PackageVersion", "version_of", "Package"]
    ho = data["PackageVersion", "hosted_on", "Repository"]
    data["Package", "version_of", "PackageVersion"].edge_index = torch.cat([vo.edge_index[1].unsqueeze(0), vo.edge_index[0].unsqueeze(0)], dim=0)
    del vo_src, vo_dst
    data["Repository", "hosted_on", "PackageVersion"].edge_index = torch.cat([ho.edge_index[1].unsqueeze(0), ho.edge_index[0].unsqueeze(0)], dim=0)
    del ho

    print("[6/7] Temporal train/val/test masks...")
    ts = pd.to_datetime(pv_df["published_at"]).values.astype("datetime64[us]")
    ts = np.where(pd.isna(ts), np.datetime64("1970-01-01"), ts)
    train_mask = torch.tensor(ts < TRAIN_CUTOFF, dtype=torch.bool)
    val_mask = torch.tensor((ts >= TRAIN_CUTOFF) & (ts < VAL_CUTOFF), dtype=torch.bool)
    test_mask = torch.tensor(ts >= VAL_CUTOFF, dtype=torch.bool)
    data["PackageVersion"].train_mask = train_mask
    data["PackageVersion"].val_mask = val_mask
    data["PackageVersion"].test_mask = test_mask
    print(f"    train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}")

    print(f"[7/7] Saving -> {args.out}")
    torch.save(data, args.out)
    print(f"SUCCESS in {time.time()-t0:.1f}s: {data}")


if __name__ == "__main__":
    main()
