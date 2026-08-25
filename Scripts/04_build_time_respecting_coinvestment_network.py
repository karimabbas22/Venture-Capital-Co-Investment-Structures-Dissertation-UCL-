"""
04_build_time_respecting_coinvestment_network.py — as-of-date co-investment
graph snapshots and startup-level network feature aggregation.

Protocol
────────
1. A grid of quarterly snapshot dates spans 1999-12-31 … 2017-12-31 (the
   grid's start is cosmetic only -- since ENTRY_START=2010-01-01, no
   startup anchor ever needs a grid point before 2009-12-31 in practice;
   the graph's actual historical depth comes from deals.parquet itself
   covering 2000-2024, not from this grid's nominal start).
2. For each snapshot date s, a co-investment graph is built from deals with
   investment_date <= s ONLY -- unbounded below, so every snapshot already
   draws on the FULL available history up to s, including pre-2010 deals.
3. Each startup is assigned the latest snapshot <= its anchor date, so all
   graph features are computed from a graph that existed at (slightly before)
   its anchor.
4. Node identity uses Refinitiv's Firm Investor Organization Id — no fuzzy
   matching needed.

Network features computed per investor node:
  degree, weighted_degree, betweenness_centrality, pagerank,
  eigenvector_lcc, clustering_coeff

Aggregated to startup level (across first-round investors):
  mean_degree, max_degree, mean_pagerank, max_pagerank,
  mean_betweenness, max_betweenness, mean_eigenvector_lcc,
  max_eigenvector_lcc, mean_clustering, has_top_pagerank_investor,
  share_newcomer_investors, n_investors_first_round

Outputs
───────
  Data/processed/investor_features_by_snapshot.parquet
  Data/processed/firms_with_network_T7.parquet
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    SEED, TRAIN_END, BETWEENNESS_K, PAGERANK_TOP_DECILE,
    build_snapshot_dates, assign_snapshot,
    build_coinvestment_snapshot, compute_snapshot_metrics, set_global_seed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "processed")

METRIC_COLS = ["degree", "weighted_degree", "betweenness_centrality",
               "pagerank", "eigenvector_lcc", "clustering_coeff"]


def main():
    set_global_seed()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Loading data ...")
    deals       = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    firms_panel = pd.read_parquet(os.path.join(DATA, "firms_panel.parquet"))

    # Exclude "Undisclosed Firm" — not a real investor, would create an
    # artificial mega-hub in the co-investment graph.
    n_before = len(deals)
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    print(f"  Excluded 'Undisclosed Firm': {n_before - len(deals):,} rows dropped")
    # Use investor_name as fallback for the ~71 named investors missing org IDs.
    # Convert all to string for consistent graph node types.
    deals["investor_org_id"] = deals["investor_org_id"].fillna(
        deals["investor_name"]
    ).astype(str)
    print(f"  deals               : {len(deals):,} rows "
          f"({deals['investment_date'].min().date()} – "
          f"{deals['investment_date'].max().date()})")
    print(f"  firms_panel         : {len(firms_panel):,} startups")

    # ── 2. Snapshot grid + per-startup assignment ────────────────────────────
    snapshot_dates = build_snapshot_dates()
    firms_panel = firms_panel.copy()
    firms_panel["snapshot_date"] = assign_snapshot(
        firms_panel["first_deal_date"], snapshot_dates
    )
    needed = sorted(set(firms_panel["snapshot_date"].unique()) | {TRAIN_END})
    print(f"\nSnapshot grid: {len(snapshot_dates)} quarter-ends "
          f"({snapshot_dates[0].date()} – {snapshot_dates[-1].date()}); "
          f"{len(needed)} needed by the panel")

    # ── 3. Per-snapshot graphs + metrics ──────────────────────────────────────
    all_metrics = []
    for i, snap in enumerate(needed):
        t0 = time.time()
        G = build_coinvestment_snapshot(deals, pd.Timestamp(snap))
        m = compute_snapshot_metrics(G, betweenness_k=BETWEENNESS_K, seed=SEED)
        m["snapshot_date"] = pd.Timestamp(snap)
        all_metrics.append(m)
        print(f"  [{i+1:2d}/{len(needed)}] {pd.Timestamp(snap).date()}  "
              f"nodes={G.number_of_nodes():>6,}  edges={G.number_of_edges():>7,}  "
              f"({time.time()-t0:5.1f}s)")

    inv_by_snap = pd.concat(all_metrics, ignore_index=True)

    pr_thresh = (
        inv_by_snap.groupby("snapshot_date")["pagerank"]
        .quantile(PAGERANK_TOP_DECILE)
        .rename("pagerank_top_decile_threshold")
        .reset_index()
    )
    inv_by_snap = inv_by_snap.merge(pr_thresh, on="snapshot_date", how="left")

    snap_path = os.path.join(DATA, "investor_features_by_snapshot.parquet")
    inv_by_snap.to_parquet(snap_path, index=False)
    print(f"\nSaved investor_features_by_snapshot.parquet "
          f"({len(inv_by_snap):,} node-snapshot rows)")

    # ── 4. Startup-level network features ─────────────────────────────────────
    print("\nComputing startup-level network features ...")

    fp = firms_panel[["company_name", "first_deal_date", "snapshot_date"]]
    die_fp = deals.merge(fp, on="company_name", how="inner")
    die_fp = die_fp[die_fp["investment_date"] <= die_fp["first_deal_date"]]
    die_fp = die_fp.dropna(subset=["investor_org_id"])
    die_fp = die_fp.drop_duplicates(subset=["company_name", "investor_org_id"])
    print(f"  Startup-investor pairs at anchor : {len(die_fp):,}")

    die_fp = die_fp.merge(
        inv_by_snap[["investor_org_id", "snapshot_date"] + METRIC_COLS
                    + ["pagerank_top_decile_threshold"]],
        on=["investor_org_id", "snapshot_date"], how="left",
    )
    die_fp["known_at_snapshot"] = die_fp["degree"].notna()
    die_fp["is_top_pagerank"] = (
        die_fp["pagerank"] >= die_fp["pagerank_top_decile_threshold"]
    ).fillna(False)

    startup_features = (
        die_fp.groupby("company_name")
        .agg(
            n_investors_first_round   =("investor_org_id", "nunique"),
            n_known_investors         =("known_at_snapshot", "sum"),
            mean_degree               =("degree", "mean"),
            max_degree                =("degree", "max"),
            mean_betweenness          =("betweenness_centrality", "mean"),
            max_betweenness           =("betweenness_centrality", "max"),
            mean_pagerank             =("pagerank", "mean"),
            max_pagerank              =("pagerank", "max"),
            mean_eigenvector_lcc      =("eigenvector_lcc", "mean"),
            max_eigenvector_lcc       =("eigenvector_lcc", "max"),
            mean_clustering           =("clustering_coeff", "mean"),
            has_top_pagerank_investor =("is_top_pagerank", "max"),
        )
        .reset_index()
    )
    startup_features["share_newcomer_investors"] = (
        1.0 - startup_features["n_known_investors"]
        / startup_features["n_investors_first_round"]
    )
    startup_features["has_top_pagerank_investor"] = (
        startup_features["has_top_pagerank_investor"].astype(int)
    )

    firms_net = firms_panel.merge(startup_features, on="company_name", how="left")

    n_any   = firms_net["n_investors_first_round"].notna().sum()
    n_known = (firms_net["n_known_investors"].fillna(0) > 0).sum()
    print(f"  Startups with ≥1 investor row            : {n_any:,} / {len(firms_net):,}")
    print(f"  Startups with ≥1 investor known at snap  : {n_known:,} / {len(firms_net):,}")
    print(f"\nNetwork feature summary:")
    print(firms_net[["mean_degree", "max_degree", "mean_pagerank", "max_pagerank",
                     "share_newcomer_investors", "n_investors_first_round"]]
          .describe().round(6).to_string())

    out_path = os.path.join(DATA, "firms_with_network_T7.parquet")
    firms_net.to_parquet(out_path, index=False)
    print(f"\nSaved firms_with_network_T7.parquet ({firms_net.shape})")


if __name__ == "__main__":
    main()
