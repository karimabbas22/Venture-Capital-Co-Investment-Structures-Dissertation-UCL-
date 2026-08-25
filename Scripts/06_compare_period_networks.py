"""
06_compare_period_networks.py — temporal extension: compare co-investment
network structure across two broad periods.

Descriptive analysis showing how the VC co-investment landscape evolved.
"""

import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    PERIOD_1, PERIOD_2, SEED, BETWEENNESS_K,
    build_coinvestment_snapshot, compute_snapshot_metrics, set_global_seed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "processed")


def graph_summary(G: nx.Graph, label: str) -> dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return {"period": label, "nodes": 0, "edges": 0}

    degrees = [d for _, d in G.degree()]
    cc = list(nx.connected_components(G))
    lcc_size = max(len(c) for c in cc)

    return {
        "period": label,
        "nodes": n,
        "edges": e,
        "density": round(nx.density(G), 6),
        "n_components": len(cc),
        "lcc_size": lcc_size,
        "lcc_pct": round(100 * lcc_size / n, 1),
        "mean_degree": round(np.mean(degrees), 2),
        "median_degree": round(np.median(degrees), 1),
        "max_degree": max(degrees),
        "mean_clustering": round(nx.average_clustering(G), 4),
    }


def top_investors(metrics: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    return (metrics.sort_values(col, ascending=False)
            .head(n)[["investor_org_id", "investor_name", col]]
            .reset_index(drop=True))


def main():
    set_global_seed()

    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(
        deals["investor_name"]).astype(str)

    print(f"{'='*70}")
    print(f"  TEMPORAL NETWORK COMPARISON")
    print(f"{'='*70}")
    print(f"  Period 1: {PERIOD_1[0].date()} – {PERIOD_1[1].date()}")
    print(f"  Period 2: {PERIOD_2[0].date()} – {PERIOD_2[1].date()}")

    # Build one graph per period
    results = []
    metrics_by_period = {}
    for label, (start, end) in [("Period_1", PERIOD_1), ("Period_2", PERIOD_2)]:
        print(f"\nBuilding graph for {label} ...")
        # For period-specific: only deals within the period
        sub = deals[(deals["investment_date"] >= start) &
                    (deals["investment_date"] <= end)]
        G_period = build_coinvestment_snapshot(sub, end)
        m = compute_snapshot_metrics(G_period, betweenness_k=BETWEENNESS_K, seed=SEED)
        metrics_by_period[label] = m
        summary = graph_summary(G_period, label)
        results.append(summary)
        print(f"  {summary}")

    # Comparison table
    comp = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("  STRUCTURAL COMPARISON")
    print(f"{'='*70}")
    print(comp.to_string(index=False))

    # Growth metrics
    r1, r2 = results[0], results[1]
    if r1["nodes"] > 0 and r2["nodes"] > 0:
        print(f"\n  Node growth: {r1['nodes']:,} → {r2['nodes']:,} "
              f"({100*(r2['nodes']-r1['nodes'])/r1['nodes']:+.0f}%)")
        print(f"  Edge growth: {r1['edges']:,} → {r2['edges']:,} "
              f"({100*(r2['edges']-r1['edges'])/r1['edges']:+.0f}%)")
        print(f"  Density change: {r1['density']:.6f} → {r2['density']:.6f}")
        print(f"  Mean degree: {r1['mean_degree']:.1f} → {r2['mean_degree']:.1f}")
        print(f"  Clustering: {r1['mean_clustering']:.4f} → {r2['mean_clustering']:.4f}")

    # Top investors per period
    for label, m in metrics_by_period.items():
        if len(m) == 0:
            continue
        print(f"\n  Top 10 investors by PageRank ({label}):")
        print(top_investors(m, "pagerank").to_string(index=False))

    # Investor overlap between periods
    if all(len(m) > 0 for m in metrics_by_period.values()):
        ids_1 = set(metrics_by_period["Period_1"]["investor_org_id"])
        ids_2 = set(metrics_by_period["Period_2"]["investor_org_id"])
        overlap = ids_1 & ids_2
        print(f"\n  Investor overlap:")
        print(f"    Period 1 only: {len(ids_1 - ids_2):,}")
        print(f"    Period 2 only: {len(ids_2 - ids_1):,}")
        print(f"    Both periods:  {len(overlap):,}")
        print(f"    Jaccard:       {len(overlap)/len(ids_1 | ids_2):.3f}")

    # Save
    comp.to_csv(os.path.join(DATA, "period_network_comparison.csv"), index=False)
    for label, m in metrics_by_period.items():
        m.to_parquet(os.path.join(DATA, f"investor_features_{label.lower()}.parquet"),
                     index=False)

    print(f"\nSaved period_network_comparison.csv")
    print(f"Saved investor_features_period_1.parquet / period_2.parquet")


if __name__ == "__main__":
    main()
