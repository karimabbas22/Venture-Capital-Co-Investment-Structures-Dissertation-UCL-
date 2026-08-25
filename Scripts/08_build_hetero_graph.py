"""
08_build_hetero_graph.py — build the three disjoint train/val/test
bipartite investor-startup HeteroData graphs.

Outputs → Data/processed/graphs/
    train_graph.pt, val_graph.pt, test_graph.pt   (torch.save'd HeteroData)
    preprocessors.pt                              (fit-on-train sklearn preprocessors)
"""

import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    T_YEARS, TRAIN_END, VAL_END, ENTRY_END, DATA_END_DATE,
    temporal_train_val_test_split, set_global_seed,
)
from graph_core import build_hetero_graph

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(ROOT, "Data", "processed")
GRAPH_DIR = os.path.join(DATA, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

LABEL_COL = "exit_within_T"
DATE_COL  = "first_deal_date"


def graph_diagnostics(name: str, data) -> None:
    n_startup = data["startup"].num_nodes
    n_investor = data["investor"].num_nodes
    n_invests = data["startup", "invests_in", "investor"].edge_index.shape[1]
    n_co = data["investor", "co_invests_with", "investor"].edge_index.shape[1]
    pos = int(data["startup"].y.sum().item()) if hasattr(data["startup"], "y") else "n/a"
    print(f"  [{name}] startup_nodes={n_startup:,}  investor_nodes={n_investor:,}  "
          f"invests_in_edges={n_invests:,}  co_invests_with_edges={n_co:,}  "
          f"exit=1: {pos}")


def main():
    set_global_seed()

    print(f"{'='*70}\n  BUILD HETEROGENEOUS BIPARTITE GRAPH  (T={T_YEARS}y)\n{'='*70}")

    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(
        deals["investor_name"]).astype(str)

    firms = pd.read_parquet(os.path.join(DATA, "firms_panel.parquet"))
    firms = firms.dropna(subset=[LABEL_COL, DATE_COL]).copy()
    assert (firms["right_censored"] == 0).all(), "censored rows reached graph construction"
    assert (firms["observation_window_end"] <= DATA_END_DATE).all()

    train_df, val_df, test_df = temporal_train_val_test_split(firms, TRAIN_END, VAL_END)
    print(f"\nSplit sizes: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Leakage check: disjoint startup sets across splits (structural, not
    # just by anchor-date filter, since the graph objects are separate).
    ids_tr = set(zip(train_df["company_name"], train_df["company_permid"]))
    ids_va = set(zip(val_df["company_name"], val_df["company_permid"]))
    ids_te = set(zip(test_df["company_name"], test_df["company_permid"]))
    assert not (ids_tr & ids_va) and not (ids_tr & ids_te) and not (ids_va & ids_te), \
        "startup identity overlap across splits — leakage risk"
    print("  Leakage check passed: zero startup-identity overlap across train/val/test")

    print("\nBuilding train graph (cutoff = TRAIN_END, fitting preprocessors)...")
    train_graph, startup_pre, investor_pre = build_hetero_graph(
        deals, train_df, TRAIN_END, fit_preprocessors=True)
    graph_diagnostics("train", train_graph)

    print("\nBuilding val graph (cutoff = VAL_END, using train preprocessors)...")
    val_graph = build_hetero_graph(
        deals, val_df, VAL_END,
        startup_preprocessor=startup_pre, investor_preprocessor=investor_pre)
    graph_diagnostics("val", val_graph)

    print("\nBuilding test graph (cutoff = ENTRY_END, using train preprocessors)...")
    test_graph = build_hetero_graph(
        deals, test_df, ENTRY_END,
        startup_preprocessor=startup_pre, investor_preprocessor=investor_pre)
    graph_diagnostics("test", test_graph)

    # Second leakage check: verify the actual saved node identity lists
    # (not just the source dataframes) are disjoint.
    assert not (set(train_graph.startup_ids) & set(val_graph.startup_ids))
    assert not (set(train_graph.startup_ids) & set(test_graph.startup_ids))
    assert not (set(val_graph.startup_ids) & set(test_graph.startup_ids))
    print("\n  Leakage check passed on graph objects: zero startup-node overlap")

    torch.save(train_graph, os.path.join(GRAPH_DIR, "train_graph.pt"))
    torch.save(val_graph, os.path.join(GRAPH_DIR, "val_graph.pt"))
    torch.save(test_graph, os.path.join(GRAPH_DIR, "test_graph.pt"))
    torch.save({"startup_preprocessor": startup_pre, "investor_preprocessor": investor_pre},
              os.path.join(GRAPH_DIR, "preprocessors.pt"))

    print(f"\nAll outputs → {GRAPH_DIR}")


if __name__ == "__main__":
    main()
