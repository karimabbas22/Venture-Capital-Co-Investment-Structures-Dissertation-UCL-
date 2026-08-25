"""
graph_core.py — bipartite investor-startup HeteroData construction.

Builds one disjoint torch_geometric.data.HeteroData graph per temporal
split (train/val/test), each cut at its split's boundary date. Node types:
`startup` and `investor`. Edge types: ('startup','invests_in','investor')
+ auto-generated reverse via ToUndirected(), and
('investor','co_invests_with','investor') reused from the existing
investor-investor co-investment graph (pipeline_core.build_coinvestment_snapshot).

Leakage discipline: 'invests_in' edges for a given startup are restricted
to deals with investment_date <= that startup's own anchor (first_deal_date)
-- not the split cutoff -- so no startup's own post-anchor funding rounds
leak into its graph neighborhood. Since anchor = first investment date by
definition, this naturally restricts to first-round syndicate deals only,
matching the scope of the existing tabular network model.

The co-investment ('co_invests_with') relation is built as a single
snapshot at the split's cutoff date -- a documented simplification vs. the
tabular pipeline's per-startup quarterly snapshot assignment. This trade-off
is accepted because a per-startup snapshot graph would require rebuilding
the co-investment network once per startup, which is computationally
infeasible at this dataset's scale within a GNN training loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from pipeline_core import (
    build_coinvestment_snapshot, compute_snapshot_metrics, BETWEENNESS_K, SEED,
    fit_preprocessor, transform_preprocessor,
    BASELINE_NUM_COLS as STARTUP_NUM_COLS,
    CAT_FEATURE_COLS as STARTUP_CAT_COLS,
    engineer_common_covariates as engineer_startup_features,
)

INVESTOR_NUM_COLS = ["degree", "weighted_degree", "betweenness_centrality",
                     "pagerank", "eigenvector_lcc", "clustering_coeff"]


def build_hetero_graph(deals: pd.DataFrame, startups: pd.DataFrame,
                       cutoff_date: pd.Timestamp,
                       startup_preprocessor=None, investor_preprocessor=None,
                       fit_preprocessors: bool = False):
    """
    Build one HeteroData graph for a set of startups belonging to a single
    temporal split, using the investor-investor co-investment graph as of
    `cutoff_date`.

    If fit_preprocessors=True, fits new preprocessors on this split's data
    (use only for the train split) and returns
    (graph, startup_preprocessor, investor_preprocessor).
    Otherwise startup_preprocessor/investor_preprocessor must already be
    fit (on train) and are applied via .transform() only.
    """
    startups = engineer_startup_features(startups).reset_index(drop=True)

    # Investor-investor structure "as of" the split cutoff.
    G = build_coinvestment_snapshot(deals, cutoff_date)
    inv_metrics = compute_snapshot_metrics(
        G, betweenness_k=BETWEENNESS_K, seed=SEED).reset_index(drop=True)
    # eigenvector_lcc is NaN for investors outside the largest connected
    # component (mirrors the tabular pipeline's missing_evc case) -- impute
    # before fitting/transforming, or NaNs propagate through the whole GNN.
    inv_metrics[INVESTOR_NUM_COLS] = inv_metrics[INVESTOR_NUM_COLS].fillna(0.0)
    investor_ids = inv_metrics["investor_org_id"].tolist()
    investor_idx = {oid: i for i, oid in enumerate(investor_ids)}

    startup_ids = list(zip(startups["company_name"], startups["company_permid"]))
    startup_idx = {sid: i for i, sid in enumerate(startup_ids)}

    # 'invests_in' edges: each startup's own first-round deals only.
    # deals.parquet has its own company_permid column -- drop it before the
    # merge so the startup-side identity (the correct one) isn't suffixed.
    d = deals.drop(columns=["company_permid"], errors="ignore").merge(
        startups[["company_name", "company_permid", "first_deal_date"]],
        on="company_name", how="inner")
    d = d[d["investment_date"] <= d["first_deal_date"]]
    d = d.drop_duplicates(subset=["company_name", "company_permid", "investor_org_id"])

    src, dst = [], []
    for row in d.itertuples(index=False):
        sid = (row.company_name, row.company_permid)
        oid = row.investor_org_id
        if sid in startup_idx and oid in investor_idx:
            src.append(startup_idx[sid])
            dst.append(investor_idx[oid])

    data = HeteroData()
    data["startup"].num_nodes = len(startup_idx)
    data["investor"].num_nodes = len(investor_idx)

    if fit_preprocessors:
        startup_preprocessor = fit_preprocessor(startups, STARTUP_NUM_COLS, [], STARTUP_CAT_COLS)
        investor_preprocessor = fit_preprocessor(inv_metrics, INVESTOR_NUM_COLS, [], [])

    X_startup = transform_preprocessor(startup_preprocessor, startups)
    X_investor = transform_preprocessor(investor_preprocessor, inv_metrics)

    data["startup"].x = torch.tensor(np.asarray(X_startup), dtype=torch.float)
    data["investor"].x = torch.tensor(np.asarray(X_investor), dtype=torch.float)

    if "exit_within_T" in startups.columns:
        data["startup"].y = torch.tensor(startups["exit_within_T"].values, dtype=torch.float)

    edge_index = (torch.tensor([src, dst], dtype=torch.long)
                 if src else torch.zeros((2, 0), dtype=torch.long))
    data["startup", "invests_in", "investor"].edge_index = edge_index

    co_src, co_dst = [], []
    for u, v in G.edges():
        if u in investor_idx and v in investor_idx:
            co_src.append(investor_idx[u])
            co_dst.append(investor_idx[v])
    co_edge_index = (torch.tensor([co_src, co_dst], dtype=torch.long)
                     if co_src else torch.zeros((2, 0), dtype=torch.long))
    data["investor", "co_invests_with", "investor"].edge_index = co_edge_index

    data = ToUndirected()(data)

    # Identity mappings for downstream evaluation / perturbation (not used
    # by PyG internals, just carried along on the object).
    data.startup_ids = startup_ids
    data.investor_ids = investor_ids

    if fit_preprocessors:
        return data, startup_preprocessor, investor_preprocessor
    return data


def remove_investor_nodes(data: HeteroData, remove_local_idx: set[int]) -> HeteroData:
    """Return a new HeteroData with the given investor node local indices
    removed entirely (whole-node hub removal), reindexing every edge type
    that references 'investor' (invests_in, rev_invests_in, co_invests_with).
    Startup nodes/features/labels are untouched. Used by script 10's GNN
    hub-removal perturbation, mirroring script 07's perturb_graph_hub_removal
    for the tabular pipeline."""
    num_investor = data["investor"].num_nodes
    keep_mask = torch.ones(num_investor, dtype=torch.bool)
    for i in remove_local_idx:
        keep_mask[i] = False
    old_to_new = -torch.ones(num_investor, dtype=torch.long)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum()))

    new_data = HeteroData()
    new_data["startup"].num_nodes = data["startup"].num_nodes
    new_data["startup"].x = data["startup"].x
    if hasattr(data["startup"], "y"):
        new_data["startup"].y = data["startup"].y
    new_data["investor"].num_nodes = int(keep_mask.sum())
    new_data["investor"].x = data["investor"].x[keep_mask]

    ei = data["startup", "invests_in", "investor"].edge_index
    edge_keep = keep_mask[ei[1]]
    new_data["startup", "invests_in", "investor"].edge_index = torch.stack(
        [ei[0][edge_keep], old_to_new[ei[1][edge_keep]]])

    ei = data["investor", "rev_invests_in", "startup"].edge_index
    edge_keep = keep_mask[ei[0]]
    new_data["investor", "rev_invests_in", "startup"].edge_index = torch.stack(
        [old_to_new[ei[0][edge_keep]], ei[1][edge_keep]])

    ei = data["investor", "co_invests_with", "investor"].edge_index
    edge_keep = keep_mask[ei[0]] & keep_mask[ei[1]]
    new_data["investor", "co_invests_with", "investor"].edge_index = torch.stack(
        [old_to_new[ei[0][edge_keep]], old_to_new[ei[1][edge_keep]]])

    return new_data
