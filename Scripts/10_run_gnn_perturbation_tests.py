"""
10_run_gnn_perturbation_tests.py — perturbation robustness tests for the
heterogeneous bipartite GNN (script 09), mirroring script 07's protocol
for the tabular network models. Uses the same experiment_config.json and
stability_metrics.evaluate_stability() output schema, so results merge
cleanly with script 07's for the combined robustness report (script 11).

Perturbation protocol now matches script 07's exactly: after perturbing
the co_invests_with edges (or removing hub nodes), investor node features
are recomputed from the perturbed graph -- the 6 raw centrality metrics
are recalculated and pushed through the already-fitted (train-only)
investor preprocessor via perturbation_core.recompute_investor_features()
-- before re-scoring. Earlier versions of this script left investor
features frozen at their clean-graph values after a topology-only
perturbation, which meant GNN and tabular robustness magnitudes were not
directly comparable (only the qualitative pattern was). This is fixed
here, so GNN-vs-tabular numbers can now be compared directly, not just
by pattern.

node_feature_noise is GNN-exclusive (see perturbation_core.py) -- the
tabular pipeline's "features" are graph-derived downstream, not stored
node attributes. The shared `budget` grid (0.5%-10%) is reused directly as
the noise sigma_scale for this perturbation type.
"""

import os
import sys
import json
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import T_YEARS, set_global_seed
from gnn_model import HeteroGNN
from graph_core import remove_investor_nodes
from perturbation_core import (
    perturb_graph, hetero_edges_to_nx, nx_to_hetero_edge_index,
    node_feature_noise, load_experiment_config, recompute_investor_features,
)
from stability_metrics import evaluate_stability

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA        = os.path.join(ROOT, "Data", "processed")
GRAPH_DIR   = os.path.join(DATA, "graphs")
MODEL_DIR   = os.path.join(DATA, "models", f"T{T_YEARS}")
OUT_DIR     = os.path.join(MODEL_DIR, "perturbation")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_config.json")
os.makedirs(OUT_DIR, exist_ok=True)

COINV_RELATION = ("investor", "co_invests_with", "investor")


def predict(model, data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        return torch.sigmoid(logits).numpy()


def perturb_coinvestment_edges(data, ptype: str, budget: float, seed: int, investor_preprocessor):
    pert = copy.deepcopy(data)
    num_investor = data["investor"].num_nodes
    G = hetero_edges_to_nx(data[COINV_RELATION].edge_index, num_investor)
    G_pert = perturb_graph(G, ptype, budget, seed)
    pert[COINV_RELATION].edge_index = nx_to_hetero_edge_index(G_pert)
    pert["investor"].x = recompute_investor_features(G_pert, num_investor, investor_preprocessor)
    return pert


def main():
    set_global_seed()
    cfg = load_experiment_config(CONFIG_PATH)

    print("\nGNN perturbation analysis")
    print(f"Config: budgets={cfg['budgets']}  seeds={cfg['seeds']}  "
          f"edge_types={cfg['perturbation_types']}  hub_k={cfg['hub_removal_k']}")

    test_path = os.path.join(GRAPH_DIR, "test_graph.pt")
    model_path = os.path.join(MODEL_DIR, "gnn_model.pt")
    prep_path = os.path.join(GRAPH_DIR, "preprocessors.pt")
    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print("[ERROR] Missing test_graph.pt or gnn_model.pt — run scripts 08 and 09 first")
        return

    test_data = torch.load(test_path, weights_only=False)
    bundle = torch.load(model_path, weights_only=False)
    investor_preprocessor = torch.load(prep_path, weights_only=False)["investor_preprocessor"]

    model = HeteroGNN(bundle["edge_types"], hidden_dim=bundle["params"]["hidden_dim"],
                      dropout=bundle["params"]["dropout"])
    with torch.no_grad():
        model(test_data.x_dict, test_data.edge_index_dict)  # materialize lazy layers
    model.load_state_dict(bundle["model_state"])

    y_test = test_data["startup"].y.numpy()
    orig_probs = predict(model, test_data)
    print(f"Clean test AUC: {roc_auc_score(y_test, orig_probs):.4f}")
    print(f"Investor nodes: {test_data['investor'].num_nodes:,}  "
          f"co_invests_with edges: {test_data[COINV_RELATION].edge_index.shape[1]:,}")

    all_rows = []
    clean_row = evaluate_stability(y_test, orig_probs, orig_probs,
                                   threshold=cfg["flip_threshold"],
                                   topk_values=tuple(cfg["topk_values"]))
    clean_row.update({"model": "GNN", "perturb_type": "clean", "budget": 0.0, "seed": -1})
    all_rows.append(clean_row)

    # Edge-level perturbations (co_invests_with relation, investor features
    # recomputed from the perturbed graph before re-scoring)
    for ptype in cfg["perturbation_types"]:
        t0 = time.time()
        for budget in cfg["budgets"]:
            for seed in cfg["seeds"]:
                pert_data = perturb_coinvestment_edges(test_data, ptype, budget, seed, investor_preprocessor)
                probs = predict(model, pert_data)
                row = evaluate_stability(y_test, orig_probs, probs,
                                         threshold=cfg["flip_threshold"],
                                         topk_values=tuple(cfg["topk_values"]))
                row.update({"model": "GNN", "perturb_type": ptype,
                           "budget": budget, "seed": seed})
                all_rows.append(row)
        n_scenarios = len(cfg["budgets"]) * len(cfg["seeds"])
        print(f"  {ptype:26s} {n_scenarios} scenarios  ({time.time()-t0:.2f}s)")

    # Node feature noise (GNN-exclusive; budget reused as sigma_scale)
    t0 = time.time()
    for budget in cfg["budgets"]:
        for seed in cfg["seeds"]:
            pert_data = copy.deepcopy(test_data)
            pert_data["investor"].x = node_feature_noise(test_data["investor"].x, budget, seed)
            probs = predict(model, pert_data)
            row = evaluate_stability(y_test, orig_probs, probs,
                                     threshold=cfg["flip_threshold"],
                                     topk_values=tuple(cfg["topk_values"]))
            row.update({"model": "GNN", "perturb_type": "node_feature_noise",
                       "budget": budget, "seed": seed})
            all_rows.append(row)
    n_scenarios = len(cfg["budgets"]) * len(cfg["seeds"])
    print(f"  {'node_feature_noise':26s} {n_scenarios} scenarios  ({time.time()-t0:.2f}s)")

    # Hub removal (K-indexed, not budget-indexed)
    t0 = time.time()
    coinv_edge_index = test_data[COINV_RELATION].edge_index
    degree = torch.bincount(coinv_edge_index[0], minlength=test_data["investor"].num_nodes)
    hub_order = torch.argsort(degree, descending=True).tolist()
    for K in cfg["hub_removal_k"]:
        remove_idx = set(hub_order[:K])
        pert_data = remove_investor_nodes(test_data, remove_idx)
        num_investor_kept = pert_data["investor"].num_nodes
        G_kept = hetero_edges_to_nx(pert_data[COINV_RELATION].edge_index, num_investor_kept)
        pert_data["investor"].x = recompute_investor_features(G_kept, num_investor_kept, investor_preprocessor)
        probs = predict(model, pert_data)
        row = evaluate_stability(y_test, orig_probs, probs,
                                 threshold=cfg["flip_threshold"],
                                 topk_values=tuple(cfg["topk_values"]))
        row.update({"model": "GNN", "perturb_type": "hub_removal",
                   "budget": None, "seed": -1, "hub_k": K})
        all_rows.append(row)
    print(f"  {'hub_removal':26s} {len(cfg['hub_removal_k'])} scenarios  ({time.time()-t0:.2f}s)")

    # Save
    summary_df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "gnn_perturbation_summary.csv")
    summary_df.to_csv(out_path, index=False)

    with open(os.path.join(OUT_DIR, "gnn_perturbation_results.json"), "w") as f:
        json.dump({
            "clean_auc": round(float(roc_auc_score(y_test, orig_probs)), 4),
            "n_test_startups": test_data["startup"].num_nodes,
            "config": cfg,
        }, f, indent=2)

    print("\nGNN perturbation summary (mean over seeds)")
    agg_cols = ["delta_roc_auc", "flip_rate", "mean_abs_delta_p", "spearman_rho",
               "jaccard_top50", "delta_brier"]
    perturbed = summary_df[summary_df["perturb_type"] != "clean"]
    agg = perturbed.groupby(["perturb_type", "budget"], dropna=False)[agg_cols].mean().round(5)
    print(agg.to_string())
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
