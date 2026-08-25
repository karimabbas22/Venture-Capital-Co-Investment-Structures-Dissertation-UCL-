"""
Experiments/gnn_ablations.py -- three architecture ablations for the
heterogeneous bipartite GNN, addressing a direct methodological question
about the RQ: is the GNN's (statistically insignificant, but nominally
highest) clean-graph AUC edge actually coming from the co-investment
network structure, or from something else?

1. EDGE-TYPE ABLATION (the one that matters most): train with only the
   bipartite edges (invests_in / rev_invests_in), dropping co_invests_with
   entirely. If clean-graph AUC barely changes without the co-investment
   edges, the GNN's edge over tabular models isn't coming from the network
   structure specifically -- it could just as easily come from the
   startup-side features alone, routed through a message-passing
   architecture instead of a linear/tree model.

   Implementation note: this does NOT touch the saved graph .pt files or
   gnn_model.py. torch_geometric.nn.HeteroConv.forward() only iterates
   edge types present in ITS OWN `convs` dict (verified against the
   installed library source before writing this script) -- any extra key
   in the edge_index_dict passed at call time that has no matching conv is
   silently skipped. So restricting HeteroGNN's `edge_types` constructor
   argument to the two bipartite relations is sufficient; the original,
   unmodified train/val/test HeteroData objects are reused as-is.

2. INVESTOR NODE-FEATURE ABLATION (the natural pair to #1): keep all three
   relations, but replace investor nodes' 6 centrality-metric features
   with a constant (all-ones) vector, so the model can only use graph
   TOPOLOGY via message passing, not the hand-computed centrality stats
   riding on top of it. Startup features are untouched. Run on in-memory
   deep copies of the saved graphs -- the saved .pt files are never
   overwritten.

3. LAYER-DEPTH ABLATION: a 1-layer variant (drop conv2), defined as a new
   class LOCAL to this script (gnn_model.py is not modified) that reuses
   the same HeteroConv/SAGEConv building blocks and the same
   materialize()-compatible forward signature.

All three reuse the EXACT training protocol from Scripts/09_train_gnn_model.py
(same 12-point hyperparameter grid, same early-stopping patience, same
seed-reset-per-config, same optimizer/loss config) so the comparison to
the production GNN (0.6582 clean-graph test ROC-AUC) is apples-to-apples,
not handicapped by a smaller search. Each ablation's test predictions are
compared to the production GNN's test predictions via DeLong's test
(reusing Experiments/significance_tests.delong_test unmodified), on the
same n=3,280 test startups.

Standalone and read-only with respect to Scripts/ and all saved production
artifacts (graphs, gnn_model.pt) -- nothing here overwrites them.
Run: python3 Experiments/gnn_ablations.py
"""

import os
import sys
import copy
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost  # noqa: F401 -- import before torch, see significance_tests.py (avoids a macOS OpenMP segfault)
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
from significance_tests import delong_test  # noqa: E402

SCRIPTS_DIR = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
from pipeline_core import T_YEARS, SEED, set_global_seed  # noqa: E402
from gnn_model import HeteroGNN, materialize  # noqa: E402

ROOT      = os.path.dirname(SCRIPTS_DIR)
DATA      = os.path.join(ROOT, "Data", "processed")
GRAPH_DIR = os.path.join(DATA, "graphs")
MODEL_DIR = os.path.join(DATA, "models", f"T{T_YEARS}")
OUT_DIR   = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

COINV_RELATION = ("investor", "co_invests_with", "investor")

MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
HIDDEN_DIMS = [16, 32, 64]
DROPOUTS = [0.2, 0.4]
LRS = [1e-3, 5e-3]


class HeteroGNN1Layer(torch.nn.Module):
    """1-layer variant of gnn_model.HeteroGNN (conv2 dropped), defined
    locally so gnn_model.py itself is never touched. Same conv1 config,
    same forward-signature contract materialize() expects."""

    def __init__(self, edge_types, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="mean")
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in x_dict.items()}
        return self.head(x_dict["startup"]).squeeze(-1)


def train_one_config(model_ctor, train_data, val_data, lr, pos_weight):
    """Verbatim protocol from 09_train_gnn_model.py's train_one_config,
    generalized over an arbitrary zero-arg model constructor."""
    torch.manual_seed(SEED)
    model = model_ctor()
    materialize(model, train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data.x_dict, train_data.edge_index_dict)
        loss = loss_fn(logits, train_data["startup"].y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_data.x_dict, val_data.edge_index_dict)
            val_prob = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(val_data["startup"].y.numpy(), val_prob)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    return best_val_auc, best_state, epoch + 1


def run_grid(build_model, train_data, val_data, pos_weight, label):
    grid = [{"hidden_dim": h, "dropout": d, "lr": lr}
            for h in HIDDEN_DIMS for d in DROPOUTS for lr in LRS]
    best_overall_auc, best_params, best_state = -1.0, None, None
    for params in grid:
        ctor = lambda hd=params["hidden_dim"], dr=params["dropout"]: build_model(hd, dr)
        val_auc, state, n_epochs = train_one_config(
            ctor, train_data, val_data, lr=params["lr"], pos_weight=pos_weight)
        print(f"  [{label}] hidden_dim={params['hidden_dim']:<3} dropout={params['dropout']} "
              f"lr={params['lr']}  val_AUC={val_auc:.4f}  (stopped after {n_epochs} epochs)")
        if val_auc > best_overall_auc:
            best_overall_auc, best_params, best_state = val_auc, params, state
    print(f"[{label}] BEST config: {best_params}  val_AUC={best_overall_auc:.4f}")
    return best_state, best_params, best_overall_auc


def build_final_model(model_cls, edge_types, best_params, best_state, train_data_for_materialize):
    model = model_cls(edge_types, hidden_dim=best_params["hidden_dim"], dropout=best_params["dropout"])
    materialize(model, train_data_for_materialize)
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, data):
    with torch.no_grad():
        return torch.sigmoid(model(data.x_dict, data.edge_index_dict)).numpy()


def evaluate_and_compare(name, test_prob, y_test, full_model_prob):
    roc = roc_auc_score(y_test, test_prob)
    pr = average_precision_score(y_test, test_prob)
    full_roc = roc_auc_score(y_test, full_model_prob)
    auc_a, auc_b, z, p = delong_test(y_test, test_prob, full_model_prob)
    row = {
        "ablation": name,
        "test_roc_auc": round(float(roc), 4),
        "test_pr_auc": round(float(pr), 4),
        "full_model_roc_auc": round(float(full_roc), 4),
        "delta_roc_auc": round(float(roc - full_roc), 4),
        "delong_z": round(float(z), 4),
        "delong_p": round(float(p), 4),
        "significant_at_05": bool(p < 0.05),
    }
    print(f"\n[{name}] test ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  "
          f"(full model: {full_roc:.4f})  Delta={roc-full_roc:+.4f}  DeLong p={p:.4f}")
    return row


def main():
    set_global_seed()
    print(f"{'='*70}\n  GNN ABLATIONS: edge-type, node-feature, layer-depth\n{'='*70}")

    train_data = torch.load(os.path.join(GRAPH_DIR, "train_graph.pt"), weights_only=False)
    val_data   = torch.load(os.path.join(GRAPH_DIR, "val_graph.pt"), weights_only=False)
    test_data  = torch.load(os.path.join(GRAPH_DIR, "test_graph.pt"), weights_only=False)

    y_tr = train_data["startup"].y.numpy()
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    y_test = test_data["startup"].y.numpy()

    # ── Load the production GNN for the baseline comparison ────────────────
    bundle = torch.load(os.path.join(MODEL_DIR, "gnn_model.pt"), weights_only=False)
    full_model = HeteroGNN(bundle["edge_types"], hidden_dim=bundle["params"]["hidden_dim"],
                           dropout=bundle["params"]["dropout"])
    materialize(full_model, test_data)
    full_model.load_state_dict(bundle["model_state"])
    full_model.eval()
    full_model_prob = predict(full_model, test_data)
    print(f"Production (3-relation) GNN clean test ROC-AUC: "
          f"{roc_auc_score(y_test, full_model_prob):.4f}  (reference for all DeLong tests below)")

    results = []

    # ═══════════════════════════════════════════════════════════════════
    # 1. EDGE-TYPE ABLATION -- bipartite edges only, co_invests_with dropped
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'-'*70}\n  1. EDGE-TYPE ABLATION (bipartite only, no co_invests_with)\n{'-'*70}")
    bipartite_edge_types = [et for et in train_data.edge_types if et != COINV_RELATION]
    print(f"Edge types used: {bipartite_edge_types}")

    build1 = lambda hd, dr: HeteroGNN(bipartite_edge_types, hidden_dim=hd, dropout=dr)
    state1, params1, val_auc1 = run_grid(build1, train_data, val_data, pos_weight, "edge-type")
    model1 = build_final_model(HeteroGNN, bipartite_edge_types, params1, state1, train_data)
    prob1 = predict(model1, test_data)
    row1 = evaluate_and_compare("edge_type_bipartite_only", prob1, y_test, full_model_prob)
    row1["best_params"] = json.dumps(params1)
    row1["val_roc_auc"] = round(float(val_auc1), 4)
    results.append(row1)

    # ═══════════════════════════════════════════════════════════════════
    # 2. INVESTOR NODE-FEATURE ABLATION -- constant investor features
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'-'*70}\n  2. INVESTOR NODE-FEATURE ABLATION (constant investor.x, topology only)\n{'-'*70}")
    train_data2 = copy.deepcopy(train_data)
    val_data2   = copy.deepcopy(val_data)
    test_data2  = copy.deepcopy(test_data)
    for d in (train_data2, val_data2, test_data2):
        d["investor"].x = torch.ones_like(d["investor"].x)
    print(f"Investor feature shape unchanged ({train_data2['investor'].x.shape}), "
          f"values replaced with constant 1.0 (startup features untouched)")

    build2 = lambda hd, dr: HeteroGNN(train_data.edge_types, hidden_dim=hd, dropout=dr)
    state2, params2, val_auc2 = run_grid(build2, train_data2, val_data2, pos_weight, "node-feature")
    model2 = build_final_model(HeteroGNN, train_data.edge_types, params2, state2, train_data2)
    prob2 = predict(model2, test_data2)
    row2 = evaluate_and_compare("investor_features_constant", prob2, y_test, full_model_prob)
    row2["best_params"] = json.dumps(params2)
    row2["val_roc_auc"] = round(float(val_auc2), 4)
    results.append(row2)

    # ═══════════════════════════════════════════════════════════════════
    # 3. LAYER-DEPTH ABLATION -- 1-layer (conv2 dropped)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'-'*70}\n  3. LAYER-DEPTH ABLATION (1-layer, conv2 dropped)\n{'-'*70}")
    build3 = lambda hd, dr: HeteroGNN1Layer(train_data.edge_types, hidden_dim=hd, dropout=dr)
    state3, params3, val_auc3 = run_grid(build3, train_data, val_data, pos_weight, "layer-depth")
    model3 = build_final_model(HeteroGNN1Layer, train_data.edge_types, params3, state3, train_data)
    prob3 = predict(model3, test_data)
    row3 = evaluate_and_compare("one_layer", prob3, y_test, full_model_prob)
    row3["best_params"] = json.dumps(params3)
    row3["val_roc_auc"] = round(float(val_auc3), 4)
    results.append(row3)

    # ── Save ─────────────────────────────────────────────────────────────
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "gnn_ablations_results.csv"), index=False)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(res[["ablation", "test_roc_auc", "test_pr_auc", "full_model_roc_auc",
              "delta_roc_auc", "delong_p", "significant_at_05"]].to_string(index=False))
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'gnn_ablations_results.csv')}")


if __name__ == "__main__":
    main()
