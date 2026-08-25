"""
09_train_gnn_model.py — clean-graph benchmark for the heterogeneous
bipartite investor-startup GNN.

Trains HeteroGNN on the train graph (built by script 08), selects
hyperparameters by validation ROC-AUC, and scores once on the test graph.
Output contract matches script 05 (network_results.csv/json,
calibration_*.png, saved model) so script 14 can ingest it with minimal
changes.

This is a clean-graph BENCHMARK, run separately from and preserving the
existing tabular network model (script 05) untouched, per
ROBUSTNESS_REFRAMING_PLAN.md requirement E ("preserve the original
clean-graph benchmark, wrap don't replace").
"""

import os
import sys
import json
import copy
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import T_YEARS, SEED, set_global_seed, evaluate_classifier, tune_threshold
from gnn_model import HeteroGNN, materialize

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(ROOT, "Data", "processed")
GRAPH_DIR = os.path.join(DATA, "graphs")
OUT_DIR   = os.path.join(DATA, "models", f"T{T_YEARS}")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_EPOCHS      = 200
EARLY_STOP_PATIENCE = 20
HIDDEN_DIMS = [16, 32, 64]
DROPOUTS    = [0.2, 0.4]
LRS         = [1e-3, 5e-3]


def train_one_config(train_data, val_data, hidden_dim, dropout, lr, pos_weight):
    torch.manual_seed(SEED)
    model = HeteroGNN(train_data.edge_types, hidden_dim=hidden_dim, dropout=dropout)
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


def evaluate_model(model, val_data, test_data, tag, out_dir):
    """GNN-specific extraction (torch model + HeteroData -> y, prob), then
    delegates metric computation + calibration plotting to the shared
    pipeline_core.evaluate_classifier (same one 03/05 use). Threshold is
    tuned on validation predictions (tune_threshold) rather than fixed at
    0.5, same convention as 03/05's evaluate_with_tuned_threshold."""
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(val_data.x_dict, val_data.edge_index_dict)).numpy()
        test_prob = torch.sigmoid(model(test_data.x_dict, test_data.edge_index_dict)).numpy()
    y_val = val_data["startup"].y.numpy()
    y_test = test_data["startup"].y.numpy()
    threshold, val_f1 = tune_threshold(y_val, val_prob)
    result = evaluate_classifier(tag, y_test, test_prob, split="Test", out_dir=out_dir,
                                 threshold=threshold)
    result["val_threshold_f1"] = val_f1
    return result


def main():
    set_global_seed()

    print(f"{'='*60}\n  HETEROGENEOUS GNN MODEL (T={T_YEARS}y)\n{'='*60}")

    train_data = torch.load(os.path.join(GRAPH_DIR, "train_graph.pt"), weights_only=False)
    val_data   = torch.load(os.path.join(GRAPH_DIR, "val_graph.pt"), weights_only=False)
    test_data  = torch.load(os.path.join(GRAPH_DIR, "test_graph.pt"), weights_only=False)

    print(f"Train: {train_data['startup'].num_nodes:,} startups, "
          f"{train_data['investor'].num_nodes:,} investors")
    print(f"Val:   {val_data['startup'].num_nodes:,} startups, "
          f"{val_data['investor'].num_nodes:,} investors")
    print(f"Test:  {test_data['startup'].num_nodes:,} startups, "
          f"{test_data['investor'].num_nodes:,} investors")

    y_tr = train_data["startup"].y.numpy()
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class balance (train): neg={n_neg}, pos={n_pos}, pos_weight={pos_weight:.3f}")

    grid = [{"hidden_dim": h, "dropout": d, "lr": lr}
            for h in HIDDEN_DIMS for d in DROPOUTS for lr in LRS]
    print(f"\nHyperparameter grid: {len(grid)} combinations")

    best_overall_auc = -1.0
    best_params = None
    best_state = None
    for params in grid:
        val_auc, state, n_epochs = train_one_config(
            train_data, val_data, pos_weight=pos_weight, **params)
        print(f"  hidden_dim={params['hidden_dim']:<3} dropout={params['dropout']} "
              f"lr={params['lr']}  val_AUC={val_auc:.4f}  (stopped after {n_epochs} epochs)")
        if val_auc > best_overall_auc:
            best_overall_auc = val_auc
            best_params = params
            best_state = state

    print(f"\nBest config: {best_params}  val_AUC={best_overall_auc:.4f}")

    model = HeteroGNN(train_data.edge_types, hidden_dim=best_params["hidden_dim"],
                      dropout=best_params["dropout"])
    materialize(model, train_data)
    model.load_state_dict(best_state)

    result = evaluate_model(model, val_data, test_data, "GNN_Network", OUT_DIR)
    result["best_params"] = best_params
    result["val_roc_auc"] = round(float(best_overall_auc), 4)

    with open(os.path.join(OUT_DIR, "gnn_results.json"), "w") as f:
        json.dump([result], f, indent=2)
    pd.DataFrame([result]).to_csv(os.path.join(OUT_DIR, "gnn_results.csv"), index=False)

    torch.save({"model_state": best_state, "params": best_params,
               "edge_types": train_data.edge_types},
              os.path.join(OUT_DIR, "gnn_model.pt"))

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
