"""
11_aggregate_robustness_report.py — aggregate perturbation results into
dissertation-ready tables and a single robustness-curve figure.

Reads Data/processed/models/T7/perturbation/perturbation_summary.csv
(tabular models, script 07) and gnn_perturbation_summary.csv (GNN, script
10) -- both share an identical schema via stability_metrics.evaluate_stability(),
so they concatenate directly -- and writes:
    robustness_summary_wide.csv   one row per (model, perturb_type, budget
                                   or hub_k), mean +/- std over seeds for
                                   every stability metric, across all models
                                   (5 tabular + GNN)
    by_metric/{metric}.csv        one pivot table per metric (budget x model),
                                   for budget-indexed perturbation types only
    robustness_grid.png           one panel per budget-indexed perturbation
                                   type (now 5, incl. preferential_edge_
                                   addition), delta ROC-AUC vs budget, one
                                   line per model (+/- std band)
    robustness_grid_flip_rate.png same layout as above, flip rate vs budget
    hub_removal_curves.png        3 stacked panels (delta ROC-AUC, flip
                                   rate, Spearman rho) vs hub-removal K,
                                   one line per model — this is the dense
                                   hub-removal curve the densified
                                   hub_removal_k grid in experiment_config.json
                                   was built for

Methodological note (also stated in script 10 and the final report): the
GNN's edge/feature perturbations change message-passing topology only --
investor features are not recomputed post-perturbation, unlike script 07's
tabular protocol. GNN vs. tabular robustness numbers in the outputs below
are therefore not a like-for-like architecture comparison.
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import T_YEARS

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA          = os.path.join(ROOT, "Data", "processed")
MODEL_DIR     = os.path.join(DATA, "models", f"T{T_YEARS}")
PERT_DIR      = os.path.join(MODEL_DIR, "perturbation")
BY_METRIC_DIR = os.path.join(PERT_DIR, "by_metric")
os.makedirs(BY_METRIC_DIR, exist_ok=True)

METRICS = ["delta_roc_auc", "delta_pr_auc", "flip_rate", "mean_abs_delta_p",
           "spearman_rho", "jaccard_top25", "jaccard_top50", "jaccard_top100",
           "delta_brier", "reliability_curve_l2"]

# preferential_edge_addition added here: the realistic counterpart to
# random_edge_addition, sampled by degree product rather than uniformly.
# Panel layout below is sized dynamically off this list's length so
# adding/removing a budget-indexed type doesn't require touching the
# plotting code.
BUDGET_TYPES = ["random_edge_deletion", "random_edge_addition",
                "preferential_edge_addition",
                "degree_aware_edge_deletion", "edge_rewiring"]


def aggregate_over_seeds(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_seeds"] = len(g)
        for m in METRICS:
            row[f"{m}_mean"] = round(g[m].mean(), 5)
            row[f"{m}_std"] = round(g[m].std(ddof=0), 5)
        rows.append(row)
    return rows


def main():
    tab_path = os.path.join(PERT_DIR, "perturbation_summary.csv")
    gnn_path = os.path.join(PERT_DIR, "gnn_perturbation_summary.csv")

    frames = []
    if os.path.exists(tab_path):
        frames.append(pd.read_csv(tab_path))
    else:
        print(f"[WARN] {tab_path} not found — run script 07 first")
    if os.path.exists(gnn_path):
        frames.append(pd.read_csv(gnn_path))
    else:
        print(f"[WARN] {gnn_path} not found — run script 10 to include the GNN")

    if not frames:
        print("[ERROR] No perturbation result files found — run script 07 and/or 11 first")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df[df["perturb_type"] != "clean"].copy()

    print("\nRobustness aggregation")
    print(f"Loaded {len(df):,} perturbation-scenario rows from {len(frames)} source file(s) "
          f"({df['model'].nunique()} models: {sorted(df['model'].unique())}, "
          f"{df['perturb_type'].nunique()} perturbation types)")

    has_hub = "hub_k" in df.columns and df["hub_k"].notna().any()

    budget_rows = aggregate_over_seeds(
        df[df["perturb_type"] != "hub_removal"], ["model", "perturb_type", "budget"])
    hub_rows = []
    if has_hub:
        hub_rows = aggregate_over_seeds(
            df[df["perturb_type"] == "hub_removal"], ["model", "perturb_type", "hub_k"])

    wide = pd.DataFrame(budget_rows + hub_rows)
    wide_path = os.path.join(PERT_DIR, "robustness_summary_wide.csv")
    wide.to_csv(wide_path, index=False)
    print(f"Wrote {wide_path}  ({len(wide)} rows)")

    # One CSV per metric: budget x model pivot (budget-indexed types only)
    budget_wide = wide[wide["perturb_type"] != "hub_removal"]
    for m in METRICS:
        pivot = budget_wide.pivot_table(
            index=["perturb_type", "budget"], columns="model", values=f"{m}_mean")
        pivot.to_csv(os.path.join(BY_METRIC_DIR, f"{m}.csv"))
    print(f"Wrote {len(METRICS)} per-metric CSVs → {BY_METRIC_DIR}")

    # Budget-indexed robustness grid (one panel per perturbation type).
    # Panel count/layout is derived from BUDGET_TYPES so it doesn't need to
    # be hand-adjusted when a type is added (as with preferential_edge_
    # addition here) or dropped.
    models = sorted(df["model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    ncols = 3 if len(BUDGET_TYPES) > 4 else 2
    nrows = math.ceil(len(BUDGET_TYPES) / ncols)

    def plot_budget_grid(metric: str, ylabel: str, title: str, fname: str):
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        flat_axes = axes.flat
        for ax, ptype in zip(flat_axes, BUDGET_TYPES):
            sub = wide[wide["perturb_type"] == ptype].sort_values("budget")
            if sub.empty:
                ax.set_title(f"{ptype.replace('_', ' ')} (no data)")
                continue
            for color, model in zip(colors, models):
                ms = sub[sub["model"] == model]
                if ms.empty:
                    continue
                ax.plot(ms["budget"], ms[mean_col], marker="o", label=model, color=color)
                ax.fill_between(ms["budget"],
                               ms[mean_col] - ms[std_col],
                               ms[mean_col] + ms[std_col],
                               color=color, alpha=0.12)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(ptype.replace("_", " "))
            ax.set_xlabel("perturbation budget")
            ax.set_ylabel(ylabel)
        # hide any unused trailing axes (grid may have more cells than types)
        for ax in list(flat_axes)[len(BUDGET_TYPES):]:
            ax.set_visible(False)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(title, y=1.00)
        fig.tight_layout()
        fig_path = os.path.join(PERT_DIR, fname)
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")

    plot_budget_grid("delta_roc_auc", "Δ ROC-AUC (perturbed − clean)",
                     "Robustness of network-model predictions under graph perturbation",
                     "robustness_grid.png")
    plot_budget_grid("flip_rate", "flip rate (perturbed vs. clean predictions)",
                     "Prediction-flip rate under graph perturbation",
                     "robustness_grid_flip_rate.png")

    # Perturbation-type comparison figure. Companion to the grid above:
    # instead of one panel per perturbation type with one line per model,
    # this is one line per perturbation type, so all 5 types are directly
    # comparable on shared axes. node_feature_noise is GNN-only (tabular
    # "features" are graph-derived downstream, not stored node attributes),
    # so GNN is the only model with all 5 types measured under an identical
    # protocol -- shown alone in the left panel. The right panel averages
    # the 4 shared types across the 5 tabular classifiers so the comparison
    # isn't a 25-line spaghetti plot.
    all_ptypes = sorted(wide.loc[wide["perturb_type"] != "hub_removal", "perturb_type"].unique())
    ptype_colors = dict(zip(all_ptypes, plt.cm.tab10(np.linspace(0, 1, len(all_ptypes)))))

    fig2, (ax_gnn, ax_tab) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    gnn_sub = wide[(wide["model"] == "GNN") & (wide["perturb_type"] != "hub_removal")]
    for ptype in all_ptypes:
        ps = gnn_sub[gnn_sub["perturb_type"] == ptype].sort_values("budget")
        if ps.empty:
            continue
        c = ptype_colors[ptype]
        ax_gnn.plot(ps["budget"], ps["delta_roc_auc_mean"], marker="o", label=ptype.replace("_", " "), color=c)
        ax_gnn.fill_between(ps["budget"],
                           ps["delta_roc_auc_mean"] - ps["delta_roc_auc_std"],
                           ps["delta_roc_auc_mean"] + ps["delta_roc_auc_std"], color=c, alpha=0.12)
    ax_gnn.axhline(0, color="black", lw=0.8, ls="--")
    ax_gnn.set_title("GNN — all 5 perturbation types\n(band = ± std over 5 seeds)")
    ax_gnn.set_xlabel("perturbation budget")
    ax_gnn.set_ylabel("Δ ROC-AUC (perturbed − clean)")

    tab_models = [m for m in wide["model"].unique() if m != "GNN"]
    tab_sub = wide[(wide["model"].isin(tab_models)) & (wide["perturb_type"] != "hub_removal")
                  & (wide["perturb_type"] != "node_feature_noise")]
    for ptype in all_ptypes:
        if ptype == "node_feature_noise":
            continue
        ps = tab_sub[tab_sub["perturb_type"] == ptype]
        if ps.empty:
            continue
        grouped = ps.groupby("budget")["delta_roc_auc_mean"].agg(["mean", "std"]).reset_index()
        c = ptype_colors[ptype]
        ax_tab.plot(grouped["budget"], grouped["mean"], marker="o", label=ptype.replace("_", " "), color=c)
        ax_tab.fill_between(grouped["budget"], grouped["mean"] - grouped["std"],
                           grouped["mean"] + grouped["std"], color=c, alpha=0.12)
    ax_tab.axhline(0, color="black", lw=0.8, ls="--")
    ax_tab.set_title(f"Tabular models (n={len(tab_models)}) — mean across classifiers\n"
                    "(band = ± std across the 5 classifiers' means)")
    ax_tab.set_xlabel("perturbation budget")

    handles, labels = ax_gnn.get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", ncol=len(all_ptypes), bbox_to_anchor=(0.5, -0.08))
    fig2.suptitle("Degradation by perturbation type, across budgets", y=1.02)
    fig2.tight_layout()
    fig2_path = os.path.join(PERT_DIR, "perturbation_type_comparison.png")
    fig2.savefig(fig2_path, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {fig2_path}")

    # Hub-removal curves (dense K grid). With hub_removal_k densified to
    # every integer 1-25 in experiment_config.json, this is now a real curve
    # rather than 3 disconnected points. hub_removal is K-indexed (not
    # budget-indexed) and deterministic (no seed loop), so there is no std
    # band here -- each (model, K) cell is a single value.
    if has_hub:
        hub_wide = wide[wide["perturb_type"] == "hub_removal"].copy()
        hub_wide = hub_wide.sort_values("hub_k")

        fig3, (ax_auc, ax_flip, ax_rho) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        for color, model in zip(colors, models):
            ms = hub_wide[hub_wide["model"] == model]
            if ms.empty:
                continue
            ax_auc.plot(ms["hub_k"], ms["delta_roc_auc_mean"], marker="o", label=model, color=color)
            ax_flip.plot(ms["hub_k"], ms["flip_rate_mean"], marker="o", label=model, color=color)
            ax_rho.plot(ms["hub_k"], ms["spearman_rho_mean"], marker="o", label=model, color=color)

        ax_auc.axhline(0, color="black", lw=0.8, ls="--")
        ax_auc.set_ylabel("Δ ROC-AUC (perturbed − clean)")
        ax_auc.set_title("Hub removal: accuracy degradation vs. K")

        ax_flip.set_ylabel("flip rate")
        ax_flip.set_title("Hub removal: prediction-flip rate vs. K")

        ax_rho.set_ylabel("Spearman ρ (clean vs. perturbed ranking)")
        ax_rho.set_xlabel("K (number of top-degree investors removed)")
        ax_rho.set_title("Hub removal: ranking stability vs. K")
        ax_rho.invert_yaxis()  # so "worse" (lower rho) reads as down, consistent with the other two panels

        handles, labels = ax_auc.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.02))
        fig3.suptitle(f"Hub-removal robustness curve (K = 1..{int(hub_wide['hub_k'].max())})", y=1.00)
        fig3.tight_layout()
        fig3_path = os.path.join(PERT_DIR, "hub_removal_curves.png")
        fig3.savefig(fig3_path, dpi=140, bbox_inches="tight")
        plt.close(fig3)
        print(f"Wrote {fig3_path}")

    # Console summary
    print("\nHeadline findings (at max tested budget per type)")
    for ptype in BUDGET_TYPES:
        sub = wide[wide["perturb_type"] == ptype]
        if sub.empty:
            continue
        max_b = sub["budget"].max()
        at_max = sub[sub["budget"] == max_b].sort_values("delta_roc_auc_mean")
        print(f"\n  {ptype} @ budget={max_b}:")
        for _, r in at_max.iterrows():
            print(f"    {r['model']:12s}  ΔAUC={r['delta_roc_auc_mean']:+.4f} ± {r['delta_roc_auc_std']:.4f}  "
                  f"flip_rate={r['flip_rate_mean']:.4f}  ρ={r['spearman_rho_mean']:.4f}")

    if has_hub:
        print(f"\n  hub_removal (whole-node removal):")
        hub_sorted = wide[wide["perturb_type"] == "hub_removal"].sort_values(["hub_k", "delta_roc_auc_mean"])
        for _, r in hub_sorted.iterrows():
            print(f"    K={int(r['hub_k']):<3} {r['model']:12s}  ΔAUC={r['delta_roc_auc_mean']:+.4f}  "
                  f"flip_rate={r['flip_rate_mean']:.4f}  ρ={r['spearman_rho_mean']:.4f}")

    print(f"\nAll outputs → {PERT_DIR}")


if __name__ == "__main__":
    main()
