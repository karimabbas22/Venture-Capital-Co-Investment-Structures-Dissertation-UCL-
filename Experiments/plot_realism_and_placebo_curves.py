"""
Experiments/plot_realism_and_placebo_curves.py — two dense-grid comparison
plots not covered by Scripts/11_aggregate_robustness_report.py, because
they need Experiments/output/placebo_vs_true_hub_comparison.csv, which only
exists after Experiments/placebo_hub_removal.py has run.

Run order matters here. placebo_hub_removal.py's own comparison step
(compare_against_true_hub_removal) reads
Data/processed/models/T7/perturbation/robustness_summary_wide.csv, which is
written by script 11. If placebo runs before 11, that file is still the
stale summary from before the K grid was densified (K in {5,10,25} only),
so the placebo-vs-true-hub comparison silently collapses to 3 points even
though placebo itself computed all 25 K values. Correct order for the
dense re-run:

    07_run_perturbation_tests.py
    10_run_gnn_perturbation_tests.py
    11_aggregate_robustness_report.py      <- writes the dense wide CSV
    Experiments/placebo_hub_removal.py     <- now compares against the dense wide CSV
    Experiments/plot_realism_and_placebo_curves.py   <- this script, last

Writes (both to Experiments/output/):
    hub_removal_vs_placebo.png           true hub removal (solid) vs.
                                          placebo random-K removal (dashed,
                                          +/- std band) per model, vs K.
                                          The gap between the line families
                                          is the "hub removal is not just
                                          any-K-removal" result.
    random_vs_preferential_addition.png  random_edge_addition vs.
                                          preferential_edge_addition, delta
                                          ROC-AUC and flip rate vs. budget.
                                          GNN solid, tabular mean dashed
                                          (band = std across the 5 tabular
                                          classifiers' means). Whether
                                          preferential addition does more
                                          damage at equal budget is the
                                          question this plot answers.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(ROOT, "Data", "processed")
PERT_DIR  = os.path.join(DATA, "models", "T7", "perturbation")
OUT_DIR   = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

WIDE_PATH    = os.path.join(PERT_DIR, "robustness_summary_wide.csv")
PLACEBO_PATH = os.path.join(OUT_DIR, "placebo_vs_true_hub_comparison.csv")


def plot_hub_vs_placebo():
    if not os.path.exists(PLACEBO_PATH):
        print(f"[WARN] {PLACEBO_PATH} not found — run Experiments/placebo_hub_removal.py "
              f"(after script 11) first. Skipping hub-removal-vs-placebo plot.")
        return

    df = pd.read_csv(PLACEBO_PATH).sort_values(["model", "hub_k"])
    models = sorted(df["model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for color, model in zip(colors, models):
        ms = df[df["model"] == model]
        if ms.empty:
            continue
        ax.plot(ms["hub_k"], ms["true_hub_delta_roc_auc"], marker="o",
               color=color, linestyle="-", label=f"{model} — true hub removal")
        ax.plot(ms["hub_k"], ms["placebo_delta_roc_auc_mean"], marker="s",
               color=color, linestyle="--", alpha=0.85, label=f"{model} — placebo (random K)")
        ax.fill_between(ms["hub_k"],
                       ms["placebo_delta_roc_auc_mean"] - ms["placebo_delta_roc_auc_std"],
                       ms["placebo_delta_roc_auc_mean"] + ms["placebo_delta_roc_auc_std"],
                       color=color, alpha=0.08)

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("K (investors removed)")
    ax.set_ylabel("Δ ROC-AUC (perturbed − clean)")
    ax.set_title("True hub removal vs. placebo random-K removal")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "hub_removal_vs_placebo.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    n_confirms = df["true_worse_than_placebo"].sum() if "true_worse_than_placebo" in df.columns else None
    if n_confirms is not None:
        print(f"  True hub removal more damaging than placebo in {n_confirms}/{len(df)} (model, K) cells.")


def plot_addition_comparison():
    if not os.path.exists(WIDE_PATH):
        print(f"[WARN] {WIDE_PATH} not found — run script 11 first. "
              f"Skipping random-vs-preferential-addition plot.")
        return

    wide = pd.read_csv(WIDE_PATH)
    ptypes = ["random_edge_addition", "preferential_edge_addition"]
    sub = wide[wide["perturb_type"].isin(ptypes)].copy()
    missing = [p for p in ptypes if p not in sub["perturb_type"].unique()]
    if missing:
        print(f"[WARN] Missing perturbation type(s) in {WIDE_PATH}: {missing}. "
              f"Re-run script 07/10 with the updated experiment_config.json "
              f"(perturbation_types now includes preferential_edge_addition).")
        return

    ptype_colors = {"random_edge_addition": "tab:blue", "preferential_edge_addition": "tab:red"}

    fig, (ax_auc, ax_flip) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)

    for metric, ax, ylabel in [
        ("delta_roc_auc_mean", ax_auc, "Δ ROC-AUC (perturbed − clean)"),
        ("flip_rate_mean", ax_flip, "flip rate"),
    ]:
        for ptype in ptypes:
            c = ptype_colors[ptype]
            ps = sub[sub["perturb_type"] == ptype]

            gnn = ps[ps["model"] == "GNN"].sort_values("budget")
            if not gnn.empty:
                ax.plot(gnn["budget"], gnn[metric], marker="o", color=c, linestyle="-",
                       label=f"{ptype.replace('_', ' ')} — GNN")

            tab = ps[ps["model"] != "GNN"]
            if not tab.empty:
                grouped = tab.groupby("budget")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(grouped["budget"], grouped["mean"], marker="s", color=c, linestyle="--",
                       alpha=0.85, label=f"{ptype.replace('_', ' ')} — tabular mean")
                ax.fill_between(grouped["budget"], grouped["mean"] - grouped["std"],
                               grouped["mean"] + grouped["std"], color=c, alpha=0.10)

        if metric == "delta_roc_auc_mean":
            ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("perturbation budget")
        ax.set_ylabel(ylabel)

    ax_auc.set_title("Accuracy degradation")
    ax_flip.set_title("Prediction-flip rate")
    handles, labels = ax_auc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Random vs. preferential-attachment edge addition, across budgets", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "random_vs_preferential_addition.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Quick console readout of which perturbation is more damaging at the
    # max budget.
    max_b = sub["budget"].max()
    at_max = sub[sub["budget"] == max_b].groupby("perturb_type")["delta_roc_auc_mean"].mean()
    print(f"\n  Mean ΔROC-AUC across all models @ budget={max_b}:")
    for ptype, val in at_max.items():
        print(f"    {ptype:28s} {val:+.5f}")


def main():
    print("Realism and placebo curves")
    plot_hub_vs_placebo()
    plot_addition_comparison()
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
