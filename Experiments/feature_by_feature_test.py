"""
Experiments/feature_by_feature_test.py -- systematic, one-by-one re-evaluation
of the ~26-column original candidate feature set against the current frozen
6-feature network model (log_first_round_raised, cohort_year,
firm_age_at_first_round, sector, mean_pagerank, missing_evc).

The original feature reduction (26 -> 6) was done via forward/backward
selection on validation ROC-AUC alone, with no formal significance testing
and no per-feature audit trail preserved. This re-tests every dropped
candidate individually -- the control (frozen 6-feature set) plus one
candidate at a time, for all 5 tabular model classes -- and runs a paired DeLong test
(same test set, same target, reused from significance_tests.py) against the
CONTROL model of the same class, so "adds performance" means something
statistically precise, not just a higher point estimate.

Two candidates are dropped from the sweep because they are constant in this
sample (first_round_deal_value_mn, first_round_size_missing -- both
nunique==1, zero information by construction, verified before writing this
script). Three network candidates (mean_weighted_degree,
max_weighted_degree, max_clustering) aren't in firms_with_network_T7.parquet
and are computed fresh via the same die_fp merge pattern already used in
full_raw_features_test.py (imported directly, not duplicated).

Standalone, read-only with respect to Scripts/ and the trained production
models -- nothing here overwrites baseline_results.json/network_results.json.
Run: python3 Experiments/feature_by_feature_test.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost  # noqa: F401 -- import before any torch-adjacent library, see significance_tests.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
from full_raw_features_test import build_full_raw_features  # noqa: E402
from significance_tests import delong_test  # noqa: E402

SCRIPTS_DIR = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
from pipeline_core import (  # noqa: E402
    SEED, TRAIN_END, VAL_END, CAT_FEATURE_COLS,
    BASELINE_NUM_COLS, NETWORK_NUM_COLS, NETWORK_BINARY_COLS,
    temporal_train_val_test_split, fit_preprocessor, transform_preprocessor,
    set_global_seed, engineer_baseline_features, tune_on_val,
    evaluate_with_tuned_threshold,
)

ROOT     = os.path.dirname(SCRIPTS_DIR)
DATA     = os.path.join(ROOT, "Data", "processed")
OUT_DIR  = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED

CONTROL_NUM = BASELINE_NUM_COLS + NETWORK_NUM_COLS   # 4 cols
CONTROL_BIN = NETWORK_BINARY_COLS                     # 1 col: missing_evc
CONTROL_CAT = CAT_FEATURE_COLS                         # 1 col: sector

# (name, kind) -- kind in {"num", "bin", "cat"}. Excludes
# first_round_deal_value_mn and first_round_size_missing (constant,
# nunique==1, verified before writing this script).
CANDIDATES = [
    ("n_first_round_deals",        "num"),
    ("first_round_number",         "num"),
    ("syndicate_size",             "num"),
    ("n_investors_first_round",    "num"),
    ("n_known_investors",          "num"),
    ("mean_degree",                "num"),
    ("max_degree",                 "num"),
    ("mean_betweenness",           "num"),
    ("max_betweenness",            "num"),
    ("max_pagerank",               "num"),
    ("mean_eigenvector_lcc",       "num"),
    ("max_eigenvector_lcc",        "num"),
    ("mean_clustering",            "num"),
    ("mean_weighted_degree",       "num"),
    ("max_weighted_degree",        "num"),
    ("max_clustering",             "num"),
    ("share_newcomer_investors",   "num"),
    ("has_top_pagerank_investor",  "bin"),
    ("company_state",              "cat"),
    ("first_round_stage",          "cat"),
]


def build_data() -> pd.DataFrame:
    firms = build_full_raw_features()  # adds mean/max_weighted_degree, max_clustering
    firms = engineer_baseline_features(firms, LABEL_COL, DATE_COL)
    firms["missing_evc"] = firms["mean_eigenvector_lcc"].isna().astype(int)

    # mean_pagerank is part of CONTROL_NUM, not the candidate loop below --
    # fill it too, same convention as every other numeric network column
    # (0 = no/unknown centrality), or every variant (including control) fails.
    firms["mean_pagerank"] = firms["mean_pagerank"].fillna(0.0)

    numeric_candidates = [name for name, kind in CANDIDATES if kind in ("num", "bin")]
    for col in numeric_candidates:
        firms[col] = firms[col].fillna(0.0)
    for col in [name for name, kind in CANDIDATES if kind == "cat"]:
        firms[col] = firms[col].fillna("Unknown").astype(str)

    return firms


def train_variant(variant_name: str, df: pd.DataFrame,
                  extra_num: list, extra_bin: list, extra_cat: list):
    num_cols = CONTROL_NUM + extra_num
    bin_cols = CONTROL_BIN + extra_bin
    cat_cols = CONTROL_CAT + extra_cat

    train, val, test = temporal_train_val_test_split(df, TRAIN_END, VAL_END)
    y_tr, y_val, y_te = train[LABEL_COL], val[LABEL_COL], test[LABEL_COL]
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    pre = fit_preprocessor(train, num_cols, bin_cols, cat_cols)
    X_tr, X_val, X_te = (transform_preprocessor(pre, s) for s in (train, val, test))

    results = []
    probs = {}

    def run_model(model_name, make_fn, grid):
        p, _, _ = tune_on_val(make_fn, grid, X_tr, y_tr, X_val, y_val)
        clf = make_fn(**p).fit(X_tr, y_tr)
        test_prob = clf.predict_proba(X_te)[:, 1]
        r = evaluate_with_tuned_threshold(f"{model_name}_{variant_name}", clf,
                                          X_val, y_val, X_te, y_te, "Test", None, verbose=False)
        r["variant"] = variant_name
        r["model_class"] = model_name
        results.append(r)
        probs[model_name] = test_prob

    run_model("LogReg",
        lambda C: LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                     max_iter=1000, random_state=RANDOM_STATE),
        [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]])

    run_model("RandomForest",
        lambda max_depth, min_samples_leaf: RandomForestClassifier(
            n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        [{"max_depth": d, "min_samples_leaf": m} for d in [10, 20, None] for m in [5, 10]])

    run_model("HistGBM",
        lambda learning_rate, max_iter, max_depth: HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
            class_weight="balanced", random_state=RANDOM_STATE),
        [{"learning_rate": lr, "max_iter": n, "max_depth": d}
         for lr in [0.05, 0.1] for n in [100, 200] for d in [3, 5]])

    if HAS_XGB:
        run_model("XGBoost",
            lambda n_estimators, max_depth, learning_rate, subsample=0.8: XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1),
            [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
             for n in [200, 400] for d in [3, 5] for lr in [0.05, 0.1] for ss in [0.7, 1.0]])

    if HAS_LGBM:
        run_model("LightGBM",
            lambda num_leaves, max_depth, learning_rate, n_estimators: LGBMClassifier(
                num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                n_estimators=n_estimators, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1),
            [{"num_leaves": nl, "max_depth": -1, "learning_rate": lr, "n_estimators": n}
             for nl in [15, 31] for lr in [0.05, 0.1] for n in [200, 400]])

    return results, probs, y_te.values


def main():
    set_global_seed()
    print(f"\nFeature-by-feature test ({len(CANDIDATES)} candidates vs. frozen control)")

    df = build_data()

    print("\nControl (frozen 6-feature network set)")
    control_results, control_probs, y_te = train_variant("control", df, [], [], [])
    control_auc = {r["model_class"]: r["roc_auc"] for r in control_results}

    all_rows = []
    for name, kind in CANDIDATES:
        print(f"\nCandidate: {name} ({kind})")
        extra_num = [name] if kind == "num" else []
        extra_bin = [name] if kind == "bin" else []
        extra_cat = [name] if kind == "cat" else []
        results, probs, y_te_variant = train_variant(name, df, extra_num, extra_bin, extra_cat)

        assert np.array_equal(y_te, y_te_variant), f"test-set row mismatch for candidate {name}"

        for r in results:
            mc = r["model_class"]
            variant_auc = r["roc_auc"]
            _, _, z, p = delong_test(y_te, probs[mc], control_probs[mc])
            all_rows.append({
                "candidate": name, "kind": kind, "model_class": mc,
                "control_auc": control_auc[mc], "variant_auc": variant_auc,
                "delta": variant_auc - control_auc[mc],
                "delong_z": z, "delong_p": p,
                "significant_at_05": p < 0.05,
            })

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "feature_by_feature_results.csv"), index=False)

    print("\nSummary -- sorted by mean delta across model classes")
    summary = res.groupby(["candidate", "kind"]).agg(
        mean_delta=("delta", "mean"),
        max_delta=("delta", "max"),
        n_significant=("significant_at_05", "sum"),
        n_models=("model_class", "count"),
    ).sort_values("mean_delta", ascending=False)
    print(summary.round(4).to_string())

    n_sig_positive = res[(res["significant_at_05"]) & (res["delta"] > 0)]
    print(f"\nCandidate x model-class cells that are both significant and positive: {len(n_sig_positive)}")
    if len(n_sig_positive):
        print(n_sig_positive[["candidate", "model_class", "delta", "delong_p"]].to_string(index=False))
    else:
        print("(none)")

    print(f"\nSaved -> {os.path.join(OUT_DIR, 'feature_by_feature_results.csv')}")


if __name__ == "__main__":
    main()
