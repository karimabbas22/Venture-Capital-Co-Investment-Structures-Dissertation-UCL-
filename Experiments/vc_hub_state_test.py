"""
Experiments/vc_hub_state_test.py -- follow-up to feature_by_feature_test.py's
company_state result: the raw 51-category state/region field hurt LogReg
significantly (p=0.049) when tested one-by-one against the frozen control.
That test used the full high-cardinality categorical (51 sparse one-hot
columns, California alone = 39% of the sample). This tests a coarser
binary alternative instead: is_vc_hub_state = 1 if the startup is
headquartered in California, New York, or Massachusetts (the three states
that dominate the sample and are the traditional US VC hubs), 0 otherwise.

Same protocol as feature_by_feature_test.py exactly (reuses its build_data,
train_variant, and frozen CONTROL_NUM/BIN/CAT unmodified) -- CONTROL +
this one binary candidate, all 5 model classes, DeLong test against the
control's own predictions on the same test set.

Standalone, read-only with respect to Scripts/ and the trained production
models. Run: python3 Experiments/vc_hub_state_test.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost  # noqa: F401 -- import before torch, see significance_tests.py

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
from feature_by_feature_test import build_data, train_variant, CONTROL_NUM, CONTROL_BIN, CONTROL_CAT  # noqa: E402
from significance_tests import delong_test  # noqa: E402

ROOT    = os.path.dirname(EXPERIMENTS_DIR)
OUT_DIR = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_HUB_STATES = {"California", "New York", "Massachusetts"}


def main():
    print("\nVC-HUB-STATE TEST (coarse binary vs. raw 51-category company_state)")

    df = build_data()
    df["is_vc_hub_state"] = df["company_state"].isin(TOP_HUB_STATES).astype(int)
    hub_rate = df["is_vc_hub_state"].mean()
    print(f"is_vc_hub_state=1 for {df['is_vc_hub_state'].sum():,} of {len(df):,} startups ({hub_rate:.1%})")
    print(f"  (California + New York + Massachusetts combined)")

    print("\n--- CONTROL (frozen 6-feature network set) ---")
    control_results, control_probs, y_test = train_variant("control", df, [], [], [])
    control_auc = {r["model_class"]: r["roc_auc"] for r in control_results}

    print("\n--- CANDIDATE: is_vc_hub_state ---")
    results, probs, y_test_variant = train_variant("is_vc_hub_state", df, [], ["is_vc_hub_state"], [])
    assert np.array_equal(y_test, y_test_variant), "test-set row mismatch"

    rows = []
    for r in results:
        mc = r["model_class"]
        variant_auc = r["roc_auc"]
        _, _, z, p = delong_test(y_test, probs[mc], control_probs[mc])
        rows.append({
            "candidate": "is_vc_hub_state", "model_class": mc,
            "control_auc": control_auc[mc], "variant_auc": variant_auc,
            "delta": round(variant_auc - control_auc[mc], 4),
            "delong_z": round(float(z), 4), "delong_p": round(float(p), 4),
            "significant_at_05": bool(p < 0.05),
        })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "vc_hub_state_test_results.csv"), index=False)

    print("\nSUMMARY")
    print(res[["model_class", "control_auc", "variant_auc", "delta", "delong_p", "significant_at_05"]]
          .to_string(index=False))
    print(f"\nMean delta across 5 models: {res['delta'].mean():+.4f}")
    print(f"Significant cells: {res['significant_at_05'].sum()}/5")
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'vc_hub_state_test_results.csv')}")


if __name__ == "__main__":
    main()
