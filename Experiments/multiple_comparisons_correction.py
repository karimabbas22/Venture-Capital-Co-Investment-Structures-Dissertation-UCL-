"""
Experiments/multiple_comparisons_correction.py -- applies a family-wise and a
false-discovery-rate correction to the 8 DeLong-test p-values already computed
in significance_tests.py, addressing a supervisor-style review gap: the
"only LogReg's network lift survives at p=0.035" claim in
results_and_methodology.md Section 2.5 uses an UNCORRECTED alpha=0.05 across
8 comparisons drawn from overlapping/related model families. With 8 tests at
alpha=0.05, the expected number of false positives under a true global null
is 0.4 -- not negligible relative to the single "significant" result found.

This is a pure post-processing step on already-computed p-values (no
retraining, no new predictions) -- reads
Experiments/output/significance_comparisons.csv (produced by
significance_tests.py) and adds two corrected-significance columns:

  1. Bonferroni (family-wise error rate control): reject if p < alpha/m.
     Conservative -- controls the probability of ANY false positive across
     the family of 8 tests.
  2. Benjamini-Hochberg (false discovery rate control, alpha=0.05): sort
     p-values ascending, reject p_(k) if p_(k) <= (k/m)*alpha for the
     largest such k. Less conservative than Bonferroni, standard in
     multiple-hypothesis-testing practice when some power is still wanted.

Run: python3 Experiments/multiple_comparisons_correction.py
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Experiments", "output")
IN_CSV = os.path.join(OUT_DIR, "significance_comparisons.csv")


def bonferroni(pvals, alpha=0.05):
    m = len(pvals)
    threshold = alpha / m
    return [p < threshold for p in pvals], threshold


def benjamini_hochberg(pvals, alpha=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def main():
    df = pd.read_csv(IN_CSV)
    pvals = df["delong_p"].tolist()
    m = len(pvals)

    bonf_reject, bonf_threshold = bonferroni(pvals, alpha=0.05)
    bh_reject = benjamini_hochberg(pvals, alpha=0.05)

    df["significant_bonferroni_05"] = bonf_reject
    df["significant_bh_fdr_05"] = bh_reject

    out_path = os.path.join(OUT_DIR, "significance_comparisons_corrected.csv")
    df.to_csv(out_path, index=False)

    print(f"{'='*70}\n  MULTIPLE-COMPARISONS CORRECTION ({m} DeLong tests)\n{'='*70}")
    print(f"Bonferroni threshold: alpha/{m} = {bonf_threshold:.5f}")
    print(df[["comparison", "delong_p", "significant_at_05",
              "significant_bonferroni_05", "significant_bh_fdr_05"]].to_string(index=False))
    print(f"\nUncorrected (alpha=0.05): {sum(df['significant_at_05'])}/{m} significant")
    print(f"Bonferroni-corrected:     {sum(bonf_reject)}/{m} significant")
    print(f"BH-FDR-corrected (5%):    {sum(bh_reject)}/{m} significant")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
