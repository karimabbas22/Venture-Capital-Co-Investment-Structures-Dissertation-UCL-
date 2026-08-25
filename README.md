<title>Reproduction Package README</title>
# Reproduction Package

This is a self-contained copy of the code needed to reproduce the empirical pipeline for this dissertation, from raw data through to the final aggregated report. It contains **code only** — no data, no trained models, no results, no figures. Running the pipeline in order regenerates everything.

## What this pipeline does

Starting from raw VC deal, exit, and company records, the pipeline:

1. Cleans and standardizes the raw data.
2. Constructs a startup panel with a fixed-horizon binary exit label, anchored at each startup's first VC round.
3. Trains a tabular classifier on deal-level features only (no network information).
4. Builds a time-respecting investor co-investment network from quarterly snapshots (no future information relative to any given snapshot date).
5. Trains a tabular classifier augmented with a network-derived feature.
6. Compares network structure across two time periods.
7. Perturbs the co-investment network under several controlled schemes (random edge deletion/addition, degree-aware deletion, preferential-attachment addition, edge rewiring, targeted removal of the most-connected investors) and re-scores the tabular models against each perturbed graph.
8. Builds a heterogeneous bipartite graph (startup and investor nodes, multiple edge types) for a graph neural network.
9. Trains the heterogeneous GNN on that graph.
10. Runs the same perturbation protocol against the GNN.
11. Aggregates both perturbation tracks into combined tables and figures.
12. Produces a single final report aggregating every stage above.

A further set of independent, standalone checks (statistical significance testing, a placebo test for the hub-removal perturbation, feature-selection ablations, GNN architectural ablations, and a few smaller robustness checks) live in `Experiments/`. Each is self-contained, documents its own prerequisites in its own docstring, and is not required to reproduce the core pipeline above.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Three raw Refinitiv Excel files are required and are not included in this package. See `Data/README.md` for exactly what's needed and where to place it.

## Running the core pipeline

From this folder, run in order:

```bash
python Scripts/01_clean_data.py
python Scripts/02_build_startup_first_round_dataset.py
python Scripts/03_train_baseline_model.py
python Scripts/04_build_time_respecting_coinvestment_network.py
python Scripts/05_train_network_model.py
python Scripts/06_compare_period_networks.py
python Scripts/07_run_perturbation_tests.py
python Scripts/08_build_hetero_graph.py
python Scripts/09_train_gnn_model.py
python Scripts/10_run_gnn_perturbation_tests.py
python Scripts/11_aggregate_robustness_report.py
python Scripts/12_evaluate_and_export_results.py
```

Each script reads its inputs from `Data/processed/` (writing there in turn) and is numbered to match dependency order exactly — running 01 through 12 in sequence is sufficient, nothing needs to be run out of order.

`Scripts/experiment_config.json` controls the perturbation grid (budgets, seeds, perturbation types, and the hub-removal K values) used by scripts 07 and 10.

Two shared modules underpin the whole pipeline and are not run directly: `pipeline_core.py` (data loading, leakage-safe feature construction, chronological splitting, shared preprocessing) and `graph_core.py` / `gnn_model.py` / `perturbation_core.py` / `stability_metrics.py` (graph construction, GNN architecture, perturbation primitives, and the stability-metric definitions used by scripts 07/10/11).

## Running the follow-up checks (`Experiments/`)

Each script below is independent and reads only from `Data/processed/` and already-trained models — none of them modify anything the core pipeline produces. Run any of them after the corresponding prerequisite stage of the core pipeline has completed:

| Script | Prerequisite | What it does |
|---|---|---|
| `significance_tests.py` | Scripts 03, 05, 09 | Paired statistical significance testing (DeLong's test + bootstrap confidence intervals) on ROC-AUC comparisons between model variants. |
| `multiple_comparisons_correction.py` | `significance_tests.py` | Applies family-wise and false-discovery-rate corrections to the significance test outputs. |
| `placebo_hub_removal.py` | Scripts 07, 09, 11 | Repeats the hub-removal perturbation with randomly-selected investors instead of the most-connected ones, as a placebo control. |
| `plot_realism_and_placebo_curves.py` | `placebo_hub_removal.py` (which itself needs script 11 to have run first) | Generates comparison figures from the placebo test and from the random-vs-preferential edge-addition perturbation types. |
| `gnn_ablations.py` | Scripts 08, 09 | Retrains the GNN under three architectural ablations (edge-type removed, investor node features replaced with a constant, one message-passing layer instead of two) and compares each to the full model. |
| `feature_by_feature_test.py` | Script 04 | Re-tests a set of originally-excluded candidate features one at a time against the production feature set. |
| `full_raw_features_test.py` | Script 04 | Retrains the tabular models on the full set of raw investor-level network metrics instead of the single aggregated feature used in production. |
| `cohort_year_ablation.py` | Script 02 | Retrains all tabular model variants with and without the cohort-year feature. |
| `test_literature_features.py` | Script 04 | Tests three additional literature-motivated network features against the production feature set. |
| `vc_hub_state_test.py` | `feature_by_feature_test.py`'s data-building step | Tests a coarser binary encoding of startup location (major VC-hub state vs. not) against the production feature set. |

## Output

All generated artifacts (cleaned data, trained models, graphs, perturbation results, figures, the final report) are written under `Data/processed/`, which is created automatically by the pipeline. This package does not include any pre-generated output.
