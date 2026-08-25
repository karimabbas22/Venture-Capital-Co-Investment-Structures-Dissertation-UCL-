"""
perturbation_core.py — reusable graph perturbation primitives.

Adapter-agnostic perturbation functions operate on plain edge lists / degree
maps so they can be reused by any graph representation: `perturb_graph()`/
`hub_removal()` work directly on the investor-investor NetworkX graph used
by the tabular pipeline (scripts 04/05/07). For the bipartite PyG graph
(script 09's GNN), `hetero_edges_to_nx()`/`nx_to_hetero_edge_index()` round-
trip the `co_invests_with` relation through the same functions rather than
duplicating them; `node_feature_noise()` is GNN-exclusive (see below).

All perturbation functions take a budget as a fraction of existing edges
(e.g. 0.05 = 5%) and a seed for reproducibility, and return a new graph —
inputs are never mutated in place, matching the rest of the pipeline's
copy-on-write convention (`build_coinvestment_snapshot` etc.).
"""

from __future__ import annotations

import json

import numpy as np
import networkx as nx


# Perturbation primitives (operate on edge lists — adapter-agnostic)

def random_edge_deletion(edges: list[tuple], budget: float, seed: int) -> list[tuple]:
    """Remove a random `budget` fraction of edges."""
    rng = np.random.default_rng(seed)
    edges = list(edges)
    n_remove = min(len(edges), max(1, int(round(budget * len(edges)))))
    remove_idx = set(rng.choice(len(edges), size=n_remove, replace=False))
    return [e for i, e in enumerate(edges) if i not in remove_idx]


def random_edge_addition(edges: list[tuple], nodes: list, budget: float, seed: int) -> list[tuple]:
    """Add a random `budget` fraction (of the current edge count) of new,
    previously-absent edges between existing nodes."""
    rng = np.random.default_rng(seed)
    edges = list(edges)
    nodes = list(nodes)
    existing = {frozenset(e) for e in edges}
    n_add = max(1, int(round(budget * len(edges))))
    added = []
    max_attempts = n_add * 20
    attempts = 0
    n_nodes = len(nodes)
    while len(added) < n_add and attempts < max_attempts and n_nodes > 1:
        i, j = rng.choice(n_nodes, size=2, replace=False)
        u, v = nodes[i], nodes[j]
        key = frozenset((u, v))
        if key not in existing:
            existing.add(key)
            added.append((u, v))
        attempts += 1
    return edges + added


def preferential_edge_addition(edges: list[tuple], nodes: list, degree_map: dict,
                               budget: float, seed: int) -> list[tuple]:
    """Add a `budget` fraction of new investor-investor edges, sampling
    endpoint pairs with probability proportional to the product of their
    current degrees (Barabasi-Albert-style preferential attachment).

    Rationale: real co-investment ties do not form uniformly at random.
    Syndication is strongly assortative and preferential (Sorenson &
    Stuart, 2001; Gu et al., 2022), so unrecorded ties are far more likely
    to sit between already-active investors than between two arbitrary
    nodes. This is the realistic counterpart to random_edge_addition,
    which is retained as the uninformative null against which this more
    realistic perturbation can be compared.

    Implementation: each endpoint is drawn independently with probability
    proportional to (degree + 1) -- the +1 Laplace-smooths zero-degree
    nodes so they retain a small nonzero chance of being selected, matching
    standard BA-style implementations -- so realized pair probability is
    approximately proportional to deg(i) * deg(j) for the budgets used
    here. Sampling is without replacement against previously-existing (and
    newly-added) edges, mirroring random_edge_addition's contract exactly
    so the two functions are directly comparable at equal budget."""
    rng = np.random.default_rng(seed)
    edges = list(edges)
    nodes = list(nodes)
    n_nodes = len(nodes)
    n_add = max(1, int(round(budget * len(edges))))
    if n_nodes <= 1:
        return edges

    existing = {frozenset(e) for e in edges}
    weights = np.array([degree_map.get(n, 0) + 1.0 for n in nodes], dtype=float)
    probs = weights / weights.sum()

    added = []
    max_attempts = n_add * 40
    attempts = 0
    while len(added) < n_add and attempts < max_attempts:
        i, j = rng.choice(n_nodes, size=2, replace=False, p=probs)
        u, v = nodes[i], nodes[j]
        key = frozenset((u, v))
        if key not in existing:
            existing.add(key)
            added.append((u, v))
        attempts += 1
    return edges + added


def degree_aware_edge_deletion(edges: list[tuple], degree_map: dict, budget: float, seed: int) -> list[tuple]:
    """Remove a `budget` fraction of edges, sampled with probability
    proportional to the sum of endpoint degrees — preferentially strips
    edges touching high-degree (hub) nodes. Distinct from whole-node hub
    removal: this perturbs individual edges, not entire nodes, and is
    expressed as a continuous budget rather than an integer K."""
    rng = np.random.default_rng(seed)
    edges = list(edges)
    weights = np.array(
        [degree_map.get(u, 0) + degree_map.get(v, 0) for u, v in edges], dtype=float
    )
    probs = weights / weights.sum() if weights.sum() > 0 else np.full(len(edges), 1.0 / len(edges))
    n_remove = min(len(edges), max(1, int(round(budget * len(edges)))))
    remove_idx = set(rng.choice(len(edges), size=n_remove, replace=False, p=probs))
    return [e for i, e in enumerate(edges) if i not in remove_idx]


def edge_rewiring(G: nx.Graph, budget: float, seed: int) -> nx.Graph:
    """Degree-preserving rewiring via nx.double_edge_swap: isolates whether
    topology beyond the degree sequence matters, holding each node's degree
    fixed. Operates on (and returns) a full nx.Graph rather than an edge
    list, since double_edge_swap is graph-native in NetworkX."""
    G_pert = G.copy()
    n_swaps = max(1, int(round(budget * G_pert.number_of_edges())))
    try:
        nx.double_edge_swap(G_pert, nswap=n_swaps, max_tries=n_swaps * 20, seed=seed)
    except nx.NetworkXError:
        pass  # graph too small/sparse for the requested swap count; best-effort
    return G_pert


def hub_removal(G: nx.Graph, hub_ids: list) -> nx.Graph:
    """Remove a fixed set of nodes (and all incident edges) entirely.
    K-indexed (integer count of top hubs), not budget-indexed — kept
    separate from the fractional-budget perturbations above."""
    G_pert = G.copy()
    for nid in hub_ids:
        if nid in G_pert:
            G_pert.remove_node(nid)
    return G_pert


# HeteroData (GNN) helpers — same-type (investor-investor) edge relation
# reuses the nx.Graph perturbation functions above via conversion; node
# feature noise is GNN-exclusive (the tabular pipeline's "features" are
# graph-derived downstream, not stored node attributes to jitter).

def hetero_edges_to_nx(edge_index, num_nodes: int) -> nx.Graph:
    """Convert a symmetric same-type edge_index tensor (as stored after
    ToUndirected) to an undirected nx.Graph with integer node labels
    0..num_nodes-1, so it can be perturbed with perturb_graph()/hub_removal
    above unchanged."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u != v:
            G.add_edge(u, v)
    return G


def nx_to_hetero_edge_index(G: nx.Graph):
    """Convert an undirected nx.Graph back to a symmetric (both-direction)
    edge_index tensor suitable for a HeteroData same-type relation."""
    import torch
    src, dst = [], []
    for u, v in G.edges():
        src.extend([u, v])
        dst.extend([v, u])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def node_feature_noise(x, sigma_scale: float, seed: int):
    """Gaussian jitter on a node feature tensor: noise_std = sigma_scale *
    per-column std(x). GNN-exclusive perturbation (see module docstring)."""
    import torch
    rng = np.random.default_rng(seed)
    x_np = x.detach().cpu().numpy()
    col_std = x_np.std(axis=0, keepdims=True)
    noise = rng.normal(loc=0.0, scale=sigma_scale * col_std, size=x_np.shape)
    return torch.tensor(x_np + noise, dtype=x.dtype)


def recompute_investor_features(G_pert: nx.Graph, num_investor: int, investor_preprocessor):
    """Recompute the 6 raw investor centrality metrics from a perturbed
    co-investment graph and transform them through the already-fitted
    (train-only) investor preprocessor.

    This closes an asymmetry between the two perturbation tracks:
    script 07 (tabular) recomputes centrality features from the damaged
    graph before re-scoring, but the GNN track originally left investor
    node features frozen at their clean-graph values after topology-only
    perturbation -- meaning the GNN was reacting to new edges with stale
    node inputs. Using this function in script 10 and the GNN placebo test
    makes both tracks' perturbation protocol identical (perturb graph ->
    recompute real centrality metrics -> re-score), so GNN vs. tabular
    robustness magnitudes become directly comparable, not just
    qualitatively similar.

    G_pert must use integer node labels 0..num_investor-1 (as produced by
    hetero_edges_to_nx), so the recomputed feature rows line up with the
    investor node order used everywhere else in the HeteroData object.
    """
    import torch
    import pandas as pd
    from pipeline_core import compute_snapshot_metrics, BETWEENNESS_K, SEED, transform_preprocessor
    from graph_core import INVESTOR_NUM_COLS

    metrics = compute_snapshot_metrics(G_pert, betweenness_k=BETWEENNESS_K, seed=SEED)
    if metrics.empty:
        metrics = pd.DataFrame({"investor_org_id": []})
    metrics = metrics.set_index("investor_org_id").reindex(range(num_investor)).reset_index()
    metrics[INVESTOR_NUM_COLS] = metrics[INVESTOR_NUM_COLS].fillna(0.0)
    X = transform_preprocessor(investor_preprocessor, metrics)
    return torch.tensor(np.asarray(X), dtype=torch.float)


# Dispatcher

EDGE_LIST_PERTURBATIONS = {
    "random_edge_deletion": lambda edges, nodes, degree_map, budget, seed:
        random_edge_deletion(edges, budget, seed),
    "random_edge_addition": lambda edges, nodes, degree_map, budget, seed:
        random_edge_addition(edges, nodes, budget, seed),
    "preferential_edge_addition": lambda edges, nodes, degree_map, budget, seed:
        preferential_edge_addition(edges, nodes, degree_map, budget, seed),
    "degree_aware_edge_deletion": lambda edges, nodes, degree_map, budget, seed:
        degree_aware_edge_deletion(edges, degree_map, budget, seed),
}


def perturb_graph(G: nx.Graph, perturbation_type: str, budget: float, seed: int) -> nx.Graph:
    """Dispatch to the appropriate perturbation function; always returns a
    new nx.Graph (same node set, perturbed edge set), never mutates G."""
    if perturbation_type == "edge_rewiring":
        return edge_rewiring(G, budget, seed)

    if perturbation_type not in EDGE_LIST_PERTURBATIONS:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    edges = list(G.edges())
    nodes = list(G.nodes())
    degree_map = dict(G.degree())
    new_edges = EDGE_LIST_PERTURBATIONS[perturbation_type](edges, nodes, degree_map, budget, seed)

    G_pert = nx.Graph()
    G_pert.add_nodes_from(G.nodes(data=True))
    G_pert.add_edges_from(new_edges)
    return G_pert


def load_experiment_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
