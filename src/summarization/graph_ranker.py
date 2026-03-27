from __future__ import annotations

import networkx as nx
import numpy as np

from .config import GraphConfig
from .utils import minmax_normalize, rank_normalize


def compute_pagerank_scores(sim_matrix: np.ndarray, cfg: GraphConfig) -> dict[int, float]:
    if sim_matrix.size == 0:
        return {}

    n_sentences = sim_matrix.shape[0]
    if n_sentences == 1:
        return {0: 1.0}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_sentences))

    mask = sim_matrix > cfg.similarity_threshold
    np.fill_diagonal(mask, False)
    rows, cols = np.where(mask)

    for row, col in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(row, col, weight=float(sim_matrix[row, col]))

    for node in graph.nodes:
        if graph.degree(node) == 0:
            graph.add_edge(node, node, weight=1.0)

    raw_scores = nx.pagerank(
        graph,
        alpha=cfg.damping,
        weight="weight",
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )

    if cfg.normalize_method == "rank":
        return rank_normalize(raw_scores)
    return minmax_normalize(raw_scores)
