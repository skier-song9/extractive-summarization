from __future__ import annotations

import numpy as np

from summarization.config import GraphConfig
from summarization.graph_ranker import compute_pagerank_scores


def test_compute_pagerank_scores_returns_normalized_scores() -> None:
    sim_matrix = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ],
        dtype=np.float32,
    )
    cfg = GraphConfig(similarity_threshold=0.2)

    scores = compute_pagerank_scores(sim_matrix, cfg)

    assert set(scores) == {0, 1, 2}
    assert max(scores.values()) == 1.0
    assert min(scores.values()) >= 0.0
    assert scores[1] >= scores[0]
