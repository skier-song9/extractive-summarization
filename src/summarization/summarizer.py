from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from storage.postgres_storage import PostgresStorage

from .config import SummarizationConfig
from .embedder import compute_similarity_matrix, embed_sentences, resolve_embedding_device
from .fusion import extract_top_sentences, fuse_scores
from .graph_ranker import compute_pagerank_scores
from .lsa_scorer import apply_gidf_boost, compute_lsa_scores
from .sentence_splitter import split_sentences
from .utils import count_tokens, get_logger, resolve_worker_count
from .vocab_builder import load_gidf


class HybridExtractiveSummarizer:
    def __init__(
        self,
        cfg: SummarizationConfig,
        gidf_storage: PostgresStorage | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__, cfg.logging.level)
        self._gidf = load_gidf(cfg.gidf, storage=gidf_storage)

    def summarize_one(self, text: str) -> str:
        sentences = split_sentences(text, self.cfg.preprocessing)
        if len(sentences) < 2:
            return text.strip()

        source_token_count = count_tokens(text)
        embeddings = embed_sentences(sentences, self.cfg.embedding)
        sim_matrix = compute_similarity_matrix(embeddings)
        lsa_scores = compute_lsa_scores(
            sentences,
            self.cfg.lsa,
            self.cfg.preprocessing.language,
            self.cfg.preprocessing.spacy_model,
            self.cfg.embedding,
        )
        boosted_lsa_scores = apply_gidf_boost(
            lsa_scores,
            sentences,
            self._gidf,
            language=self.cfg.preprocessing.language,
            spacy_model=self.cfg.preprocessing.spacy_model,
            embedding_cfg=self.cfg.embedding,
        )
        pagerank_scores = compute_pagerank_scores(sim_matrix, self.cfg.graph)
        fused_scores = fuse_scores(boosted_lsa_scores, pagerank_scores, self.cfg.fusion)

        if self.cfg.logging.log_sentence_scores:
            for index, sentence in enumerate(sentences):
                self.logger.debug(
                    "[%02d] lsa=%.3f gidf_lsa=%.3f pr=%.3f fused=%.3f | %s",
                    index,
                    lsa_scores[index],
                    boosted_lsa_scores[index],
                    pagerank_scores[index],
                    fused_scores[index],
                    sentence[:80],
                )

        return " ".join(
            extract_top_sentences(
                sentences,
                fused_scores,
                sim_matrix,
                self.cfg.extraction,
                source_token_count=source_token_count,
            )
        )

    def summarize_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        n_workers = resolve_worker_count(self.cfg.parallel.n_workers)
        resolved_device = resolve_embedding_device(self.cfg.embedding.device)
        if resolved_device == "mps" and n_workers > 1:
            self.logger.info(
                "Using a single summarization worker because embedding device '%s' is active.",
                resolved_device,
            )
            n_workers = 1
        if n_workers == 1 or len(texts) == 1:
            return [self.summarize_one(text) for text in texts]

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self.summarize_one, text): index
                for index, text in enumerate(texts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        return [results[index] for index in range(len(texts))]
