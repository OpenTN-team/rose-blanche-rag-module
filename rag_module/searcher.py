"""
searcher.py
Core of the RAG pipeline: computes cosine similarity between the query
embedding and all stored document embeddings, then returns the top-k results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .embeddings import EmbeddingModel
from .indexer import Fragment, VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result returned by SemanticSearcher."""
    rank: int
    fragment: Fragment
    score: float           # cosine similarity ∈ [-1, 1]
    score_percent: float   # score mapped to [0, 100] for display

    def __str__(self) -> str:
        border = "─" * 70
        source_info = f"  Source : {self.fragment.source}"
        if self.fragment.page is not None:
            source_info += f"  |  Page : {self.fragment.page}"
        return (
            f"\n{border}\n"
            f"  Résultat #{self.rank}  |  Score de similarité : {self.score:.4f}  ({self.score_percent:.1f}%)\n"
            f"{source_info}\n"
            f"{border}\n"
            f"{self.fragment.text.strip()}\n"
        )


class SemanticSearcher:
    """
    Semantic search engine based on cosine similarity.

    Workflow:
        1. Encode the user query with EmbeddingModel.
        2. Compute cosine similarity against all indexed embeddings.
        3. Rank by decreasing similarity.
        4. Return top-k SearchResult objects.

    Args:
        embedding_model: An EmbeddingModel instance.
        index:           A populated VectorIndex.
        top_k:           Default number of results to return (overridable per query).
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index: VectorIndex,
        top_k: int = 3,
    ) -> None:
        self.model = embedding_model
        self.index = index
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: float = -1.0,
    ) -> List[SearchResult]:
        """
        Run semantic search for *query*.

        Args:
            query:     The user question in natural language.
            top_k:     Override the default number of results.
            threshold: Minimum cosine similarity score to include a result.

        Returns:
            A list of SearchResult sorted by descending similarity.
        """
        k = top_k if top_k is not None else self.top_k
        if self.index.size == 0:
            raise RuntimeError("The index is empty. Please index documents first.")

        # 1 — Encode query
        query_vec = self.model.encode_query(query)          # (1, dim)

        # 2 — Retrieve embedding matrix
        matrix = self.index.get_matrix()                    # (N, dim)

        # 3 — Cosine similarity (dot product since both are L2-normalised)
        scores: np.ndarray = (matrix @ query_vec.T).squeeze()  # (N,)

        # 4 — Rank: argsort descending
        ranked_indices = np.argsort(scores)[::-1]

        # 5 — Build results
        results: List[SearchResult] = []
        for rank_i, idx in enumerate(ranked_indices, start=1):
            score = float(scores[idx])
            if score < threshold:
                break
            fragment = self.index.get_fragment(int(idx))
            results.append(
                SearchResult(
                    rank=rank_i,
                    fragment=fragment,
                    score=score,
                    score_percent=self._to_percent(score),
                )
            )
            if len(results) >= k:
                break

        logger.info(
            "Query: '%s' → %d results returned (top score: %.4f).",
            query[:80],
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_percent(score: float) -> float:
        """Convert cosine similarity [-1, 1] → percentage [0, 100]."""
        return round((score + 1) / 2 * 100, 2)

    def display_results(self, query: str, results: List[SearchResult]) -> None:
        """Pretty-print search results to stdout."""
        print("\n" + "═" * 70)
        print(f"  QUESTION : {query}")
        print("═" * 70)
        if not results:
            print("  Aucun résultat trouvé.")
        for res in results:
            print(res)
        print("═" * 70 + "\n")
