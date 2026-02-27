"""
pipeline.py
High-level façade that wires loader → embeddings → index → searcher together.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .embeddings import EmbeddingModel
from .indexer import Fragment, VectorIndex
from .loader import DocumentLoader
from .searcher import SearchResult, SemanticSearcher

logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_INDEX_PATH = "data/vector_index.pkl"
DEFAULT_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"


class RAGPipeline:
    """
    All-in-one RAG pipeline.

    Quick start::

        pipeline = RAGPipeline()
        pipeline.index_directory("data/documents/")
        results  = pipeline.search("Quelles sont les recommandations sanitaires ?")
        pipeline.display(results)

    Persistence::

        pipeline.save_index("data/index.pkl")
        pipeline.load_index("data/index.pkl")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 3,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        self.index_path = index_path
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.loader          = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index           = VectorIndex()
        self.searcher        = SemanticSearcher(self.embedding_model, self.index, top_k=top_k)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, path: str) -> int:
        """Index a single file. Returns the number of new fragments added."""
        fragments = self.loader.load_file(path)
        return self._add_fragments(fragments)

    def index_directory(self, directory: str, recursive: bool = True) -> int:
        """Index all supported files in *directory*. Returns total new fragments."""
        fragments = self.loader.load_directory(directory, recursive=recursive)
        return self._add_fragments(fragments)

    def index_texts(self, texts: List[str], source: str = "inline") -> int:
        """Index raw text strings directly."""
        fragments = self.loader.load_texts(texts, source=source)
        return self._add_fragments(fragments)

    def _add_fragments(self, fragments: List[Fragment]) -> int:
        if not fragments:
            logger.warning("No usable fragments found.")
            return 0
        texts = [f.text for f in fragments]
        embeddings = self.embedding_model.encode_documents(texts, show_progress=True)
        self.index.add(fragments, embeddings)
        return len(fragments)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: float = -1.0,
    ) -> List[SearchResult]:
        """Perform semantic search and return ranked results."""
        return self.searcher.search(query, top_k=top_k, threshold=threshold)

    def display(self, query: str, results: List[SearchResult]) -> None:
        """Pretty-print results to stdout."""
        self.searcher.display_results(query, results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: Optional[str] = None) -> None:
        self.index.save(path or self.index_path)

    def load_index(self, path: Optional[str] = None) -> None:
        loaded = VectorIndex.load(path or self.index_path)
        self.index._fragments = loaded._fragments
        self.index._matrix    = loaded._matrix

    def export_metadata(self, path: str = "data/index_metadata.json") -> None:
        self.index.export_json(path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fragment_count(self) -> int:
        return self.index.size
