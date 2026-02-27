"""
embeddings.py
Generates semantic embeddings using sentence-transformers.
Supports multilingual models (French, English, Arabic).
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper around a SentenceTransformer model to generate dense vector embeddings
    for both documents (fragments) and queries.

    Args:
        model_name: HuggingFace model identifier.
                    Default: 'paraphrase-multilingual-MiniLM-L12-v2'
                    (supports French, English, Arabic, etc.)
        device: 'cpu' or 'cuda'. Auto-detected if None.
        normalize: If True, embeddings are L2-normalised (recommended for cosine sim).
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("Model ready — embedding dimension: %d", self.dimension)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or many texts into dense float32 vectors.

        Returns:
            np.ndarray of shape (N, dim) — always 2-D even for a single text.
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → shape (1, dim)."""
        return self.encode(query)

    def encode_documents(
        self, documents: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Encode a list of document fragments → shape (N, dim)."""
        logger.info("Encoding %d documents…", len(documents))
        return self.encode(documents, show_progress=show_progress)
