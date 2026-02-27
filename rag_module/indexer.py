"""
indexer.py
Manages the in-memory vector index: stores fragment texts + their embeddings,
persists to / loads from disk.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fragment:
    """A single indexed document fragment."""
    id: int
    text: str
    source: str = ""          # filename / document title
    page: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class VectorIndex:
    """
    Lightweight in-memory vector store.

    Stores:
        • A list of Fragment objects (text + metadata).
        • A 2-D float32 matrix of their embeddings (n_fragments × dim).

    Provides fast cosine similarity search via NumPy.

    Usage::

        index = VectorIndex()
        index.add(fragments, embeddings)
        index.save("my_index.pkl")

        index2 = VectorIndex.load("my_index.pkl")
    """

    def __init__(self) -> None:
        self._fragments: List[Fragment] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed fragments."""
        return len(self._fragments)

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimension of stored embeddings, or None if empty."""
        return self._matrix.shape[1] if self._matrix is not None else None

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def add(self, fragments: List[Fragment], embeddings: np.ndarray) -> None:
        """
        Add fragments and their corresponding embeddings to the index.

        Args:
            fragments:  list of Fragment objects, length N.
            embeddings: np.ndarray of shape (N, dim), already L2-normalised.
        """
        if len(fragments) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(fragments)} fragments vs {embeddings.shape[0]} embeddings."
            )

        # Reassign IDs to be global
        offset = len(self._fragments)
        for i, frag in enumerate(fragments):
            frag.id = offset + i

        self._fragments.extend(fragments)

        if self._matrix is None:
            self._matrix = embeddings.astype(np.float32)
        else:
            self._matrix = np.vstack(
                [self._matrix, embeddings.astype(np.float32)]
            )

        logger.info(
            "Index updated: %d fragments total (added %d).",
            self.size,
            len(fragments),
        )

    def clear(self) -> None:
        """Reset the index."""
        self._fragments = []
        self._matrix = None

    # ------------------------------------------------------------------
    # Retrieval helpers (used by SemanticSearcher)
    # ------------------------------------------------------------------

    def get_matrix(self) -> np.ndarray:
        """Return the embedding matrix, raising if empty."""
        if self._matrix is None:
            raise RuntimeError("Index is empty. Index documents first.")
        return self._matrix

    def get_fragment(self, idx: int) -> Fragment:
        return self._fragments[idx]

    def get_all_fragments(self) -> List[Fragment]:
        return list(self._fragments)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index to disk (pickle)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"fragments": self._fragments, "matrix": self._matrix}, f
            )
        logger.info("Index saved to %s (%d fragments).", path, self.size)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._fragments = data["fragments"]
        idx._matrix = data["matrix"]
        logger.info("Index loaded from %s (%d fragments).", path, idx.size)
        return idx

    def export_json(self, path: str) -> None:
        """Export metadata (without embeddings) to JSON for inspection."""
        records = [
            {
                "id": f.id,
                "source": f.source,
                "page": f.page,
                "text_preview": f.text[:200],
                "metadata": f.metadata,
            }
            for f in self._fragments
        ]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(records, fp, ensure_ascii=False, indent=2)
        logger.info("Metadata exported to %s.", path)
