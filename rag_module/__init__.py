"""
RAG Semantic Search Module
Retrieval-Augmented Generation for intelligent document retrieval.
"""

from .embeddings import EmbeddingModel
from .indexer import VectorIndex
from .searcher import SemanticSearcher
from .loader import DocumentLoader
from .pg_searcher import SemanticSearchPG, semantic_search, get_connection

__all__ = [
    "EmbeddingModel", "VectorIndex", "SemanticSearcher", "DocumentLoader",
    "SemanticSearchPG", "semantic_search", "get_connection",
]
__version__ = "1.0.0"
