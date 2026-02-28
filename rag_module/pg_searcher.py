"""
pg_searcher.py
Module de recherche sémantique RAG sur base PostgreSQL + pgvector.

Spec imposée :
  - Modèle  : all-MiniLM-L6-v2  (sentence-transformers, dim=384)
  - Table   : embeddings  (id, id_document, texte_fragment, vecteur VECTOR(384))
  - Méthode : Cosine Similarity
  - Top-K   : 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constantes imposées ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
TOP_K_DEFAULT   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Résultat de recherche
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    rank: int
    id: int
    id_document: int
    texte_fragment: str
    score: float          # cosine similarity ∈ [0, 1]

    def display(self) -> None:
        border = "─" * 60
        print(f"\n{border}")
        print(f"  Résultat {self.rank}")
        print(f"  Texte  : {self.texte_fragment.strip()}")
        print(f"  Score  : {self.score:.4f}")
        print(border)


# ─────────────────────────────────────────────────────────────────────────────
# Connexion PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(
    host: str     = "localhost",
    port: int     = 5432,
    dbname: str   = "boulangerie_db",
    user: str     = "rag_user",
    password: str = "rag_secret",
) -> psycopg2.extensions.connection:
    """
    Ouvre et retourne une connexion psycopg2.
    Toutes les valeurs peuvent être surchargées ou lues depuis .env.
    """
    conn = psycopg2.connect(
        host=host, port=port,
        dbname=dbname, user=user, password=password
    )
    logger.info("Connexion PostgreSQL établie (%s:%d/%s)", host, port, dbname)
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Moteur de recherche sémantique
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSearchPG:
    """
    Recherche sémantique sur la table PostgreSQL `embeddings`.

    Usage::

        searcher = SemanticSearchPG(conn)
        results  = searcher.search("Alpha-amylase : dosage recommandé ?")
        searcher.display_results(results)
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        top_k: int = TOP_K_DEFAULT,
    ) -> None:
        self.conn  = conn
        self.top_k = top_k

        logger.info("Chargement du modèle d'embedding : %s", EMBEDDING_MODEL)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Modèle prêt (dim=%d)", EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def search(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        1. Génère l'embedding de la question.
        2. Interroge PostgreSQL via la similarité cosinus (pgvector).
        3. Retourne les top_k fragments classés par score décroissant.
        """
        k = top_k or self.top_k

        # validate connection is still open (psycopg2 sets .closed=0 when open)
        if getattr(self.conn, "closed", 1):
            raise psycopg2.InterfaceError(
                "La connexion PostgreSQL est fermée. "
                "Appelez get_connection() pour en obtenir une nouvelle."
            )
        # ── Étape 1 : Embedding de la question ────────────────────────
        q_vec = self._model.encode(question, normalize_embeddings=True)
        q_vec_list = str(q_vec.tolist())     # format attendu par pgvector ('[0.1, 0.2, ...]')

        # ── Étape 2 : Requête SQL avec similarité cosinus ─────────────
        # pgvector : 1 - (vecteur <=> query)  = cosine similarity
        sql = """
            SELECT
                id,
                id_document,
                texte_fragment,
                1 - (vecteur <=> %s::vector) AS score
            FROM embeddings
            ORDER BY score DESC
            LIMIT %s;
        """

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (q_vec_list, k))
            rows = cur.fetchall()

        # ── Étape 3 : Construction des résultats ──────────────────────
        results = [
            SearchResult(
                rank=i + 1,
                id=row["id"],
                id_document=row["id_document"],
                texte_fragment=row["texte_fragment"],
                score=float(row["score"]),
            )
            for i, row in enumerate(rows)
        ]

        logger.info(
            "Question : '%s' → %d résultat(s) | top score : %.4f",
            question[:80], len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def display_results(self, question: str, results: List[SearchResult]) -> None:
        """Affiche les résultats dans la console."""
        print("\n" + "═" * 60)
        print(f"  QUESTION : {question}")
        print("═" * 60)
        if not results:
            print("  Aucun résultat trouvé.")
        for res in results:
            res.display()
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Fonction autonome (usage direct sans instancier la classe)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(
    question: str,
    conn: psycopg2.extensions.connection,
    top_k: int = TOP_K_DEFAULT,
) -> List[SearchResult]:
    """
    Fonction utilitaire : encode la question et interroge la base.
    Retourne une liste de SearchResult triée par score décroissant.
    """
    searcher = SemanticSearchPG(conn, top_k=top_k)
    return searcher.search(question, top_k=top_k)
