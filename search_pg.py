#!/usr/bin/env python
"""
search_pg.py  —  Point d'entrée principal du module RAG PostgreSQL.

Usage :
    python search_pg.py                        # mode interactif
    python search_pg.py --query "votre question"   # one-shot
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

import config
from rag_module.pg_searcher import SemanticSearchPG, get_connection


def run(query: str | None = None) -> None:
    # ── Connexion ─────────────────────────────────────────────────────
    print("[INFO] Connexion à la base PostgreSQL…")
    try:
        conn = get_connection(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter : {e}")
        sys.exit(1)

    searcher = SemanticSearchPG(conn, top_k=3)

    # ── Boucle de recherche ───────────────────────────────────────────
    print("[INFO] Module prêt. Tapez 'quit' pour quitter.\n")
    while True:
        question = query or input("Question : ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        results = searcher.search(question)
        searcher.display_results(question, results)
        if query:          # mode one-shot : on quitte après une question
            break

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Semantic Search — PostgreSQL")
    parser.add_argument("--query", default=None, help="Question one-shot (optionnel)")
    args = parser.parse_args()
    run(args.query)
