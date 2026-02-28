#!/usr/bin/env python
"""
ingest_to_pg.py

Script utilitaire pour alimenter la table PostgreSQL `embeddings` à partir
d'un répertoire de documents (fichiers texte, PDF, DOCX, CSV, JSON, etc.).

Rappel : les embeddings sont générés avec le modèle imposé
`all-MiniLM-L6-v2` et stockés dans la colonne `vecteur VECTOR(384)`.

Usage :
    python ingest_to_pg.py --folder data/enzymes
    python ingest_to_pg.py --folder /path/to/mes_fiches --clear

Options :
    --folder    : chemin du dossier contenant les documents à indexer.
    --clear     : effacer la table `embeddings` avant l'insertion.
    --db-host   : hôte PostgreSQL (passe sur value de config if omitted)
    --db-port   : port PostgreSQL
    --db-name   : nom de la base
    --db-user   : utilisateur
    --db-pass   : mot de passe

Ce script peut être exécuté depuis l'environnement virtuel du projet.
"""
import argparse
import os
import sys
import logging

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

import config
from rag_module.loader import DocumentLoader
from rag_module.pg_searcher import get_connection

# --------------------------------------------------
# Paramètres par défaut (reprennent ceux de config.py)
# --------------------------------------------------
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# --------------------------------------------------
# Fonctions
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents into PostgreSQL embeddings table")
    p.add_argument("--folder", required=True, help="Dossier contenant les fiches à indexer")
    p.add_argument("--clear", action="store_true", help="Vider la table avant l'insertion")
    p.add_argument("--db-host", default=None)
    p.add_argument("--db-port", type=int, default=None)
    p.add_argument("--db-name", default=None)
    p.add_argument("--db-user", default=None)
    p.add_argument("--db-pass", default=None)
    return p.parse_args()


def find_documents(folder: str) -> list[str]:
    """Renvoie la liste de chemins de fichiers supportés dans *folder*."""
    exts = (".txt", ".md", ".pdf", ".docx", ".csv", ".json")
    docs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                docs.append(os.path.join(root, fn))
    return sorted(docs)


def main():
    args = parse_args()

    # 1. Connexion à la base
    conn = get_connection(
        host=args.db_host or config.DB_HOST,
        port=args.db_port or config.DB_PORT,
        dbname=args.db_name or config.DB_NAME,
        user=args.db_user or config.DB_USER,
        password=args.db_pass or config.DB_PASSWORD,
    )
    cur = conn.cursor()

    if args.clear:
        cur.execute("TRUNCATE TABLE embeddings;")
        conn.commit()
        print("Table \"embeddings\" vidée.")

    # 2. Recherche de documents et chargement
    docs = find_documents(args.folder)
    if not docs:
        print(f"Aucun fichier trouvé dans {args.folder}.")
        sys.exit(1)

    loader = DocumentLoader(chunk_size=512, chunk_overlap=64)
    model = SentenceTransformer(DEFAULT_MODEL)
    print(f"Modèle d'embedding : {DEFAULT_MODEL}")

    doc_id = 0
    for path in docs:
        doc_id += 1
        print(f"Indexation du document #{doc_id} : {path}")
        fragments = loader.load_file(path)
        texts = [f.text for f in fragments]
        if not texts:
            continue
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        # insertion en base
        for frag, vec in zip(fragments, embeddings.tolist()):
            cur.execute(
                "INSERT INTO embeddings (id_document, texte_fragment, vecteur) VALUES (%s, %s, %s::vector)",
                (doc_id, frag.text, str(vec)),
            )
        conn.commit()
        print(f"  inséré {len(texts)} fragments")

    cur.close()
    conn.close()
    print("✅ Ingestion terminée.")


if __name__ == "__main__":
    main()
