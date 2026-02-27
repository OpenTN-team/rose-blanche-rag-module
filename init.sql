-- init.sql
-- Exécuté automatiquement au premier démarrage du conteneur PostgreSQL.

-- Active l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Crée la table imposée par le cahier des charges
CREATE TABLE IF NOT EXISTS embeddings (
    id           SERIAL PRIMARY KEY,
    id_document  INTEGER NOT NULL,
    texte_fragment TEXT   NOT NULL,
    vecteur      VECTOR(384)            -- dimension all-MiniLM-L6-v2
);

-- Index HNSW pour une recherche cosinus rapide (optionnel mais recommandé)
CREATE INDEX IF NOT EXISTS idx_embeddings_cosine
    ON embeddings
    USING hnsw (vecteur vector_cosine_ops);
