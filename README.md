# 🔍 Module de Recherche Sémantique RAG
### STE AGRO MELANGE TECHNOLOGIE — ROSE BLANCHE Group

Un moteur de **Retrieval-Augmented Generation (RAG)** entièrement fonctionnel qui retrouve automatiquement les fragments documentaires les plus pertinents à partir d'une question en langage naturel, en s'appuyant sur la **similarité cosinus** dans l'espace vectoriel des embeddings avec **PostgreSQL + pgvector**.

---

## 📐 Architecture du système

```
question (langage naturel)
        │
        ▼
┌─────────────────────────────────┐
│   EmbeddingModel                │
│   all-MiniLM-L6-v2              │  (100+ langues : FR, EN, AR, etc.)
└────────┬───────────────────────┘
         │  vecteur requête (1 × 384)
         ▼
┌─────────────────────────────────┐
│  PostgreSQL + pgvector          │
│  SQL: 1 - (vecteur <=> query)   │  Distance cosinus inversée
│  ORDER BY score DESC LIMIT 3    │  HNSW index (haute performance)
└────────┬───────────────────────┘
         │  scores cosinus [0, 1]
         ▼
┌─────────────────────────────────┐
│   Top-3 fragments               │
│   + scores + document_id        │
└────────┬───────────────────────┘
         │
         ▼
   Résultats classés
```

---

## 🗂️ Structure du projet

```
projet/
├── rag_module/
│   ├── __init__.py              # Exports publics
│   ├── embeddings.py            # EmbeddingModel — vecteurs (SentenceTransformers)
│   ├── indexer.py               # VectorIndex — stockage fichier (pickle)
│   ├── loader.py                # DocumentLoader — extraction & chunking texte
│   ├── searcher.py              # SemanticSearcher — recherche fichier
│   ├── pg_searcher.py           # SemanticSearchPG — recherche PostgreSQL
│   └── pipeline.py              # RAGPipeline — façade fichier
├── data/
│   ├── documents/               # Exemples (3 TXT)
│   └── enzymes/                 # 35 fiches TDS (PDF) — dataset complet
├── *.sql / docker-compose.yml   # Support PostgreSQL local
├── config.py                    # DB credentials
├── app.py                       # Interface Streamlit (optionnel)
├── main.py                      # CLI mode fichier
├── search_pg.py                 # CLI mode PostgreSQL
├── ingest_to_pg.py              # Ingestion documents → table embeddings
├── demo_rag.ipynb               # Notebook PostgreSQL complet (9 sections)
└── requirements.txt
```

---

## ⚡ Installation rapide

### 1. Environnement virtuel

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Inclut : `sentence-transformers`, `psycopg2`, `pdfplumber`, `python-docx`, `pandas`, `numpy`, `scikit-learn`, `streamlit`.

### 3. Démarrer PostgreSQL + pgvector via Docker

```bash
docker compose up -d
```

**Résultat** : Conteneur `rag_postgres` sur `localhost:5432`
- Base de données : `boulangerie_db`
- Utilisateur : `rag_user`
- Mot de passe : `rag_secret`

**Vérifier** :
```bash
docker ps
```

---

## 🚀 Utilisation

### Option 1 — Mode PostgreSQL (⭐ Recommandé)

#### a) Ingérer des documents

```bash
python ingest_to_pg.py --folder data/enzymes
```

Charge tous les PDFs dans la table `embeddings` PostgreSQL.

#### b) Recherche interactive

```bash
python search_pg.py
```

Tapez votre question :
```
Quelles sont les quantités recommandées d'alpha-amylase et d'acide ascorbique ?
```

Résultat : Top 3 fragments avec scores cosinus.

#### c) Notebook Jupyter

```bash
jupyter notebook demo_rag.ipynb
```

9 sections : connexion DB → modèle → requête SQL → résultats.

### Option 2 — Mode fichier (sans DB)

```bash
# Indexer un dossier
python main.py index --source data/documents/ --save data/index.pkl

# Recherche interactive
python main.py search --index data/index.pkl

# En une commande
python main.py run --source data/documents/ --query "Quelles sont les recommandations HACCP ?"
```

### Option 3 — API Python

```python
from rag_module.pg_searcher import semantic_search

results = semantic_search(
    query="Xylanase : dosages et applications",
    top_k=3
)

for r in results:
    print(f"Score {r.score:.3f}: {r.text[:100]}...")
```

### Option 4 — Interface Web Streamlit

L'utilitaire `app.py` fournit une interface graphique interactive.

1. Assurez‑vous d'avoir préparé votre environnement et installé les dépendances :

    ```bash
    # activer virtualenv puis :
    pip install -r requirements.txt   # contient streamlit
    ```

2. Lancer le serveur Streamlit :

    ```bash
    streamlit run app.py
    ```

3. Une fois démarré, le navigateur ouvrira automatiquement (ou visitez manuellement)
   l'URL : `http://localhost:8501`.

4. **Changer le thème** : cliquez sur l'icône en forme de hamburger en haut à droite
   → *Settings* → *Theme* → sélectionnez *Light* ou *Dark*, ou laissez le navigateur
   suivre le mode clair/sombre du système. Les cartes (`.result-card`, `.metric-card`) et le conteneur Markdown
   s'adaptent automatiquement à la couleur de fond en mode clair ou sombre.

> ⚠️ Si vous rencontrez des problèmes de mise en forme du texte markdown, la
> correction a été ajoutée pour que le fond de `.stMarkdownContainer` soit
> transparent / hérite du thème, rendant l'affichage compatible avec les modes
> clair et sombre.

---

## 📊 Formats de documents supportés

| Format | Extension | Outil |
|--------|-----------|-------|
| Texte brut | `.txt`, `.md` | Python natif |
| PDF | `.pdf` | `pdfplumber` |
| Word | `.docx` | `python-docx` |
| CSV | `.csv` | `csv` module |
| JSON | `.json` | `json` module |

---

## 🧠 Modèle d'embedding

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `all-MiniLM-L6-v2` |
| Langues | 100+ (français, anglais, arabe, chinois…) |
| Dimension | 384 |
| Normalisation | L2 (cosinus = dot product) |
| Source | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

---

## 🗄️ PostgreSQL + pgvector

### Schéma de la table

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    id_document INT,
    texte_fragment TEXT,
    vecteur VECTOR(384)
);

CREATE INDEX idx_vecteur ON embeddings USING HNSW (vecteur vector_cosine_ops);
```

### Avantages

- Persistence en base de données
- HNSW index pour recherche O(log N)
- Scalabilité : millions de fragments
- Requêtes SQL natives

---

## 📈 Dataset

Le projet inclut **35 fiches TDS enzymes** pour panification :

- **BVZyme** : GOX, TG (MAX63, MAX64, 881, 883), GO, A FRESH, A SOFT, AF, AMG, L MAX, HCB, HCF
- **Acide ascorbique** : utilisation culinaire et dosage

**Total** : ~190 fragments (après chunking)

```
data/enzymes/
├── BVZyme GOX 110 TDS(1).pdf
├── BVZyme TG MAX63 TDS.pdf
├── ... (35 fichiers)
└── acide ascorbique.pdf
```

---

## 🔬 Formule mathématique

Similarité cosinus entre requête **Q** et fragment **D_i** :

$$\text{sim}(Q, D_i) = \frac{Q \cdot D_i}{\|Q\| \cdot \|D_i\|}$$

Puisque tous les vecteurs sont **L2-normalisés**, cela se simplifie en :

$$\text{sim}(Q, D_i) = Q \cdot D_i$$

### SQL PostgreSQL

```sql
SELECT id, texte_fragment, (1 - (vecteur <=> %s::vector)) AS score
FROM embeddings
ORDER BY score DESC
LIMIT 3;
```

---

## 🐳 Gestion Docker

```bash
# Démarrer
docker compose up -d

# Arrêter (données conservées)
docker compose down

# Arrêter + supprimer données
docker compose down -v

# Vérifier l'état
docker ps
docker logs rag_postgres

# Accès SQL
docker exec -it rag_postgres psql -U rag_user -d boulangerie_db
```

---

## ✅ Workflow complet

```bash
# 1. Activation environnement
.venv\Scripts\activate

# 2. Démarrer BD
docker compose up -d

# 3. Ingérer documents
python ingest_to_pg.py --folder data/enzymes

# 4. Recherche
python search_pg.py

# 5. Arrêter (optionnel)
docker compose down
```

---

## 🔧 Dépannage

### Erreur : `psycopg2` introuvable

```bash
pip install psycopg2-binary --user
```

### Erreur : `pdfplumber` introuvable

```bash
pip install pdfplumber --user
```

### Erreur : PostgreSQL connection refused

- Vérifier : `docker ps` → `rag_postgres` doit être `Up`
- Attendre 5 secondes après `docker compose up -d`
- Vérifier `config.py` : credentials correctes

### Erreur : MemoryError lors ingestion

Cause : Bug de boucle infinie dans `_split_text()`  
Status : **FIXÉ** dans `rag_module/loader.py`

---

## 👤 Informations du projet

| Champ | Valeur |
|-------|--------|
| Email | a.changuel@rose-blanche.com |
| Organisation | STE AGRO MELANGE TECHNOLOGIE — ROSE BLANCHE Group |
| Gratification | Chèque cadeau 1000 DT |
| Livrable | Prototype RAG PostgreSQL complet |
| Langues | FR, EN, AR, +100 autres |
| Dataset | 35 fiches TDS enzymes |
| Modèle | `all-MiniLM-L6-v2` (384 dimensions) |
| Base de données | PostgreSQL 16 + pgvector 0.8.2 |

---

**Status** : ✅ Production-ready — Tous les composants testés et fonctionnels.
