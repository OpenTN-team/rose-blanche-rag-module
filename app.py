"""
app.py  —  Streamlit web interface for the RAG Semantic Search module.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Recherche Sémantique",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title   { font-size: 2.2rem; font-weight: 800; color: #1a3c6e; }
    .sub-title    { font-size: 1.05rem; color: #555; margin-bottom: 1.5rem; }
    .result-card  {
        background: #f8faff;
        border-left: 5px solid #1a3c6e;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .score-badge  {
        display: inline-block;
        background: #1a3c6e;
        color: white;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .source-tag   { color: #888; font-size: 0.82rem; margin-top: 0.25rem; }
    .metric-card  {
        background: #eef2fa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline  (cached across reruns)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the pipeline once and cache it in session."""
    from rag_module.pipeline import RAGPipeline
    return RAGPipeline(top_k=3)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — document upload / index management
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    top_k = st.slider("Nombre de résultats", min_value=1, max_value=10, value=3)
    threshold = st.slider(
        "Score minimum (similarité)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )

    st.markdown("---")
    st.markdown("### 📂 Charger des documents")

    upload_mode = st.radio(
        "Mode de chargement",
        ["Téléverser des fichiers", "Dossier local (chemin)"],
    )

    if upload_mode == "Téléverser des fichiers":
        uploaded_files = st.file_uploader(
            "Sélectionner des fichiers",
            type=["txt", "pdf", "docx", "csv", "json", "md"],
            accept_multiple_files=True,
        )

        if st.button("🔄 Indexer les documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Aucun fichier sélectionné.")
            else:
                pipeline = get_pipeline()
                pipeline.index.clear()
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uf in uploaded_files:
                        dest = os.path.join(tmpdir, uf.name)
                        with open(dest, "wb") as f:
                            f.write(uf.read())
                    with st.spinner("Génération des embeddings…"):
                        n = pipeline.index_directory(tmpdir)
                    st.success(f"{n} fragments indexés avec succès ✅")
                    st.session_state["index_ready"] = True
                    st.session_state["file_names"] = [uf.name for uf in uploaded_files]

    else:
        folder_path = st.text_input("Chemin du dossier local", value="data/documents/")
        if st.button("🔄 Indexer le dossier", use_container_width=True):
            if not os.path.isdir(folder_path):
                st.error(f"Dossier introuvable : {folder_path}")
            else:
                pipeline = get_pipeline()
                pipeline.index.clear()
                with st.spinner("Génération des embeddings…"):
                    n = pipeline.index_directory(folder_path)
                st.success(f"{n} fragments indexés avec succès ✅")
                st.session_state["index_ready"] = True

    st.markdown("---")
    st.markdown("### 💾 Gérer l'index")

    col_a, col_b = st.columns(2)
    with col_a:
        save_path = st.text_input("Chemin de sauvegarde", value="data/index.pkl", label_visibility="collapsed")
        if st.button("💾 Sauvegarder", use_container_width=True):
            try:
                get_pipeline().save_index(save_path)
                st.success("Index sauvegardé ✅")
            except Exception as e:
                st.error(str(e))
    with col_b:
        load_path = st.text_input("Chemin de chargement", value="data/index.pkl", label_visibility="collapsed")
        if st.button("📂 Charger", use_container_width=True):
            try:
                get_pipeline().load_index(load_path)
                st.success(f"{get_pipeline().fragment_count} fragments chargés ✅")
                st.session_state["index_ready"] = True
            except Exception as e:
                st.error(str(e))

    # Stats
    if st.session_state.get("index_ready"):
        pipeline = get_pipeline()
        st.markdown("---")
        st.markdown("### 📊 Statistiques")
        st.metric("Fragments dans l'index", pipeline.fragment_count)


# ──────────────────────────────────────────────────────────────────────────────
# Main area — Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🔍 Recherche Sémantique RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interrogez votre base documentaire en langage naturel '
    "grâce à la similarité vectorielle (cosinus).</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Search bar
# ──────────────────────────────────────────────────────────────────────────────

with st.form("search_form", clear_on_submit=False):
    query = st.text_area(
        "💬 Posez votre question :",
        placeholder="Ex : Quelles sont les recommandations pour améliorer la sécurité alimentaire ?",
        height=100,
    )
    submitted = st.form_submit_button("🔎 Rechercher", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

if submitted:
    if not query.strip():
        st.warning("Veuillez entrer une question.")
    elif not st.session_state.get("index_ready") and get_pipeline().fragment_count == 0:
        st.error(
            "L'index est vide. Veuillez charger ou téléverser des documents "
            "depuis le panneau latéral."
        )
    else:
        pipeline = get_pipeline()
        t0 = time.perf_counter()

        # Compute threshold: slider is [0,1] → map to cosine [-1, 1]
        cos_threshold = threshold * 2 - 1

        try:
            results = pipeline.search(query, top_k=top_k, threshold=cos_threshold)
            elapsed = time.perf_counter() - t0

            st.markdown(f"**{len(results)} résultat(s)** trouvé(s) en `{elapsed:.3f}s`")
            st.markdown("---")

            if not results:
                st.info("Aucun fragment ne dépasse le seuil de similarité défini.")
            else:
                for res in results:
                    score_pct = res.score_percent
                    color = (
                        "#2ecc71" if score_pct >= 70
                        else "#f39c12" if score_pct >= 50
                        else "#e74c3c"
                    )
                    source_info = f"📄 {res.fragment.source}"
                    if res.fragment.page:
                        source_info += f"  —  page {res.fragment.page}"

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div>
                                <span style="font-weight:700; font-size:1rem;">
                                    Résultat #{res.rank}
                                </span>
                                &nbsp;&nbsp;
                                <span style="background:{color}; color:white;
                                      border-radius:20px; padding:2px 12px;
                                      font-size:0.85rem; font-weight:700;">
                                    Similarité : {res.score:.4f} &nbsp;|&nbsp; {score_pct:.1f}%
                                </span>
                            </div>
                            <div class="source-tag">{source_info}</div>
                            <hr style="border:none; border-top:1px solid #dde3f0; margin:0.6rem 0;">
                            <div style="white-space:pre-wrap; line-height:1.6;">
                                {res.fragment.text.strip()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Similarity bar chart
                with st.expander("📊 Visualisation des scores"):
                    import pandas as pd

                    chart_data = pd.DataFrame({
                        "Résultat": [f"#{r.rank}" for r in results],
                        "Similarité (%)": [r.score_percent for r in results],
                        "Score cosinus": [round(r.score, 4) for r in results],
                    })
                    st.bar_chart(chart_data.set_index("Résultat")["Similarité (%)"])
                    st.dataframe(chart_data, use_container_width=True, hide_index=True)

        except RuntimeError as e:
            st.error(str(e))

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "RAG Semantic Search Module · paraphrase-multilingual-MiniLM-L12-v2 · Cosine Similarity"
    "</div>",
    unsafe_allow_html=True,
)
