# -*- coding: utf-8 -*-
"""app.py – Streamlit Front-End für die OOP-RAG-Pipeline (OpenRouter)"""

from __future__ import annotations

import streamlit as st

from ingest_docs import DocumentIngestor
from rag_graph import RAGPipeline
from pathlib import Path

# ---------------------------------------------------------------------------
# Grundlayout
st.set_page_config(page_title="PDF-RAG (OpenRouter)", page_icon="📄")
st.title("📄 Chat mit deinem PDF – OpenRouter Edition")

# ---------------------------------------------------------------------------
# Session-Singletons
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DocumentIngestor()   # nutzt .env-Chunk-Settings
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None                 # wird nach Ingestion gebaut

# ---------------------------------------------------------------------------
# PDF-Upload & Indexieren
uploaded = st.file_uploader("PDF hochladen", type="pdf")
if uploaded and st.session_state.pipeline is None:
    with st.spinner("Indexiere Dokument …"):
        tmp_path = "/tmp/upload.pdf"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # ► Dateiname (ohne .pdf) als Collection-Name verwenden
        collection = Path(uploaded.name).stem
        n_chunks = st.session_state.ingestor.ingest(tmp_path, collection)
        st.session_state.pipeline = RAGPipeline(collection_name=collection)
    st.success(f"✅ {n_chunks} Chunks indiziert.")

# ---------------------------------------------------------------------------
# Chat-Interaktion
prompt = st.chat_input("Frage stellen …")
if prompt and st.session_state.pipeline:
    with st.spinner("Denke nach …"):
        answer = st.session_state.pipeline.answer(prompt)
    st.chat_message("assistant").write(answer)
