# -*- coding: utf-8 -*-
"""app.py – Streamlit Front-End für die OOP-RAG-Pipeline (OpenRouter)"""

from __future__ import annotations

import streamlit as st

from ingest_docs import DocumentIngestor
from rag_graph import RAGPipeline
from pathlib import Path

# ---------------------------------------------------------------------------
# Grundlayout
st.set_page_config(page_title="PDF-RAG (OpenRouter)", page_icon="📄", layout="wide")
st.title("📄 Chat mit deinem PDF – OpenRouter Edition")

# ---------------------------------------------------------------------------
# Session-Singletons
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DocumentIngestor()   # nutzt .env-Chunk-Settings
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None                 # wird nach Ingestion gebaut
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None

# ---------------------------------------------------------------------------
# Sidebar - Document Management
with st.sidebar:
    st.header("📚 Dokument-Verwaltung")
    
    # List existing collections
    collections = st.session_state.ingestor.list_collections()
    
    if collections:
        st.subheader("Vorhandene Dokumente")
        for collection in collections:
            if st.button(f"📄 {collection}", key=f"select_{collection}"):
                st.session_state.selected_collection = collection
                st.session_state.pipeline = RAGPipeline(collection_name=collection)
                st.rerun()
    else:
        st.info("Keine Dokumente in der Datenbank gefunden.")
    
    st.divider()
    
    # Upload new document
    st.subheader("Neues Dokument hochladen")
    uploaded = st.file_uploader("PDF hochladen", type="pdf")
    
    if uploaded:
        if st.button("Dokument indexieren"):
            with st.spinner("Indexiere Dokument …"):
                tmp_path = "/tmp/upload.pdf"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # ► Dateiname (ohne .pdf) als Collection-Name verwenden
                collection = Path(uploaded.name).stem
                n_chunks = st.session_state.ingestor.ingest(tmp_path, collection)
                st.session_state.pipeline = RAGPipeline(collection_name=collection)
                st.session_state.selected_collection = collection
            st.success(f"✅ {n_chunks} Chunks indiziert.")
            st.rerun()

# ---------------------------------------------------------------------------
# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    
    # Show current document
    if st.session_state.selected_collection:
        st.info(f"Aktuelles Dokument: **{st.session_state.selected_collection}**")
    else:
        st.warning("Bitte wählen Sie ein Dokument aus der Seitenleiste oder laden Sie ein neues hoch.")
    
    # Chat-Interaktion
    prompt = st.chat_input("Frage stellen …")
    if prompt and st.session_state.pipeline:
        with st.spinner("Denke nach …"):
            answer = st.session_state.pipeline.answer(prompt)
        st.chat_message("assistant").write(answer)

with col2:
    st.header("📊 Dokument-Info")
    
    if st.session_state.selected_collection and st.session_state.pipeline:
        st.write(f"**Sammlung:** {st.session_state.selected_collection}")
        st.write("**Status:** Bereit für Fragen")
        
        # You could add more document statistics here
        # For example: number of chunks, document metadata, etc.
    else:
        st.write("Kein Dokument ausgewählt")
