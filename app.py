from __future__ import annotations
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from ingest_docs import DocumentIngestor
from rag_graph import RAGPipeline
from pathlib import Path
from config import RAGConfig

# ---------------------------------------------------------------------------
# Grundlayout
st.set_page_config(page_title="PDF-RAG (OpenRouter)", page_icon="", layout="wide")
st.title("PDF-Chat")

# ---------------------------------------------------------------------------
# Session-Singletons
if "cfg" not in st.session_state:
    st.session_state.cfg = RAGConfig()
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DocumentIngestor(cfg=st.session_state.cfg)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None
if "pdf_page" not in st.session_state:
    st.session_state.pdf_page = 1
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar 
with st.sidebar:
    st.header("Parameter")

    # initiale Defaults aus cfg oder Session
    def _dflt(name, fallback):
        return st.session_state.get(name, getattr(st.session_state.cfg, name, fallback))

    retrieval_k    = st.number_input("Retrieval k",       1,  20, _dflt("retrieval_k",      4))
    temperature    = st.slider      ("Temperature",    0.0, 2.0, _dflt("temperature",    0.3), 0.05)
    llm_model      = st.text_input  ("LLM-Model",          _dflt("openrouter_model",  "openai/o4-mini"))
    #embed_model    = st.text_input  ("Embedding-Model",    _dflt("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

    if st.button("Anwenden"):
        # 2a – neue Config zusammenbauen
        st.session_state.cfg = RAGConfig(
            retrieval_k    = retrieval_k,
            temperature    = temperature,
            openrouter_model = llm_model,
            #embedding_model= embed_model,
            # Pflichtfelder, die du NICHT änderst, automatisch aus .env
        )

        # 2b – Ingestor & Pipeline neu instanziieren
        st.session_state.ingestor = DocumentIngestor(cfg=st.session_state.cfg)
        if st.session_state.selected_collection:
            st.session_state.pipeline = RAGPipeline(
                collection_name = st.session_state.selected_collection,
                cfg             = st.session_state.cfg,
            )
        st.rerun()


# - Document Management
with st.sidebar:
    st.header("Dokument-Verwaltung")
    # List existing collections
    collections = st.session_state.ingestor.list_collections()
    if collections:
        st.subheader("Vorhandene Dokumente")
        for collection in collections:
            col_btn, del_btn = st.columns([3, 1])
            
            with col_btn:
                if st.button(f"{collection}", key=f"select_{collection}"):
                    st.session_state.selected_collection = collection
                    st.session_state.pipeline = RAGPipeline(collection_name=collection, cfg=st.session_state.cfg)
                    st.rerun()
            
            with del_btn:
                if st.button("🗑️", key=f"delete_{collection}", help=f"Lösche {collection}"):
                    if st.session_state.ingestor.delete_collection(collection):
                        # Clear selection if deleted collection was selected
                        if st.session_state.selected_collection == collection:
                            st.session_state.selected_collection = None
                            st.session_state.pipeline = None
                        st.success(f"'{collection}' wurde gelöscht.")
                        st.rerun()
                    else:
                        st.error(f"Fehler beim Löschen von '{collection}'.")
    else:
        st.info("Keine Dokumente in der Datenbank gefunden.")
    
    st.divider()
    
    # Upload new document
    st.subheader("Neues Dokument hochladen")
    uploaded = st.file_uploader("PDF hochladen", type="pdf")
    
    if uploaded:
        # Chunk settings for this document
        chunk_size = st.number_input("Chunk Size", 256, 4096, _dflt("chunk_size", 1024), step=256)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 2048, _dflt("chunk_overlap", 64), step=32)
        
        # Show suggested collection name and allow editing
        suggested_name = st.session_state.ingestor.sanitize_collection_name(Path(uploaded.name).stem)
        collection_name = st.text_input(
            "Collection Name:", 
            value=suggested_name,
            help="Name für die Dokumentensammlung (3-512 Zeichen, nur a-z, A-Z, 0-9, ., _, -)"
        )
        
        if st.button("Dokument indexieren", disabled=len(collection_name.strip()) < 3):
            # Sanitize the user input
            collection = st.session_state.ingestor.sanitize_collection_name(collection_name.strip())
            
            if collection != collection_name.strip():
                st.warning(f"Collection Name wurde angepasst zu: '{collection}'")
            
            with st.spinner("Indexiere Dokument …"):
                try:
                    # Create a temporary config with the chunk settings for this document
                    upload_cfg = RAGConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        # Keep other settings from current config
                        retrieval_k=st.session_state.cfg.retrieval_k,
                        temperature=st.session_state.cfg.temperature,
                        openrouter_model=st.session_state.cfg.openrouter_model,
                    )
                    
                    # Create temporary ingestor with upload config
                    upload_ingestor = DocumentIngestor(cfg=upload_cfg)
                    
                    tmp_path = "/tmp/upload.pdf"
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    n_chunks = upload_ingestor.ingest(tmp_path, collection)
                    st.session_state.pipeline = RAGPipeline(collection_name=collection, cfg=st.session_state.cfg)
                    st.session_state.selected_collection = collection
                    st.success(f"{n_chunks} Chunks indiziert als '{collection}' (Chunk Size: {chunk_size}, Overlap: {chunk_overlap}).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler beim Indexieren: {str(e)}")

# ---------------------------------------------------------------------------
# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Chat")
    
    # Show current document
    if st.session_state.selected_collection:
        st.info(f"Aktuelles Dokument: **{st.session_state.selected_collection}**")
    else:
        st.warning("Bitte wählen Sie ein Dokument aus der Seitenleiste oder laden Sie ein neues hoch.")
    
    # Display chat history
    for i, (question, answer, chunks) in enumerate(st.session_state.chat_history):
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)
        
        # Display retrieved chunks
        if chunks:
            with st.expander(f"📄 Verwendete Textabschnitte ({len(chunks)} Chunks)", expanded=False):
                for j, chunk in enumerate(chunks):
                    st.write(f"**Chunk {j+1} (Seite {chunk['page']}):**")
                    st.text_area(
                        f"Inhalt {j+1}:",
                        value=chunk['content'],
                        height=100,
                        key=f"chunk_{i}_{j}",
                        disabled=True
                    )
                    st.divider()
        
        # Extract page numbers from answer and create clickable buttons
        import re
        page_matches = re.findall(r'Seite (\d+)', answer)
        if page_matches:
            st.write("**Zu Seiten springen:**")
            for page_num in sorted(set(page_matches), key=int):  # sortiert und Duplikate entfernt
                if st.button(f"Seite {page_num}", key=f"goto_page_{page_num}_{i}"):
                    st.session_state.pdf_page = int(page_num)
                    st.rerun()

    # Chat-Interaktion
    prompt = st.chat_input("Frage stellen …")
    if prompt and st.session_state.pipeline:
        with st.spinner("Denke nach …"):
            answer, chunks = st.session_state.pipeline.answer(prompt)
        
        # Add to chat history
        st.session_state.chat_history.append((prompt, answer, chunks))
        st.rerun()

with col2:
    st.header("PDF Viewer")
    
    if st.session_state.selected_collection:
        pdf_path = st.session_state.ingestor.get_pdf_path(st.session_state.selected_collection)
        
        if pdf_path and pdf_path.exists():
            try:
                # Read PDF file
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            except Exception as e:
                st.error(f"Fehler beim Laden der PDF: {str(e)}")
                pdf_bytes = None
        else:
            pdf_bytes = None
        
        if pdf_bytes:
            
            # Display PDF using streamlit-pdf-viewer
            st.write("**PDF Dokument:**")
            st.write(f" **{st.session_state.selected_collection}.pdf**")
            
            # Use streamlit-pdf-viewer for better PDF display
            # Force re-render by changing key when page changes
            viewer_key = f"pdf_viewer_{st.session_state.selected_collection}_page_{st.session_state.pdf_page}"
            
            pdf_viewer(
                input=pdf_bytes,
                width=500,
                height=700,
                key=viewer_key,
                pages_to_render=[st.session_state.pdf_page],  # Only render current page
                render_text=True
            )
            
            # Page navigation controls
            st.write(f"**Aktuelle Seite:** {st.session_state.pdf_page}")
            col_prev, col_next, col_goto = st.columns([1, 1, 2])
            
            with col_prev:
                if st.button("Vorherige", disabled=st.session_state.pdf_page <= 1):
                    st.session_state.pdf_page = max(1, st.session_state.pdf_page - 1)
                    st.rerun()
            
            with col_next:
                if st.button("Nächste"):
                    st.session_state.pdf_page += 1
                    st.rerun()
            
            with col_goto:
                page_input = st.number_input("Gehe zu Seite:", min_value=1, value=st.session_state.pdf_page, key="page_input")
                if st.button("Gehe zu Seite"):
                    st.session_state.pdf_page = page_input
                    st.rerun()
            
                # Show PDF info
                st.write(f"**Dateigröße:** {len(pdf_bytes) / 1024:.1f} KB")
                
                # Download button
                st.download_button(
                    label="PDF herunterladen",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.selected_collection}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("PDF-Datei nicht gefunden. Das Dokument wurde möglicherweise vor der PDF-Speicher-Funktion hochgeladen.")
    else:
        st.write("Wählen Sie ein Dokument aus, um das PDF anzuzeigen.")

