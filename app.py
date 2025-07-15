import streamlit as st
from ingest_docs import ingest
from rag_graph import answer

st.title("📄 Chat mit deinem PDF")
uploaded = st.file_uploader("PDF hochladen", type="pdf")

if uploaded and "db_built" not in st.session_state:
    with open("tmp.pdf", "wb") as f: f.write(uploaded.getbuffer())
    ingest("tmp.pdf"); st.session_state.db_built = True
    st.success("✅ Dokument indexiert!")

prompt = st.chat_input("Frage stellen …")
if prompt:
    with st.spinner("Denke nach …"):
        st.chat_message("assistant").write(answer(prompt))