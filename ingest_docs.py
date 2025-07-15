from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings            # oder HF
from langchain.vectorstores import Chroma


def ingest(pdf_path: str, persist_dir: str = "db"):
    """Parst ein PDF, splittet in Chunks, legt Embeddings in Chroma ab."""
    docs = PyPDFLoader(pdf_path).load()                       # ➊
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=64, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)                   # ➋
    vectordb = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=persist_dir)  # ➌
    vectordb.persist()
    print(f"🗂️  {len(chunks)} Chunks indiziert")


if __name__ == "__main__":
    import sys; ingest(sys.argv[1])