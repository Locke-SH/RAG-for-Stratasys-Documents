# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

class DocumentIngestor:
    def __init__(self, persist_dir: str | Path = "db",
                 chunk_size: int | None = None,
                 chunk_overlap: int | None = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_dir = Path(persist_dir); self.persist_dir.mkdir(exist_ok=True)
        cs = chunk_size or int(os.getenv("CHUNK_SIZE", "1024"))
        co = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "64"))
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=cs,
                                                        chunk_overlap=co,
                                                        separators=["\n\n", "\n", " ", ""])
        self._embedding = HuggingFaceEmbeddings(model_name=model_name)

    def ingest(self, pdf: str | Path, collection: str | None = None) -> int:
        docs = PyPDFLoader(str(pdf)).load()
        chunks = self._splitter.split_documents(docs)
        Chroma.from_documents(chunks, self._embedding,
                              persist_directory=str(self.persist_dir),
                              collection_name=collection)
        return len(chunks)


# ---------------------------------------------------------------------------#
# CLI helper
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Chroma DB")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--db", default="db", help="Chroma persistence directory")
    parser.add_argument("--collection", default=None, help="Chroma collection name")
    args = parser.parse_args()

    ingestor = DocumentIngestor(args.db)
    n = ingestor.ingest(args.pdf, args.collection)
    print(f"✅ Indexed {n} chunks from {args.pdf}")
