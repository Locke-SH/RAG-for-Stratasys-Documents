from __future__ import annotations
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import sqlite3
import re
from config import RAGConfig

class DocumentIngestor:
    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """Sanitize collection name to meet Chroma requirements."""
        # Replace spaces and invalid chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        # Remove leading/trailing non-alphanumeric chars
        sanitized = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', sanitized)
        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = f"doc_{sanitized}"
        # Ensure maximum length
        if len(sanitized) > 512:
            sanitized = sanitized[:512]
        return sanitized

    def __init__(self, 
                 persist_dir: str | Path | None = None,
                 chunk_size: int | None = None,
                 chunk_overlap: int | None = None,
                 model_name: str | None = None,
                 cfg: RAGConfig | None = None
                 ) -> None:
                self.cfg = cfg or RAGConfig()
                self.persist_dir = Path(persist_dir or self.cfg.db_dir)
                self.persist_dir.mkdir(exist_ok=True)
                self.pdfs_dir = Path("data"); self.pdfs_dir.mkdir(exist_ok=True)
                cs = chunk_size if chunk_size is not None else self.cfg.chunk_size
                co = chunk_overlap if chunk_overlap is not None else self.cfg.chunk_overlap
                emb_model = model_name if model_name is not None else self.cfg.embedding_model
                self._splitter = RecursiveCharacterTextSplitter(
                                                        chunk_size=cs,
                                                        chunk_overlap=co,
                                                        separators=["\n\n", "\n", " ", ""])
                self._embedding = HuggingFaceEmbeddings(model_name=emb_model)

    def ingest(self, pdf: str | Path, collection: str | None = None) -> int:
        docs = PyPDFLoader(str(pdf)).load()
        chunks = self._splitter.split_documents(docs)
        Chroma.from_documents(chunks, self._embedding,
                              persist_directory=str(self.persist_dir),
                              collection_name=collection)
        
        # Store the original PDF file
        if collection:
            pdf_copy_path = self.pdfs_dir / f"{collection}.pdf"
            import shutil
            shutil.copy2(pdf, pdf_copy_path)
        
        return len(chunks)

    def list_collections(self) -> list[str]:
        """List all collections in the Chroma database."""
        db_path = self.persist_dir / "chroma.sqlite3"
        if not db_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM collections")
            collections = [row[0] for row in cursor.fetchall()]
            conn.close()
            return collections
        except Exception:
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the Chroma database."""
        try:
            # Create a temporary Chroma instance to delete the collection
            chroma_client = Chroma(
                persist_directory=str(self.persist_dir),
                collection_name=collection_name,
                embedding_function=self._embedding
            )
            chroma_client.delete_collection()
            
            # Also delete the stored PDF file
            pdf_path = self.pdfs_dir / f"{collection_name}.pdf"
            if pdf_path.exists():
                pdf_path.unlink()
            
            return True
        except Exception:
            return False

    def get_pdf_path(self, collection_name: str) -> Path | None:
        """Get the path to the stored PDF for a collection."""
        pdf_path = self.pdfs_dir / f"{collection_name}.pdf"
        return pdf_path if pdf_path.exists() else None
    
    def get_pdf_url(self, collection_name: str) -> str | None:
        """Get the URL to serve the PDF file."""
        pdf_path = self.get_pdf_path(collection_name)
        if pdf_path and pdf_path.exists():
            return f"./data/{collection_name}.pdf"
        return None

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Chroma DB")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--db", default="db", help="Chroma persistence directory")
    parser.add_argument("--collection", default=None, help="Chroma collection name")
    args = parser.parse_args()

    ingestor = DocumentIngestor(args.db)
    n = ingestor.ingest(args.pdf, args.collection)
    print(f"Indexed {n} chunks from {args.pdf}")
"""

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    collections = ingestor.list_collections()
    print("Verfügbare Collections:")
    for c in collections:
        print(f"- {c}")

