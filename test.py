#import chromadb
#from langchain.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings

"""
collection = "AVV-Labom_11_03_25"           # dein Dateiname ohne .pdf
db = Chroma(
        persist_directory="db",
        collection_name=collection,
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"))

docs = db.similarity_search("Worum geht es in diesem Dokument?", k=1)
print("Treffer:", len(docs))
print(docs[0].page_content[:200] if docs else "KEINE")
"""

#client = chromadb.PersistentClient(path="db")      # dein Persist-Ordner
#collection = client.get_collection("AVV-Labom_11_03_25")   # Name aus app.py

#print("⮕ Anzahl Chunks:", collection.count())
#print("⮕ Beispiel-Metadaten:")
#print(collection.peek())           # zeigt die ersten 10 Einträge (IDs, Embeds, Metadaten)

#from rag_graph import RAGPipeline
#pipe = RAGPipeline(collection_name="AVV-Labom_11_03_25")
#print(pipe.answer("Worum geht es in diesem Vertrag?")[:400])

#from rag_graph import RAGPipeline
#pipe = RAGPipeline(collection_name="AVV-Labom_11_03_25")
#print(pipe._retriever._collection.name)   # sollte exakt AVV-Labom_11_03_25 sein
#print(pipe.answer("Worum geht es in diesem Vertrag?")[:400])

from rag_graph import RAGPipeline

pipe = RAGPipeline(collection_name="AVV-Labom_11_03_25")

# Sammlung prüfen
print("Collection-Name:", pipe._retriever.vectorstore._collection.name)

# Erster Chunk zur Kontrolle
doc = pipe._retriever.vectorstore.similarity_search("dummy", k=1)[0]
print("Erstes Dokument-Snippet:", doc.page_content[:120], "…")

# LLM-Antwort
print("Antwort:", pipe.answer("Worum geht es in diesem Vertrag?")[:400])