RAG for Stratasys Documents

A minimal, object-oriented retrieval-augmented generation (RAG) stack for chatting with your technical PDFs. It pairs a concise LangGraph pipeline with plug-and-play front-end Streamlit.

⸻

Features

Module	Purpose
DocumentIngestor	Parse PDF → chunk → embed → store in Chroma
RAGPipeline	LangGraph graph (retrieve ⇒ generate) wrapping
Streamlit UI	Lightweight single-page chat app

Easily swap embeddings, vector stores, or LLMs without touching the UIs.

⸻

Quick-Start

# 1) Clone & enter
$ git clone https://github.com/your-org/pdf-rag-langgraph-stack.git
$ cd pdf-rag-langgraph-stack

# 2) Install (isolated env recommended)
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt   # or the one-liner below


# 3) Create .env from env_example and set your OPENROUTER key
$ OPENROUTER="sk-…"

# 5) Launch a front-end
$ streamlit run app.py

