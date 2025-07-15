RAG for Stratasys Documents

A minimal, object-oriented retrieval-augmented generation (RAG) stack for chatting with your technical PDFs. It pairs a concise LangGraph pipeline with two plug-and-play front-end Streamlit.

⸻

✨ Features

Module	Purpose
DocumentIngestor	Parse PDF → chunk → embed → store in Chroma
RAGPipeline	LangGraph graph (retrieve ⇒ generate) wrapping GPT-4o-mini
Streamlit UI	Lightweight single-page chat app
Chainlit UI	Full-fledged chat interface with streaming & attachments

*️⃣ Easily swap embeddings, vector stores, or LLMs without touching the UIs.

⸻

Quick-Start

# 1) Clone & enter
$ git clone https://github.com/your-org/pdf-rag-langgraph-stack.git
$ cd pdf-rag-langgraph-stack

# 2) Install (isolated env recommended)
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt   # or the one-liner below

# alternative one-liner
$ pip install langgraph langchain langchain-community \
      chromadb pypdf tiktoken python-dotenv streamlit chainlit[all]

# 3) Set your OpenAI key (and optional .env for dotenv)
$ export OPENAI_API_KEY="sk-…"

# 4) Ingest a PDF
$ python rag_oop_stack.py ingest docs/my_manual.pdf

# 5) Launch a front-end
$ streamlit run rag_oop_stack.py   # or
$ chainlit run rag_oop_stack.py

Heads-up: Both front-ends prompt you to upload a PDF if you skipped step 4.

⸻

🗂️ Repository Layout

├── rag_oop_stack.py   # ← core classes + both UIs
├── docs/              # place your PDFs here (ignored by Git)
├── db/                # Chroma persistence
└── requirements.txt   # exact versions (optional lock-file)

Feel free to split rag_oop_stack.py into dedicated modules — everything is import-safe.

⸻

🛠️ Configuration

Variable	Default	Description
OPENAI_API_KEY	—	Your OpenAI key (mandatory)
EMBEDDING_MODEL	text-embedding-3-small	Change via .env to use local HF models
CHUNK_SIZE	1024 tokens	Tune in DocumentIngestor ctor
CHUNK_OVERLAP	64 tokens	—

Add a .env file in repo root – it is automatically loaded via python-dotenv.

⸻

🏗️ Architecture

PDF → DocumentIngestor ──▶ Chroma Vector-Store
                              │
User-Question ─▶ RAGPipeline (Retriever) ─▶ Context
                                 │
                                 ▼
                           GPT-4o-mini
                                 │
                                 ▼
                            Final Answer

Two nodes, zero fluff. Extend the graph with tool-use, multi-document routing, etc.

⸻

📈 Roadmap
	•	Support multiple collections (multi-PDF projects)
	•	Add InstructorEmbeddings + Faiss offline mode
	•	Dockerfile & GitHub Actions CI
	•	LangSmith tracing presets

PRs welcome! 🙌

⸻

🪪 License

This template is released under the MIT License — have fun and build something great!