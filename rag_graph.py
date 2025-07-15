from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# Environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY (or OPENAI_API_KEY) is missing")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium")
DB_DIR = os.getenv("DB_DIR", "db")
K = int(os.getenv("RETRIEVAL_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))


# LangGraph state
@dataclass
class QAState:
    question: str
    context: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    answer: Optional[str] = None


# RAGPipeline class
class RAGPipeline:
    """Retrieval-augmented generation wrapped in a LangGraph state machine."""

    PROMPT = PromptTemplate.from_template(
        """Du bist ein hilfsreicher Experte für technische Dokumente. Beantworte die Frage ausführlich und detailliert basierend auf dem gegebenen Kontext.

Kontext:
{context}

Quellen:
{sources}

Frage: {question}

Antwort: Gib eine vollständige und detaillierte Antwort auf Deutsch. Erkläre alle relevanten Aspekte und verwende konkrete Informationen aus dem Kontext. Gib am Ende der Antwort die Quellen an (Seiten), wo die Informationen gefunden wurden."""
    )

    def __init__(
        self,
        db_dir: str = DB_DIR,
        collection_name: str | None = None,
        k: int = K,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
    ) -> None:
        collection_name = collection_name or "default" 
        # Retriever
        self._retriever = Chroma(
            persist_directory=db_dir,
            collection_name=collection_name,
            embedding_function=HuggingFaceEmbeddings(                # <—
                model_name="sentence-transformers/all-MiniLM-L6-v2"),
        ).as_retriever(k=k)

        # OpenRouter LLM
        self._llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=temperature,
            request_timeout=90,
        )
        self._graph = self._build_graph()


    def _build_graph(self):
        graph = StateGraph(QAState)

        def retrieve_node(state: QAState) -> QAState:
            docs = self._retriever.invoke(state.question) 
            state.context = [d.page_content for d in docs]
            # Extract page information from metadata
            sources = []
            for doc in docs:
                page = doc.metadata.get('page', 'Unbekannt')
                source = doc.metadata.get('source', 'Unbekannt')
                if isinstance(page, int):
                    page_num = page + 1  # PDF pages are 0-indexed
                    sources.append(f"Seite {page_num}")  # Simple text format
                else:
                    sources.append(f"Quelle: {source}")
            state.sources = sources
            return state

        def generate_node(state: QAState) -> QAState:
            response = self._llm.invoke(self.PROMPT.format(**state.__dict__))
            state.answer = response.content
            return state

        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.set_finish_point("generate")
        return graph.compile()


    def answer(self, question: str) -> str:
        result = self._graph.invoke({"question": question})
        return result["answer"]
    
    ask = answer 

