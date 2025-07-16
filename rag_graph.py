from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional



from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from config import RAGConfig

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
        Kontext:{context}
        Quellen:{sources}
        Frage: {question}
        Antwort: Gib eine vollständige und detaillierte Antwort auf Deutsch. Erkläre alle relevanten Aspekte und verwende konkrete Informationen aus dem Kontext. Gib am Ende der Antwort die Quellen an (Seiten), wo die Informationen gefunden wurden."""
    )

    def __init__(
        self, 
        collection_name: str | None = None,
        cfg: RAGConfig | None = None
    ) -> None:
        self.cfg = cfg or RAGConfig()
        self.collection_name = collection_name or "default" 
        # ---- Retriever ----
        self._retriever = Chroma(
            persist_directory=self.cfg.db_dir,
            collection_name=self.collection_name,
            embedding_function=HuggingFaceEmbeddings(
            model_name=self.cfg.embedding_model),
        ).as_retriever(k=self.cfg.retrieval_k)

        # ---- OpenRouter LLM ----
        self._llm = ChatOpenAI(
            model=self.cfg.openrouter_model,
            api_key=self.cfg.openrouter_api_key,
            base_url=self.cfg.openrouter_base_url,
            temperature=self.cfg.temperature,
            timeout=self.cfg.request_timeout
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

