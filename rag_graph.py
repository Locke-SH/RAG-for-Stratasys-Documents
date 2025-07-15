from dataclasses import dataclass, field
from typing import List, Optional

from langgraph.graph import StateGraph
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate


# ----------  State ----------
@dataclass
class QAState:
    question: str
    context: List[str] = field(default_factory=list)
    answer: Optional[str] = None


# ----------  Ressourcen ----------
DB_DIR = "db"
retriever = Chroma(persist_directory=DB_DIR,
                   embedding_function=OpenAIEmbeddings()).as_retriever(k=4)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

PROMPT = PromptTemplate.from_template(
    """Du bist ein Experte für technische Dokumente.
    Nutze ausschließlich den Kontext, um die Frage zu beantworten.
    Kontext:
    {context}

    Frage: {question}
    Antwort (Deutsch):""")

# ----------  Nodes ----------
def retrieve_node(state: QAState) -> QAState:
    docs = retriever.get_relevant_documents(state.question)
    state.context = [d.page_content for d in docs]
    return state

def generate_node(state: QAState) -> QAState:
    state.answer = llm.predict(PROMPT.format(**state.__dict__))
    return state


# ----------  Graph ----------
graph = StateGraph(QAState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

rag_pipeline = graph.compile()      # <- importierbar im Front-End

def answer(question: str) -> str:
    return rag_pipeline.invoke(QAState(question)).answer