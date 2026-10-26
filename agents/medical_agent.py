"""
LangGraph-based Medical Diagnosis Agent with:
  - Groq LLM (llama3-70b)
  - RAG via FAISS retriever
  - Prompt caching (disk)
  - Conversation history with rolling summarization
  - Retry logic with answer validation (up to MAX_RETRIES attempts)

Graph nodes:
  cache_check  ──► retrieve  ──► generate  ──► validate
                                    ▲               │ (fail)
                                    └───────────────┘ (retry)
                                                    │ (pass)
                                                  END
"""
import logging
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from indexing.faiss_store import get_retriever
from cache.prompt_cache import get_cached, set_cached
from memory.conversation_history import get_chat_history, add_user_message, add_ai_message
import config

logger = logging.getLogger(__name__)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=config.GROQ_TEMPERATURE,
    max_retries=2,  # Groq-level HTTP retries
)

retriever = get_retriever()

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert medical AI assistant powered by peer-reviewed PubMed literature.

Your responsibilities:
1. Extract diseases, symptoms, diagnostic tests and treatments from the context.
2. Ground every claim in the provided PubMed context.
3. Always cite the PMID source at the end of relevant sentences.
4. If you are uncertain, state the uncertainty clearly.
5. NEVER fabricate medical information.

IMPORTANT DISCLAIMER: This assistant is for informational purposes only and does not replace professional medical advice.

Relevant PubMed context:
{context}
"""

VALIDATION_PROMPT = """You are a medical answer quality evaluator.

Given the user question and the assistant's answer, assess whether the answer:
1. Directly addresses the medical question.
2. Is grounded in the provided context (not hallucinated).
3. Is medically coherent and safe.

Respond with ONLY one of:
- PASS  (answer is acceptable)
- RETRY (answer needs improvement, briefly explain why on the same line)

Question: {question}
Answer: {answer}
Context used: {context}

Verdict:"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)

# ── Graph State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    question: str
    context: str
    history: list[BaseMessage]
    answer: str
    validation_verdict: str
    retry_count: int
    final_answer: str
    cache_hit: bool
    sources: list[dict]

# ── Node: cache_check ─────────────────────────────────────────────────────────
def node_cache_check(state: AgentState) -> AgentState:
    cached = get_cached(state["question"], state["session_id"])
    if cached:
        logger.info("Cache HIT for session %s", state["session_id"])
        return {**state, "final_answer": cached, "cache_hit": True}
    return {**state, "cache_hit": False}


def should_skip_after_cache(state: AgentState) -> str:
    return "end" if state.get("cache_hit") else "retrieve"


# ── Node: retrieve ────────────────────────────────────────────────────────────
def node_retrieve(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["question"])
    context_parts = []
    sources = []
    for doc in docs:
        pmid = doc.metadata.get("pmid", "N/A")
        title = doc.metadata.get("title", "")
        context_parts.append(f"[PMID:{pmid}] {doc.page_content}")
        sources.append({"pmid": pmid, "title": title, "url": doc.metadata.get("source", "")})
    return {**state, "context": "\n\n".join(context_parts), "sources": sources}


# ── Node: load_history ────────────────────────────────────────────────────────
def node_load_history(state: AgentState) -> AgentState:
    history = get_chat_history(state["session_id"])
    return {**state, "history": history}


# ── Node: generate ────────────────────────────────────────────────────────────
def node_generate(state: AgentState) -> AgentState:
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "history": state.get("history", []),
        "question": state["question"],
    })
    return {**state, "answer": answer}


# ── Node: validate ────────────────────────────────────────────────────────────
def node_validate(state: AgentState) -> AgentState:
    val_chain = validation_prompt | llm | StrOutputParser()
    verdict = val_chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
        "context": state["context"],
    })
    logger.info("Validation verdict (attempt %d): %s", state["retry_count"] + 1, verdict[:60])
    return {**state, "validation_verdict": verdict.strip()}


def should_retry(state: AgentState) -> str:
    verdict = state.get("validation_verdict", "")
    retry_count = state.get("retry_count", 0)
    if verdict.startswith("PASS") or retry_count >= config.MAX_RETRIES:
        return "finalize"
    return "generate"   # loop back with incremented retry


# ── Node: finalize ────────────────────────────────────────────────────────────
def node_finalize(state: AgentState) -> AgentState:
    answer = state["answer"]
    # Append source citations block
    if state.get("sources"):
        citations = "\n".join(
            f"- [{s['title'] or s['pmid']}]({s['url']})" for s in state["sources"]
        )
        answer = f"{answer}\n\n**Sources:**\n{citations}"

    # Persist to history
    add_user_message(state["session_id"], state["question"])
    add_ai_message(state["session_id"], answer)

    # Cache successful answer
    set_cached(state["question"], answer, state["session_id"])

    return {**state, "final_answer": answer, "retry_count": state.get("retry_count", 0) + 1}


# ── Node: increment_retry ─────────────────────────────────────────────────────
def node_increment_retry(state: AgentState) -> AgentState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("cache_check", node_cache_check)
    g.add_node("load_history", node_load_history)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("validate", node_validate)
    g.add_node("increment_retry", node_increment_retry)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("cache_check")

    g.add_conditional_edges("cache_check", should_skip_after_cache, {
        "end": END,
        "retrieve": "load_history",
    })
    g.add_edge("load_history", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", should_retry, {
        "finalize": "finalize",
        "generate": "increment_retry",
    })
    g.add_edge("increment_retry", "generate")
    g.add_edge("finalize", END)

    return g.compile()


# ── Compiled graph singleton ──────────────────────────────────────────────────
graph = build_graph()


def run_agent(session_id: str, question: str) -> dict:
    """
    Entry point. Returns dict with:
      - final_answer: str
      - sources: list[dict]
      - cache_hit: bool
      - retry_count: int
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "context": "",
        "history": [],
        "answer": "",
        "validation_verdict": "",
        "retry_count": 0,
        "final_answer": "",
        "cache_hit": False,
        "sources": [],
    }
    result = graph.invoke(initial_state)
    return {
        "final_answer": result["final_answer"],
        "sources": result.get("sources", []),
        "cache_hit": result.get("cache_hit", False),
        "retry_count": result.get("retry_count", 0),
    }
