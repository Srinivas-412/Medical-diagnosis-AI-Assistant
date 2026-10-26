"""
SQLite-backed conversation history with rolling summarization.

Design:
  - Every turn (human + AI message) is saved to the DB.
  - After SUMMARY_AFTER_TURNS new turns, the oldest un-summarized turns
    are compressed into a single summary message so the context window
    stays bounded.
  - The summary itself is stored as a special 'summary' role message and
    is prepended to the active window on the next call.
"""
import asyncio
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Text, DateTime, select
)
from sqlalchemy.orm import DeclarativeBase, Session
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import config

logger = logging.getLogger(__name__)

engine = create_engine(f"sqlite:///{config.CONVERSATION_DB}", echo=False)


class Base(DeclarativeBase):
    pass


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)   # human | ai | summary
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_summarized = Column(Integer, default=0)  # 0=active, 1=replaced by summary


Base.metadata.create_all(engine)


# ── Low-level DB helpers ──────────────────────────────────────────────────────

def save_turn(session_id: str, role: str, content: str) -> None:
    with Session(engine) as db:
        db.add(ConversationTurn(session_id=session_id, role=role, content=content))
        db.commit()


def get_active_turns(session_id: str) -> list[ConversationTurn]:
    with Session(engine) as db:
        rows = db.scalars(
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .where(ConversationTurn.is_summarized == 0)
            .order_by(ConversationTurn.id)
        ).all()
        return list(rows)


def mark_summarized(turn_ids: list[int]) -> None:
    with Session(engine) as db:
        db.query(ConversationTurn).filter(
            ConversationTurn.id.in_(turn_ids)
        ).update({"is_summarized": 1}, synchronize_session=False)
        db.commit()


# ── Summarizer ────────────────────────────────────────────────────────────────

_summarizer_llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
    max_retries=2,
)

SUMMARIZE_PROMPT = (
    "You are a medical assistant. Summarize the following conversation turns "
    "concisely, preserving all medically relevant details (diseases, symptoms, "
    "tests, medications, patient context). The summary will replace these turns "
    "in future context windows.\n\nConversation:\n{turns}\n\nSummary:"
)


def _build_summary(turns: list[ConversationTurn]) -> str:
    turns_text = "\n".join(
        f"[{t.role.upper()}]: {t.content}" for t in turns
    )
    prompt = SUMMARIZE_PROMPT.format(turns=turns_text)
    response = _summarizer_llm.invoke([HumanMessage(content=prompt)])
    return response.content


def maybe_summarize(session_id: str) -> None:
    """Compress old turns into a summary when the buffer exceeds the threshold."""
    turns = get_active_turns(session_id)
    # Only count human+ai turns, not existing summaries
    regular = [t for t in turns if t.role in ("human", "ai")]
    if len(regular) < config.SUMMARY_AFTER_TURNS:
        return

    # Summarize all but the last 2 turns (keep recent context intact)
    to_summarize = regular[:-2]
    if not to_summarize:
        return

    logger.info("Summarizing %d turns for session %s", len(to_summarize), session_id)
    summary_text = _build_summary(to_summarize)

    # Persist summary + mark old turns as archived
    save_turn(session_id, "summary", summary_text)
    mark_summarized([t.id for t in to_summarize])


# ── Public interface ──────────────────────────────────────────────────────────

def get_chat_history(session_id: str) -> list[BaseMessage]:
    """
    Return LangChain message objects for the active window.
    Summaries are injected as SystemMessages at the top.
    """
    turns = get_active_turns(session_id)
    messages: list[BaseMessage] = []
    for t in turns:
        if t.role == "summary":
            messages.append(SystemMessage(content=f"[Prior conversation summary]\n{t.content}"))
        elif t.role == "human":
            messages.append(HumanMessage(content=t.content))
        elif t.role == "ai":
            messages.append(AIMessage(content=t.content))
    return messages


def add_user_message(session_id: str, content: str) -> None:
    save_turn(session_id, "human", content)
    maybe_summarize(session_id)


def add_ai_message(session_id: str, content: str) -> None:
    save_turn(session_id, "ai", content)
