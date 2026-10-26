"""FastAPI REST API for the Medical Diagnosis Chatbot."""
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agents.medical_agent import run_agent

logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Medical Diagnosis AI Chatbot",
    description="LangGraph + Groq + RAG + PubMed powered medical assistant",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=5, max_length=2000)


class SourceCitation(BaseModel):
    pmid: str
    title: str
    url: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceCitation]
    cache_hit: bool
    retry_count: int


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_agent(session_id=request.session_id, question=request.question)
    except Exception as exc:
        logging.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        session_id=request.session_id,
        answer=result["final_answer"],
        sources=[SourceCitation(**s) for s in result.get("sources", [])],
        cache_hit=result.get("cache_hit", False),
        retry_count=result.get("retry_count", 0),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
