"""Central configuration for the Medical Chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")   # llama3-8b-8192 for speed
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ── PubMed ────────────────────────────────────────────────────────────────────
NCBI_EMAIL: str = os.environ["NCBI_EMAIL"]
NCBI_API_KEY: str | None = os.getenv("NCBI_API_KEY")
PUBMED_MAX_RESULTS: int = int(os.getenv("PUBMED_MAX_RESULTS", "20"))
PUBMED_FETCH_BATCH: int = int(os.getenv("PUBMED_FETCH_BATCH", "5"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
)

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K: int = int(os.getenv("TOP_K", "6"))

# ── Prompt Cache ──────────────────────────────────────────────────────────────
CACHE_DIR: str = os.getenv("CACHE_DIR", "data/prompt_cache")
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60 * 24)))  # 24 h

# ── Conversation History ──────────────────────────────────────────────────────
CONVERSATION_DB: str = os.getenv("CONVERSATION_DB", "data/conversations.db")
SUMMARY_AFTER_TURNS: int = int(os.getenv("SUMMARY_AFTER_TURNS", "6"))
MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
