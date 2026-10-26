"""FAISS vector store: build, persist, and load."""
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)

_embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def build_index(docs: list[Document]) -> FAISS:
    """Create a FAISS index from a list of Documents."""
    logger.info("Building FAISS index from %d chunks...", len(docs))
    store = FAISS.from_documents(docs, _embeddings)
    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
    store.save_local(config.FAISS_INDEX_PATH)
    logger.info("FAISS index saved to %s", config.FAISS_INDEX_PATH)
    return store


def load_index() -> FAISS:
    """Load persisted FAISS index from disk."""
    if not os.path.exists(config.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {config.FAISS_INDEX_PATH}. Run build_index first."
        )
    return FAISS.load_local(
        config.FAISS_INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True,
    )


def add_documents(store: FAISS, docs: list[Document]) -> FAISS:
    """Incrementally add new documents to an existing index."""
    store.add_documents(docs)
    store.save_local(config.FAISS_INDEX_PATH)
    return store


def get_retriever(store: FAISS | None = None):
    """Return a MMR retriever (diversity-aware) for the FAISS index."""
    if store is None:
        store = load_index()
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 3, "lambda_mult": 0.6},
    )
