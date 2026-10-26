"""Text splitting and metadata enrichment for PubMed documents."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""
    chunks = _splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
