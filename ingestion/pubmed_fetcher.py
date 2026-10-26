"""PubMed document fetcher using Biopython Entrez."""
import time
import logging
from typing import Generator
from Bio import Entrez
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)

Entrez.email = config.NCBI_EMAIL
if config.NCBI_API_KEY:
    Entrez.api_key = config.NCBI_API_KEY


def search_pubmed(query: str, max_results: int = config.PUBMED_MAX_RESULTS) -> list[str]:
    """Return a list of PMIDs for the given query."""
    with Entrez.esearch(db="pubmed", term=query, retmax=max_results) as handle:
        record = Entrez.read(handle)
    return record["IdList"]


def fetch_abstracts(pmids: list[str], batch_size: int = config.PUBMED_FETCH_BATCH) -> Generator[Document, None, None]:
    """Fetch abstracts in batches and yield LangChain Documents."""
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        try:
            with Entrez.efetch(
                db="pubmed", id=",".join(batch), rettype="abstract", retmode="xml"
            ) as handle:
                records = Entrez.read(handle)

            for article in records.get("PubmedArticle", []):
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                art = medline["Article"]

                title = str(art.get("ArticleTitle", ""))
                abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)

                if not abstract.strip():
                    continue

                # MeSH terms as metadata
                mesh_list = medline.get("MeshHeadingList", [])
                mesh_terms = [
                    str(m["DescriptorName"]) for m in mesh_list
                ] if mesh_list else []

                year = ""
                pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")))

                yield Document(
                    page_content=f"Title: {title}\n\nAbstract: {abstract}",
                    metadata={
                        "pmid": pmid,
                        "title": title,
                        "year": year,
                        "mesh_terms": mesh_terms,
                        "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    },
                )
        except Exception as exc:
            logger.warning("Failed to fetch batch %s: %s", batch, exc)

        time.sleep(0.4)  # NCBI rate limit: ~3 req/s without API key, 10/s with


def fetch_documents_for_query(query: str, max_results: int = config.PUBMED_MAX_RESULTS) -> list[Document]:
    """High-level helper: search + fetch all documents for a query string."""
    pmids = search_pubmed(query, max_results)
    logger.info("Found %d PMIDs for query: %s", len(pmids), query)
    docs = list(fetch_abstracts(pmids))
    logger.info("Fetched %d documents", len(docs))
    return docs
