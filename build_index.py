"""
One-shot script to build the FAISS index from PubMed.

Usage:
    python build_index.py --queries "diabetes mellitus" "hypertension diagnosis"
    python build_index.py --queries-file queries.txt
"""
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

from ingestion.pubmed_fetcher import fetch_documents_for_query
from ingestion.document_processor import split_documents
from indexing.faiss_store import build_index, load_index, add_documents
import config

os.makedirs("data", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--queries", nargs="+", help="One or more PubMed search queries")
    group.add_argument("--queries-file", help="File with one query per line")
    parser.add_argument("--max-results", type=int, default=config.PUBMED_MAX_RESULTS)
    parser.add_argument("--incremental", action="store_true", help="Add to existing index")
    args = parser.parse_args()

    if args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = args.queries

    all_chunks = []
    for query in queries:
        logging.info("Fetching PubMed docs for: %s", query)
        docs = fetch_documents_for_query(query, max_results=args.max_results)
        chunks = split_documents(docs)
        logging.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)

    logging.info("Total chunks to index: %d", len(all_chunks))

    if args.incremental:
        store = load_index()
        add_documents(store, all_chunks)
        logging.info("Incremental index updated.")
    else:
        build_index(all_chunks)
        logging.info("New index built and saved.")


if __name__ == "__main__":
    main()
