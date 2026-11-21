"""
rag_service.py — Retrieval-augmented grounding for the chatbot's business
advice. Embeds a curated retail-domain knowledge base (backend/knowledge_base/)
into a local ChromaDB collection and retrieves the most relevant documents
for a given question.

This is deliberately separate from product_analytics.py: numeric questions
("which products sell best") are answered with deterministic facts computed
straight from the user's data, not retrieval — mixing the two increases
hallucination risk instead of reducing it. This module only grounds
qualitative/advice-style questions in domain knowledge.
"""
import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

from services.llm_retry import call_with_retry

load_dotenv()

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_DIR = Path(__file__).parent.parent / ".chroma"
COLLECTION_NAME = "retail_knowledge"
EMBEDDING_MODEL = "models/gemini-embedding-001"

_client = None
_chroma_client = None
_collection = None


def _get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _embed(text: str, task_type: str) -> list[float]:
    """task_type is 'RETRIEVAL_DOCUMENT' at ingest time, 'RETRIEVAL_QUERY' at query time."""
    result = call_with_retry(
        _get_genai_client().models.embed_content,
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": task_type},
    )
    return result.embeddings[0].values


def _get_collection():
    """Get (or lazily build) the knowledge-base collection.

    Only documents that are new or whose content changed (by SHA-256) are
    re-embedded, so edits to knowledge_base/*.md take effect on the next
    start without paying to re-embed everything. (Previously this skipped
    whenever the document COUNT matched, so an edited file was never
    re-indexed.) Removed documents are deleted from the collection.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)

    doc_paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    existing = collection.get(include=["metadatas"])
    stored = {i: (m or {}).get("sha256") for i, m in zip(existing["ids"], existing["metadatas"])}

    ids, embeddings, documents, metadatas = [], [], [], []
    for path in doc_paths:
        text = path.read_text()
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if stored.get(path.stem) == sha:
            continue
        ids.append(path.stem)
        embeddings.append(_embed(text, "RETRIEVAL_DOCUMENT"))
        documents.append(text)
        metadatas.append({"source": path.name, "sha256": sha})

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    stale = [i for i in stored if i not in {p.stem for p in doc_paths}]
    if stale:
        collection.delete(ids=stale)

    _collection = collection
    return _collection


def retrieve_relevant_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """
    Return the top-k most relevant knowledge-base documents for a question.

    Returns:
        [{"source": "seasonal_holiday_demand_planning.md", "text": "...", "distance": 0.31}, ...]
    """
    try:
        collection = _get_collection()
        query_embedding = _embed(query, "RETRIEVAL_QUERY")
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"source": meta["source"], "text": doc, "distance": dist})
        return chunks
    except Exception as e:
        # Retrieval is an enhancement, not a hard dependency — if it fails
        # (e.g. embedding API hiccup), the chatbot should still work without it.
        print(f"⚠️ RAG retrieval failed: {e}")
        return []
