from pathlib import Path

import chromadb

from config import VECTOR_STORE_PATH

COLLECTION_NAME = "profile_chunks"
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        path = Path(VECTOR_STORE_PATH)
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(path))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection = _get_collection()
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"section": c["section"]} for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "section": meta.get("section", ""),
                "distance": dist,
            }
        )
    return chunks
