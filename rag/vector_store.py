from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
 
from config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
 
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
PROFILE_POINT_ID = 0  # must match profile_store.py
 
_client: QdrantClient | None = None
 
 
def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise RuntimeError(
                "QDRANT_URL and QDRANT_API_KEY must be set in environment variables."
            )
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client
 
 
def _ensure_collection(client: QdrantClient) -> None:
    """Create the collection and required payload indexes if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
            ),
        )
    # Qdrant requires a keyword index on any field used in a filter.
    # create_payload_index is idempotent — safe to call on every startup.
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="type",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
 
 
def _chunks_only_filter() -> qmodels.Filter:
    """Filter that matches only chunk points, never the profile document."""
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="type",
                match=qmodels.MatchValue(value="chunk"),
            )
        ]
    )
 
 
def _chunk_point_id(chunk_id: str) -> int:
    """Map a string chunk id to a stable positive integer, never 0."""
    return (abs(hash(chunk_id)) % (2**63 - 1)) + 1
 
 
def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Upsert profile chunk points into Qdrant."""
    client = _get_client()
    _ensure_collection(client)
 
    points = [
        qmodels.PointStruct(
            id=_chunk_point_id(chunk["id"]),
            vector=embedding,
            payload={
                "type": "chunk",
                "text": chunk["text"],
                "section": chunk["section"],
                "chunk_id": chunk["id"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
 
 
def query_chunks_with_scores(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Return the top-k most similar chunks with their raw cosine similarity score.
 
    Returns list of dicts: { text, section, chunk_id, score }
    score is cosine similarity in [0, 1] — higher is more similar.
    """
    client = _get_client()
    _ensure_collection(client)
 
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        query_filter=_chunks_only_filter(),
        limit=top_k,
        with_payload=True,
    )
 
    return [
        {
            "text": hit.payload.get("text", ""),
            "section": hit.payload.get("section", ""),
            "chunk_id": hit.payload.get("chunk_id", str(hit.id)),
            "score": hit.score,  # cosine similarity, higher = better
        }
        for hit in response.points
    ]
 
 
# Keep backward-compatible alias used nowhere now but safe to keep
def query_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    results = query_chunks_with_scores(query_embedding, top_k)
    return [
        {"text": r["text"], "section": r["section"], "distance": 1 - r["score"]}
        for r in results
    ]
 
 
def delete_all_chunks() -> None:
    """Delete all chunk points — the profile document (id=0) is preserved."""
    client = _get_client()
    _ensure_collection(client)
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=_chunks_only_filter()
        ),
    )