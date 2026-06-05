from config import TOP_K_CHUNKS
from rag.embedder import embed_text
from rag.vector_store import query_chunks


def retrieve_context(query: str, top_k: int | None = None) -> str:
    """Retrieve top-k relevant profile chunks and format as context string."""
    k = top_k or TOP_K_CHUNKS
    embedding = embed_text(query)
    chunks = query_chunks(embedding, k)

    if not chunks:
        return "No profile data indexed. Please save your profile first."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk["section"].replace("_", " ").title()
        parts.append(f"[{i}] ({section})\n{chunk['text']}")

    return "\n\n".join(parts)
