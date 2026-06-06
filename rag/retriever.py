import math
import re
from collections import defaultdict
 
from config import TOP_K_CHUNKS
from rag.embedder import embed_texts
from rag.vector_store import query_chunks_with_scores
 
# Minimum cosine similarity to include a chunk (0–1 scale).
# Chunks scoring below this are too dissimilar and pollute the prompt.
MIN_SCORE_THRESHOLD = 0.25
 
# How many chunks to fetch per sub-query before re-ranking.
# Fetching more than TOP_K and then re-ranking gives better final results.
FETCH_PER_QUERY = 15
 
 
def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful tokens for keyword overlap scoring."""
    # Keep alphanumeric tokens of length >= 3, lowercase
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#.\-]{2,}", text.lower())
    # Drop very common English stop words
    stopwords = {
        "the", "and", "for", "with", "that", "this", "are", "has", "have",
        "will", "from", "using", "use", "you", "your", "our", "their",
        "who", "what", "how", "about", "into", "over", "such", "any",
        "all", "also", "its", "not", "but", "can", "should", "would",
        "well", "been", "more", "than", "was", "were", "experience",
        "skills", "role", "job", "work", "team", "ability", "strong",
    }
    return {t for t in tokens if t not in stopwords}
 
 
def _keyword_score(query_keywords: set[str], chunk_text: str) -> float:
    """BM25-inspired keyword overlap score (0–1)."""
    if not query_keywords:
        return 0.0
    chunk_tokens = _extract_keywords(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = query_keywords & chunk_tokens
    # Normalise by query length (recall-style) then apply log dampening
    raw = len(overlap) / len(query_keywords)
    return math.log1p(raw * 10) / math.log1p(10)  # maps 0–1 → 0–1
 
 
def _build_sub_queries(role: str, jd: str, extra: str = "") -> list[str]:
    """
    Decompose the job context into focused sub-queries.
 
    Rather than embedding the whole JD (which hits the 256-token limit of
    all-MiniLM-L6-v2 and produces a diffuse vector), we create short,
    targeted queries that each pull a specific type of chunk.
    """
    role = role.strip()
    jd_short = jd.strip()[:800]  # cap to avoid token overflow
 
    queries = []
 
    # 1. Role-skills query — pulls skill chunks aligned to the role
    if role:
        queries.append(f"{role} technical skills technologies tools")
 
    # 2. Experience query — pulls work experience chunks
    if role:
        queries.append(f"{role} work experience responsibilities achievements")
 
    # 3. Projects query — pulls project chunks matching the domain
    if role:
        queries.append(f"{role} projects built developed implemented")
 
    # 4. JD-anchored query — semantic match against the actual JD text
    if jd_short:
        # Use only the first ~400 chars (covers requirements section)
        queries.append(jd_short[:400])
 
    # 5. Extra context (e.g. form questions, outreach context)
    if extra.strip():
        queries.append(extra.strip()[:300])
 
    # Always include a personal-info anchor so identity chunks rank highly
    queries.append("personal information contact summary years of experience")
 
    return queries
 
 
def _hybrid_retrieve(
    role: str,
    jd: str,
    extra: str = "",
    top_k: int | None = None,
) -> list[dict]:
    """
    Run multi-query hybrid retrieval and return the top-k merged chunks.
 
    Returns list of dicts: { text, section, score }
    """
    k = top_k or TOP_K_CHUNKS
    sub_queries = _build_sub_queries(role, jd, extra)
 
    # Embed all sub-queries in one batch (efficient)
    embeddings = embed_texts(sub_queries)
 
    # Collect raw results: chunk_id → best scores across all sub-queries
    best_semantic: dict[str, float] = defaultdict(float)
    best_keyword: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}
 
    all_query_keywords = _extract_keywords(role + " " + jd + " " + extra)
 
    for emb, query_text in zip(embeddings, sub_queries):
        raw_chunks = query_chunks_with_scores(emb, FETCH_PER_QUERY)
        q_keywords = _extract_keywords(query_text)
 
        for chunk in raw_chunks:
            cid = chunk["chunk_id"]
            sem_score = chunk["score"]  # cosine similarity 0–1
 
            # Skip immediately if semantically irrelevant
            if sem_score < MIN_SCORE_THRESHOLD:
                continue
 
            kw_score = _keyword_score(
                q_keywords | all_query_keywords,
                chunk["text"],
            )
 
            # Keep the best scores seen across all sub-queries for this chunk
            if sem_score > best_semantic[cid]:
                best_semantic[cid] = sem_score
            if kw_score > best_keyword[cid]:
                best_keyword[cid] = kw_score
 
            chunk_data[cid] = chunk
 
    if not chunk_data:
        return []
 
    # Hybrid score: 70% semantic + 30% keyword
    # Semantic is the primary signal; keyword breaks ties and rescues
    # rare-term chunks (e.g. "Qdrant", "Redshift") that semantic alone misses.
    scored = []
    for cid, chunk in chunk_data.items():
        hybrid = 0.70 * best_semantic[cid] + 0.30 * best_keyword[cid]
        scored.append((hybrid, chunk))
 
    scored.sort(key=lambda x: x[0], reverse=True)
 
    return [
        {**c, "score": score}
        for score, c in scored[:k]
    ]
 
 
def retrieve_context(
    query: str,
    *,
    role: str = "",
    jd: str = "",
    top_k: int | None = None,
) -> str:
    """
    Public API called by agents.
 
    Parameters
    ----------
    query   : legacy single-string query (used when role/jd not available)
    role    : job title / role name  (preferred)
    jd      : full job description text  (preferred)
    top_k   : override TOP_K_CHUNKS
    """
    # If agents pass role+jd use the full hybrid pipeline;
    # fall back to treating query as the role for backwards compatibility.
    effective_role = role or query
    effective_jd = jd
 
    chunks = _hybrid_retrieve(
        role=effective_role,
        jd=effective_jd,
        extra=query if (role or jd) else "",
        top_k=top_k,
    )
 
    if not chunks:
        return "No profile data indexed. Please save your profile first."
 
    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk["section"].replace("_", " ").title()
        score_pct = f"{chunk['score']:.0%}"
        parts.append(f"[{i}] ({section} | relevance {score_pct})\n{chunk['text']}")
 
    return "\n\n".join(parts)