"""
embedder.py  –  Chunking, embedding, and indexing logic.

All local-file dependency (index marker, VECTOR_STORE_PATH, PROFILE_PATH)
has been removed.  The only persistent state lives in Qdrant Cloud.
"""

from sentence_transformers import SentenceTransformer

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_compensation_value(key: str, value: str) -> str:
    """Format raw CTC numbers as LPA for clearer agent answers."""
    if key not in ("current_ctc", "expected_ctc") or not value:
        return value
    digits = "".join(c for c in str(value) if c.isdigit())
    if not digits:
        return value
    amount = int(digits)
    lakhs = amount / 100_000
    if lakhs >= 1:
        return f"{lakhs:g} LPA (₹{amount:,} per annum)"
    return f"₹{amount:,} per annum"


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _embed_model.encode(texts, show_progress_bar=False).tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_profile(profile: dict) -> list[dict]:
    """Split a profile dict into semantic chunks for vector storage."""
    chunks: list[dict] = []

    def add_chunk(chunk_id: str, section: str, text: str) -> None:
        text = text.strip()
        if text:
            chunks.append({"id": chunk_id, "section": section, "text": text})

    pi = profile.get("personal_info", {})
    add_chunk(
        "personal_info",
        "personal_info",
        "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in pi.items() if v),
    )

    comp = profile.get("compensation", {})
    comp_lines = []
    for k, v in comp.items():
        if not v:
            continue
        display = _format_compensation_value(k, v)
        comp_lines.append(f"{k.replace('_', ' ').title()}: {display}")
    add_chunk("compensation", "compensation", "\n".join(comp_lines))

    if profile.get("summary"):
        add_chunk("summary", "summary", f"Professional Summary:\n{profile['summary']}")

    for i, edu in enumerate(profile.get("education", [])):
        lines = [
            f"Degree: {edu.get('degree', '')}",
            f"Major: {edu.get('major', '')}",
            f"Institution: {edu.get('institution', '')}",
            f"Year: {edu.get('year_of_graduation', '')}",
            f"CGPA/Percentage: {edu.get('cgpa_or_percentage', '')}",
        ]
        coursework = edu.get("relevant_coursework", [])
        if coursework:
            lines.append(f"Relevant Coursework: {', '.join(coursework)}")
        add_chunk(f"education_{i}", "education", "\n".join(lines))

    for i, work in enumerate(profile.get("work_experience", [])):
        lines = [
            f"Company: {work.get('company', '')}",
            f"Role: {work.get('role', '')}",
            f"Duration: {work.get('duration', '')}",
        ]
        if work.get("responsibilities"):
            lines.append("Responsibilities:\n- " + "\n- ".join(work["responsibilities"]))
        if work.get("tech_used"):
            lines.append(f"Technologies: {', '.join(work['tech_used'])}")
        if work.get("achievements"):
            lines.append("Achievements:\n- " + "\n- ".join(work["achievements"]))
        add_chunk(f"work_{i}", "work_experience", "\n".join(lines))

    skills = profile.get("skills", {})
    for category, items in skills.items():
        if items:
            label = category.replace("_", " ").title()
            add_chunk(
                f"skills_{category}",
                "skills",
                f"{label}: {', '.join(items)}",
            )

    for i, proj in enumerate(profile.get("projects", [])):
        lines = [
            f"Project: {proj.get('name', '')}",
            f"Description: {proj.get('description', '')}",
        ]
        if proj.get("tech_stack"):
            lines.append(f"Tech Stack: {', '.join(proj['tech_stack'])}")
        if proj.get("github_url"):
            lines.append(f"GitHub: {proj['github_url']}")
        if proj.get("highlights"):
            lines.append("Highlights:\n- " + "\n- ".join(proj["highlights"]))
        add_chunk(f"project_{i}", "projects", "\n".join(lines))

    for i, pub in enumerate(profile.get("publications", [])):
        lines = [
            f"Title: {pub.get('title', '')}",
            f"Venue: {pub.get('venue', '')}",
            f"Year: {pub.get('year', '')}",
            f"Description: {pub.get('description', '')}",
        ]
        if pub.get("url"):
            lines.append(f"URL: {pub['url']}")
        add_chunk(f"publication_{i}", "publications", "\n".join(lines))

    for i, cert in enumerate(profile.get("certifications", [])):
        lines = [
            f"Certification: {cert.get('name', '')}",
            f"Issuer: {cert.get('issuer', '')}",
            f"Year: {cert.get('year', '')}",
        ]
        if cert.get("url"):
            lines.append(f"URL: {cert['url']}")
        add_chunk(f"certification_{i}", "certifications", "\n".join(lines))

    awards = profile.get("achievements_and_awards", [])
    if awards:
        add_chunk(
            "achievements",
            "achievements",
            "Achievements and Awards:\n- " + "\n- ".join(awards),
        )

    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_profile(profile: dict) -> int:
    """Chunk, embed, and upsert the full profile into Qdrant.

    Always does a full replace: deletes existing points first so stale
    chunks from a previous (larger) profile don't linger.
    Returns the number of chunks upserted.
    """
    from rag.vector_store import delete_all_chunks, upsert_chunks

    chunks = chunk_profile(profile)
    if not chunks:
        return 0

    # Wipe old data so a re-index is always clean
    delete_all_chunks()

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(chunks, embeddings)
    return len(chunks)