"""
profile_store.py  –  Cloud-native profile persistence via Qdrant.

The profile JSON is stored as a single special point (id=0) in the Qdrant
collection alongside the chunk points.  No local filesystem access at all.

Point layout
────────────
id      : 0
vector  : [0.0] * 384   (dummy — excluded from all searches via type filter)
payload : {
    "type": "profile_document",
    "data": "<json-serialised profile string>"
}

Why store data as a JSON string (not a raw dict)?
Qdrant flattens nested dicts in payloads in some SDK versions, which can
corrupt deeply-nested profile structures.  Storing as a string and parsing
on read is the safest cross-version approach.
"""

import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

logger = logging.getLogger(__name__)

PROFILE_POINT_ID = 0   # reserved — must match vector_store.py
VECTOR_SIZE = 384      # all-MiniLM-L6-v2 output dimension


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set. "
            "Check your environment variables."
        )
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
            ),
        )


# ── Empty profile template ────────────────────────────────────────────────────

def empty_profile() -> dict:
    return {
        "personal_info": {
            "full_name": "",
            "email": "",
            "phone": "",
            "linkedin_url": "",
            "github_url": "",
            "portfolio_url": "",
            "location": "",
        },
        "compensation": {
            "current_ctc": "",
            "expected_ctc": "",
            "notice_period": "",
            "years_of_experience": "",
        },
        "summary": "",
        "education": [],
        "work_experience": [],
        "skills": {
            "programming_languages": [],
            "frameworks_and_libraries": [],
            "databases": [],
            "cloud_and_devops": [],
            "ai_ml": [],
            "tools": [],
            "soft_skills": [],
        },
        "projects": [],
        "publications": [],
        "certifications": [],
        "achievements_and_awards": [],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def load_profile() -> dict:
    """Load the profile from Qdrant.

    Returns an empty profile dict if:
    - The point does not exist yet (first run)
    - Qdrant is unreachable
    - The stored payload is malformed

    Raises RuntimeError only if credentials are missing (config error).
    """
    try:
        client = _get_client()
        _ensure_collection(client)

        results = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[PROFILE_POINT_ID],
            with_payload=True,
        )

        if not results:
            logger.info("No profile found in Qdrant (first run). Returning empty profile.")
            return empty_profile()

        payload = results[0].payload or {}

        if payload.get("type") != "profile_document":
            logger.warning("Point 0 exists but is not a profile_document. Returning empty.")
            return empty_profile()

        raw = payload.get("data", "")

        # data is stored as a JSON string — parse it back to a dict
        if isinstance(raw, str):
            profile = json.loads(raw)
        elif isinstance(raw, dict):
            # Fallback: if an older save stored it as a dict directly
            profile = raw
        else:
            logger.warning("Unexpected data type in profile payload: %s", type(raw))
            return empty_profile()

        # Merge against empty_profile so any missing keys are always present
        base = empty_profile()
        base.update(profile)
        # Preserve nested dicts properly
        for key in ("personal_info", "compensation", "skills"):
            if key in profile and isinstance(profile[key], dict):
                base[key] = {**base.get(key, {}), **profile[key]}

        return base

    except RuntimeError:
        raise   # Missing credentials — let this bubble up visibly
    except Exception as exc:
        logger.error("Failed to load profile from Qdrant: %s", exc, exc_info=True)
        return empty_profile()


def save_profile(profile: dict) -> None:
    """Persist the profile dict to Qdrant as point id=0.

    Upserts — safe to call on every form save.  The chunk points are
    completely separate (id >= 1) and are not affected.
    """
    client = _get_client()
    _ensure_collection(client)

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=PROFILE_POINT_ID,
                vector=[0.0] * VECTOR_SIZE,
                payload={
                    "type": "profile_document",
                    "data": json.dumps(profile, ensure_ascii=False),
                },
            )
        ],
    )
    logger.info("Profile saved to Qdrant (point id=0).")


def profile_is_populated(profile: dict) -> bool:
    info = profile.get("personal_info", {})
    return bool(info.get("full_name") or profile.get("summary"))