import json
from pathlib import Path

from config import PROFILE_PATH


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


def load_profile() -> dict:
    path = Path(PROFILE_PATH)
    if not path.exists():
        profile = empty_profile()
        save_profile(profile)
        return profile
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_profile(profile: dict) -> None:
    path = Path(PROFILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def profile_is_populated(profile: dict) -> bool:
    info = profile.get("personal_info", {})
    return bool(info.get("full_name") or profile.get("summary"))
