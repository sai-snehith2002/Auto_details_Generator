from user_profile.profile_store import load_profile

STRICT_OUTPUT_RULES = """STRICT OUTPUT RULES (must follow):
- NEVER use placeholders or fill-in blanks. Forbidden examples: [Your Name], [Company Name],
  [Insert X], <name>, {{company}}, "your company", "the company" (when a company name is given).
- Use the EXACT company name, job title, and recipient/contact name from JOB APPLICATION DETAILS.
- Use the candidate's REAL full name, email, and links from CANDIDATE IDENTITY for sign-off and contact info.
- Tailor the entire email to the specific job title and job description — not a generic template.
- Highlight only profile experience, projects, and skills that directly match the role/JD requirements.
- Write a complete, ready-to-send email. The user should be able to copy-paste without editing."""

EMAIL_SYSTEM_PROMPT = (
    "You are a professional career writing assistant. You write complete, ready-to-send emails "
    "tailored to a specific job application. Use ONLY facts from the candidate profile provided. "
    "Never invent credentials or experience. Never use placeholders — always fill in real names, "
    "company names, and contact details from the provided context."
)


def build_identity_section() -> str:
    """Always inject core identity fields so sign-off never needs placeholders."""
    profile = load_profile()
    pi = profile.get("personal_info", {})
    lines = []
    field_labels = {
        "full_name": "Full Name",
        "email": "Email",
        "phone": "Phone",
        "linkedin_url": "LinkedIn",
        "github_url": "GitHub",
        "portfolio_url": "Portfolio",
        "location": "Location",
    }
    for key, label in field_labels.items():
        if pi.get(key):
            lines.append(f"{label}: {pi[key]}")
    return "\n".join(lines) if lines else "Full Name: (not set in profile)"


def build_job_section(inputs: dict, fields: list[tuple[str, str]]) -> str:
    """Format job-specific form inputs as a structured block."""
    lines = []
    for key, label in fields:
        value = inputs.get(key, "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)
