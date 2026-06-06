from agents.base_agent import BaseAgent
from agents.email_prompts import (
    EMAIL_SYSTEM_PROMPT,
    STRICT_OUTPUT_RULES,
    build_identity_section,
    build_job_section,
)

ROLE_ALIGNMENT_RULES = """
CONTEXT USAGE RULES (critical):
- The RELEVANT PROFILE CONTEXT below has already been pre-filtered for this role.
- Each chunk is tagged with a relevance score — prioritise higher-scored chunks.
- ONLY reference experience, projects, and skills that directly match the JD requirements.
- Discard any context chunk that belongs to a different domain than this role.
- Never mention unrelated projects or technologies just because they exist in the profile.
"""


class ReferralEmailAgent(BaseAgent):
    agent_id = "refmail"
    agent_name = "Referral Email"
    agent_description = "Draft a referral request email to a contact at the target company."

    def build_system_prompt(self) -> str:
        return EMAIL_SYSTEM_PROMPT

    def build_retrieval_role(self, inputs: dict) -> str:
        return inputs.get("job_title", "")

    def build_retrieval_jd(self, inputs: dict) -> str:
        return inputs.get("job_description", "")

    def build_retrieval_query(self, inputs: dict) -> str:
        return f"{inputs.get('job_title', '')} {inputs.get('company_name', '')}"

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        identity = build_identity_section()
        job_details = build_job_section(
            inputs,
            [
                ("contact_name", "Contact Name"),
                ("contact_role", "Contact Role"),
                ("company_name", "Company Name"),
                ("job_title", "Job Title Applying For"),
                ("job_link", "Job Link"),
                ("relationship", "Relationship"),
            ],
        )
        job_desc = inputs.get("job_description", "").strip()

        prompt = (
            "Write a referral request email from the candidate to their contact.\n\n"
            "The email MUST be tailored to the exact job title and job description. "
            "Pick 2-3 qualifications from the profile that directly match what THIS role requires.\n\n"
            "Requirements:\n"
            "- Professional, respectful tone — acknowledge their time\n"
            "- Open by referencing the exact job title at the exact company name\n"
            "- Connect the candidate's relevant experience to specific JD requirements\n"
            "- Make it easy for them to refer (mention resume is available)\n"
            "- Sign off with the candidate's real full name from CANDIDATE IDENTITY\n"
            "- Include Subject line referencing the exact job title and company\n\n"
            f"{ROLE_ALIGNMENT_RULES}\n"
            f"{STRICT_OUTPUT_RULES}\n\n"
            f"--- CANDIDATE IDENTITY ---\n{identity}\n\n"
            f"--- JOB APPLICATION DETAILS ---\n{job_details}\n"
        )
        if job_desc:
            prompt += (
                f"\n--- JOB DESCRIPTION (tailor qualifications to these requirements) ---\n"
                f"{job_desc}\n"
            )
        prompt += (
            f"\n--- RELEVANT PROFILE CONTEXT (pre-filtered for this role) ---\n{context}\n\n"
            "Format:\nSubject: [exact job title] at [exact company name]\n\n[Complete email body]"
        )
        return prompt