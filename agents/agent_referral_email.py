from agents.base_agent import BaseAgent
from agents.email_prompts import (
    EMAIL_SYSTEM_PROMPT,
    STRICT_OUTPUT_RULES,
    build_identity_section,
    build_job_section,
)


class ReferralEmailAgent(BaseAgent):
    agent_id = "refmail"
    agent_name = "Referral Email"
    agent_description = "Draft a referral request email to a contact at the target company."

    def build_system_prompt(self) -> str:
        return EMAIL_SYSTEM_PROMPT

    def build_retrieval_query(self, inputs: dict) -> str:
        parts = [
            inputs.get("job_title", ""),
            inputs.get("company_name", ""),
            inputs.get("job_description", ""),
            inputs.get("contact_role", ""),
            "referral work experience skills qualifications",
        ]
        return " ".join(p for p in parts if p)

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
            "The email MUST be specifically tailored to the job title and job description below. "
            "Pick 2-3 qualifications from the profile that directly match what this role requires.\n\n"
            "Requirements:\n"
            "- Professional, respectful tone — acknowledge their time\n"
            "- Open by referencing the exact job title at the exact company name\n"
            "- Connect the candidate's relevant experience to specific JD requirements\n"
            "- Make it easy for them to refer (mention resume is available)\n"
            "- Sign off with the candidate's real full name from CANDIDATE IDENTITY\n"
            "- Include Subject line referencing the exact job title and company\n\n"
            f"{STRICT_OUTPUT_RULES}\n\n"
            f"--- CANDIDATE IDENTITY (use for sign-off and contact details) ---\n{identity}\n\n"
            f"--- JOB APPLICATION DETAILS (use exact values, do not replace with placeholders) ---\n"
            f"{job_details}\n"
        )
        if job_desc:
            prompt += f"\n--- JOB DESCRIPTION (tailor qualifications to these requirements) ---\n{job_desc}\n"
        prompt += (
            f"\n--- RELEVANT PROFILE CONTEXT (select only what matches this role) ---\n{context}\n\n"
            "Format:\nSubject: [exact job title] at [exact company name]\n\n[Complete email body]"
        )
        return prompt
