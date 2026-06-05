from agents.base_agent import BaseAgent
from agents.email_prompts import (
    EMAIL_SYSTEM_PROMPT,
    STRICT_OUTPUT_RULES,
    build_identity_section,
    build_job_section,
)


class JobEmailAgent(BaseAgent):
    agent_id = "jobemail"
    agent_name = "Cold Job Outreach"
    agent_description = "Draft a cold outreach email to a hiring manager or recruiter."

    def build_system_prompt(self) -> str:
        return EMAIL_SYSTEM_PROMPT

    def build_retrieval_query(self, inputs: dict) -> str:
        parts = [
            inputs.get("job_title", ""),
            inputs.get("company_name", ""),
            inputs.get("job_description", ""),
            "work experience skills projects qualifications",
        ]
        return " ".join(p for p in parts if p)

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        identity = build_identity_section()
        job_details = build_job_section(
            inputs,
            [
                ("recipient_name", "Recipient Name"),
                ("company_name", "Company Name"),
                ("job_title", "Job Title"),
            ],
        )
        job_desc = inputs.get("job_description", "").strip()

        return (
            "Write a cold job outreach email from the candidate to the recipient below.\n\n"
            "The email MUST be specifically tailored to the job title and job description. "
            "Analyze the JD to understand what the role needs, then select matching experience "
            "from the profile. Express genuine interest based on the role requirements and company — "
            "do not ask the user why they are interested; infer it from the JD and profile fit.\n\n"
            "Requirements:\n"
            "- Professional and concise (under 250 words)\n"
            "- Open with why the candidate is a strong fit for THIS specific role at THIS company\n"
            "- Reference 2-3 concrete skills/experiences from the profile that match JD requirements\n"
            "- Address the recipient by their exact name\n"
            "- Use the exact company name throughout — never a placeholder\n"
            "- Clear call to action (brief call or conversation about this role)\n"
            "- Sign off with the candidate's real full name from CANDIDATE IDENTITY\n"
            "- Include Subject line referencing the exact job title\n\n"
            f"{STRICT_OUTPUT_RULES}\n\n"
            f"--- CANDIDATE IDENTITY (use for sign-off and contact details) ---\n{identity}\n\n"
            f"--- JOB APPLICATION DETAILS (use exact values, do not replace with placeholders) ---\n"
            f"{job_details}\n\n"
            f"--- JOB DESCRIPTION (this is the primary guide — tailor everything to these requirements) ---\n"
            f"{job_desc}\n\n"
            f"--- RELEVANT PROFILE CONTEXT (select only what matches this role) ---\n{context}\n\n"
            "Format:\nSubject: Application for [exact job title] — [candidate full name]\n\n"
            "[Complete email body]"
        )
