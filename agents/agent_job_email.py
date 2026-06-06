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
- ONLY use experience, projects, and skills from the context that directly match
  the job description requirements. Ignore any chunk that is about a different
  domain (e.g. if this is an SDE role, do not mention BI dashboards or vice versa).
- If a chunk's content does not map to any requirement in the JD, skip it entirely.
- Never pad the output with loosely related experience just to fill space.
"""
 
 
class JobEmailAgent(BaseAgent):
    agent_id = "jobemail"
    agent_name = "Cold Job Outreach"
    agent_description = "Draft a cold outreach email to a hiring manager or recruiter."
 
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
                ("recipient_name", "Recipient Name"),
                ("company_name", "Company Name"),
                ("job_title", "Job Title"),
            ],
        )
        job_desc = inputs.get("job_description", "").strip()
 
        return (
            "Write a cold job outreach email from the candidate to the recipient below.\n\n"
            "The email MUST be specifically tailored to the job title and job description. "
            "Analyse the JD requirements first, then select only the matching experience "
            "from the profile context.\n\n"
            "Requirements:\n"
            "- Professional and concise (under 250 words)\n"
            "- Open with why the candidate is a strong fit for THIS specific role\n"
            "- Reference 2-3 concrete skills/experiences that directly match JD requirements\n"
            "- Address the recipient by their exact name\n"
            "- Use the exact company name throughout — never a placeholder\n"
            "- Clear call to action\n"
            "- Sign off with the candidate's real full name from CANDIDATE IDENTITY\n"
            "- Include Subject line referencing the exact job title\n\n"
            f"{ROLE_ALIGNMENT_RULES}\n"
            f"{STRICT_OUTPUT_RULES}\n\n"
            f"--- CANDIDATE IDENTITY ---\n{identity}\n\n"
            f"--- JOB APPLICATION DETAILS ---\n{job_details}\n\n"
            f"--- JOB DESCRIPTION (primary guide — every claim must map to a requirement here) ---\n"
            f"{job_desc}\n\n"
            f"--- RELEVANT PROFILE CONTEXT (pre-filtered for this role) ---\n{context}\n\n"
            "Format:\nSubject: Application for [exact job title] — [candidate full name]\n\n"
            "[Complete email body]"
        )