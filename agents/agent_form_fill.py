from agents.base_agent import BaseAgent

ROLE_ALIGNMENT_RULES = """
CONTEXT USAGE RULES (critical):
- The RELEVANT PROFILE CONTEXT below has already been pre-filtered for this role/JD.
- Each chunk is tagged with a relevance score — prioritise higher-scored chunks.
- Answer each question using ONLY experience and skills that are relevant to this role.
- If the question is about skills or experience, only mention items from the context
  that align with the job description / company. Do not list everything in the profile.
- For generic questions (e.g. "tell us about yourself"), focus on the most role-relevant
  experience first.
"""


class FormFillAgent(BaseAgent):
    agent_id = "form"
    agent_name = "Google Form Fill"
    agent_description = "Generate answers for job application Google Form questions."

    def build_retrieval_role(self, inputs: dict) -> str:
        # Infer role from questions or company if no explicit title given
        questions = inputs.get("form_questions", "")
        company = inputs.get("company_name", "")
        return f"{company} {questions[:200]}"

    def build_retrieval_jd(self, inputs: dict) -> str:
        return inputs.get("job_description", "")

    def build_retrieval_query(self, inputs: dict) -> str:
        return inputs.get("company_name", "") or inputs.get("form_questions", "")[:100]

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        questions = inputs.get("form_questions", "")
        job_desc = inputs.get("job_description", "")
        company = inputs.get("company_name", "")

        prompt = (
            "Using the candidate profile context below, answer each form question.\n"
            "Format your response as:\nQ: [question]\nA: [your answer]\n\n"
            f"{ROLE_ALIGNMENT_RULES}\n"
            f"--- RELEVANT PROFILE CONTEXT (pre-filtered for this role) ---\n{context}\n\n"
            f"--- FORM QUESTIONS ---\n{questions}\n"
        )
        if company:
            prompt += f"\nCompany: {company}\n"
        if job_desc:
            prompt += f"\nJob Description:\n{job_desc}\n"
        prompt += (
            "\nProvide concise, honest answers grounded in the profile. "
            "For compensation questions, express CTC in LPA (lakhs per annum). "
            "For notice period, use the exact value from the profile. "
            "Only reference skills and projects relevant to this role and company."
        )
        return prompt