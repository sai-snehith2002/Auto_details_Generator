from agents.base_agent import BaseAgent


class FormFillAgent(BaseAgent):
    agent_id = "form"
    agent_name = "Google Form Fill"
    agent_description = "Generate answers for job application Google Form questions."

    def build_retrieval_query(self, inputs: dict) -> str:
        parts = [
            inputs.get("form_questions", ""),
            inputs.get("job_description", ""),
            inputs.get("company_name", ""),
        ]
        return " ".join(p for p in parts if p)

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        questions = inputs.get("form_questions", "")
        job_desc = inputs.get("job_description", "")
        company = inputs.get("company_name", "")

        prompt = (
            "Using the candidate profile below, answer each form question. "
            "Format your response as:\n"
            "Q: [question]\nA: [your answer]\n\n"
            f"--- CANDIDATE PROFILE ---\n{context}\n\n"
            f"--- FORM QUESTIONS ---\n{questions}\n"
        )
        if company:
            prompt += f"\nCompany: {company}\n"
        if job_desc:
            prompt += f"\nJob Description:\n{job_desc}\n"
        prompt += (
            "\nProvide concise, honest answers grounded in the profile. "
            "For compensation questions, express CTC in LPA (lakhs per annum) "
            "as shown in the profile (e.g. 7 LPA, 12 LPA). "
            "For notice period, use the exact value from the profile."
        )
        return prompt
