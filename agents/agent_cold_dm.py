from agents.base_agent import BaseAgent


class ColdDMAgent(BaseAgent):
    agent_id = "dm"
    agent_name = "LinkedIn Cold DM"
    agent_description = "Craft a concise, personalized LinkedIn direct message."

    def build_retrieval_query(self, inputs: dict) -> str:
        parts = [
            inputs.get("recipient_role", ""),
            inputs.get("company", ""),
            inputs.get("context", ""),
            inputs.get("job_posting_link", ""),
        ]
        return " ".join(p for p in parts if p)

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        return (
            "Write a LinkedIn cold DM (direct message) from the candidate to the recipient.\n\n"
            "Requirements:\n"
            "- Keep it under 300 characters if possible, max 500 characters\n"
            "- Professional but warm, not salesy\n"
            "- Reference something specific about the recipient or company when possible\n"
            "- Include a clear, low-friction call to action\n"
            "- Use facts only from the profile\n\n"
            f"--- CANDIDATE PROFILE ---\n{context}\n\n"
            f"Recipient Name: {inputs.get('recipient_name', '')}\n"
            f"Recipient Role: {inputs.get('recipient_role', '')}\n"
            f"Company: {inputs.get('company', '')}\n"
            f"Context / Reason for Outreach: {inputs.get('context', '')}\n"
            f"Job Posting Link: {inputs.get('job_posting_link', 'N/A')}\n\n"
            "Output only the DM text, no subject line or labels."
        )
