from agents.base_agent import BaseAgent

ROLE_ALIGNMENT_RULES = """
CONTEXT USAGE RULES (critical):
- The RELEVANT PROFILE CONTEXT below has already been pre-filtered for this role.
- ONLY mention experience or skills from the context that are relevant to the
  recipient's role or the job being referenced. Do not mention unrelated domains.
"""


class ColdDMAgent(BaseAgent):
    agent_id = "dm"
    agent_name = "LinkedIn Cold DM"
    agent_description = "Craft a concise, personalized LinkedIn direct message."

    def build_retrieval_role(self, inputs: dict) -> str:
        return inputs.get("recipient_role", "") or inputs.get("context", "")[:100]

    def build_retrieval_jd(self, inputs: dict) -> str:
        # DMs rarely have a full JD; use context as a proxy
        return inputs.get("context", "")

    def build_retrieval_query(self, inputs: dict) -> str:
        return f"{inputs.get('recipient_role', '')} {inputs.get('company', '')}"

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        return (
            "Write a LinkedIn cold DM (direct message) from the candidate to the recipient.\n\n"
            "Requirements:\n"
            "- Keep it under 300 characters if possible, max 500 characters\n"
            "- Professional but warm, not salesy\n"
            "- Reference something specific about the recipient's role or company\n"
            "- Mention 1 highly relevant skill or achievement from the profile\n"
            "- Include a clear, low-friction call to action\n"
            "- Use facts only from the profile\n\n"
            f"{ROLE_ALIGNMENT_RULES}\n"
            f"--- RELEVANT PROFILE CONTEXT (pre-filtered for this role) ---\n{context}\n\n"
            f"Recipient Name: {inputs.get('recipient_name', '')}\n"
            f"Recipient Role: {inputs.get('recipient_role', '')}\n"
            f"Company: {inputs.get('company', '')}\n"
            f"Context / Reason for Outreach: {inputs.get('context', '')}\n"
            f"Job Posting Link: {inputs.get('job_posting_link', 'N/A')}\n\n"
            "Output only the DM text, no subject line or labels."
        )