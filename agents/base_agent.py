from config import OLLAMA_MODEL
from rag.ollama_client import get_ollama_client
from rag.retriever import retrieve_context


class BaseAgent:
    agent_id: str = "base"
    agent_name: str = "Base Agent"
    agent_description: str = ""

    def build_retrieval_query(self, inputs: dict) -> str:
        return " ".join(str(v) for v in inputs.values() if v)

    def build_system_prompt(self) -> str:
        return (
            "You are a professional career assistant. Use ONLY the candidate profile "
            "context provided to generate accurate, tailored content. Do not invent "
            "facts, credentials, or experience not present in the profile. "
            "Write in a natural, professional tone."
        )

    def build_user_prompt(self, inputs: dict, context: str) -> str:
        raise NotImplementedError

    def build_refine_prompt(
        self, inputs: dict, context: str, previous_output: str, feedback: str
    ) -> str:
        base = self.build_user_prompt(inputs, context)
        return (
            f"{base}\n\n"
            f"--- PREVIOUS DRAFT ---\n{previous_output}\n\n"
            f"--- REFINEMENT FEEDBACK ---\n{feedback}\n\n"
            "Revise the previous draft based on the feedback. Keep facts accurate "
            "to the profile context. Never introduce placeholders — use real names, "
            "company names, and details from the provided context."
        )

    def generate(self, inputs: dict) -> str:
        query = self.build_retrieval_query(inputs)
        context = retrieve_context(query)
        system = self.build_system_prompt()
        user = self.build_user_prompt(inputs, context)

        client = get_ollama_client()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.message.content

    def refine(self, inputs: dict, previous_output: str, feedback: str) -> str:
        query = self.build_retrieval_query(inputs)
        context = retrieve_context(query)
        system = self.build_system_prompt()
        user = self.build_refine_prompt(inputs, context, previous_output, feedback)

        client = get_ollama_client()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.message.content
