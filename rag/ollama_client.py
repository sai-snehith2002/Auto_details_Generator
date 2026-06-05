from ollama import Client

from config import OLLAMA_API_KEY, OLLAMA_BASE_URL


def get_ollama_client() -> Client:
    headers = {}
    if OLLAMA_API_KEY and OLLAMA_API_KEY != "your_key_here":
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return Client(host=OLLAMA_BASE_URL, headers=headers or None)
