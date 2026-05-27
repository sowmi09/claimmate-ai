import requests
from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL, timeout: int = 90) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]

def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
