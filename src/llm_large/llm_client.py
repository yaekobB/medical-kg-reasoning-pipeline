# src/assistant/llm_client.py
from __future__ import annotations

import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def run_kg_explanation_prompt(prompt: str) -> str:
    """
    Call Ollama via HTTP. Works in Docker when OLLAMA_BASE_URL points to host:
      http://host.docker.internal:11434
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

