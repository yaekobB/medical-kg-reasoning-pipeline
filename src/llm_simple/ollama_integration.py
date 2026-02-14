from __future__ import annotations

"""
Ollama LLM integration for the medical-kg-reasoning project.

This module does NOT run SPARQL itself; instead it:
  - takes structured KG results (from sparql_queries_llm)
  - builds prompts
  - calls a local LLM via Ollama for natural-language explanations.

It exposes:
  - llm_explain_differential(symptoms, candidates) -> str
  - llm_explain_disease(expl) -> str

and a CLI demo:
  python -m src.llm.ollama_integration
"""

from typing import List
import os
import ollama

from src.sparql.sparql_queries_llm import (
    DifferentialCandidate,
    DiseaseExplanation,
    load_graph_for_llm,
    kg_differential_diagnosis,
    kg_explain_disease,
)

# ================================================================
# 1. Low-level Ollama call
# ================================================================


def call_ollama_chat(prompt: str, model: str | None = None) -> str:
    """
    Works in Docker by using OLLAMA_BASE_URL to reach the host Ollama server.

    Requires:
      - Ollama running on host with: OLLAMA_HOST=0.0.0.0:11434
      - docker-compose sets OLLAMA_BASE_URL=http://host.docker.internal:11434
    """
    print(f"[LLM] Calling Ollama chat model...")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")

    client = ollama.Client(host=base_url)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical explanation assistant working on top of a "
                    "symbolic knowledge graph. You MUST NOT give real medical advice. "
                    "Use ONLY the provided knowledge graph context, and always add "
                    "a disclaimer that this is not a diagnosis or treatment guidance."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]



# ================================================================
# 2. High-level helpers for the UI
# ================================================================


def llm_explain_differential(
    symptoms: List[str],
    candidates: List[DifferentialCandidate],
    model: str = "llama3.2:1b",
) -> str:
    """
    Given:
      - a list of symptom strings
      - a list of DifferentialCandidate from the KG

    Build a concise context and ask the LLM for an explanation.

    Returns the LLM's answer as a string.
    """
    if not candidates:
        return (
            "The knowledge graph did not return any candidate diseases for these "
            "symptoms. I cannot provide a differential diagnosis. "
            "Please consult a medical professional. (KG-based explanation)"
        )
        
    print(f"[LLM] Preparing prompt for differential explanation...")
    context_lines = []
    for i, c in enumerate(candidates, start=1):
        context_lines.append(
            f"{i}. {c.disease_label} | system: {c.system_label} | "
            f"matching symptoms: {', '.join(sorted(c.matched_symptoms))}"
        )

    context = "\n".join(context_lines)
    symptoms_str = ", ".join(symptoms)

    user_question = (
        f"A patient presents with the following symptoms: {symptoms_str}.\n"
        "Using ONLY the knowledge graph candidates listed above, explain which "
        "diseases are plausible and how the symptoms relate to them. Do NOT "
        "provide real medical advice or treatment suggestions. Summarize the "
        "KG evidence and end with a clear disclaimer."
    )

    prompt = f"""
We have a medical knowledge graph that returned the following candidate diseases
for the symptom set [{symptoms_str}]:

{context}

{user_question}
"""

    return call_ollama_chat(prompt, model=model)


def llm_explain_disease(
    expl: DiseaseExplanation,
    model: str = "llama3.2:1b",
) -> str:
    """
    Given a DiseaseExplanation object from the KG, ask the LLM to explain it.

    Returns the LLM's answer as a string.
    """
    print(f"[LLM] Preparing prompt for disease explanation...")
    systems = ", ".join(expl.system_labels or ["Unknown / not classified"])
    types = ", ".join(expl.type_labels or ["med:Disease"])
    symptoms = ", ".join(expl.symptom_labels or ["none listed"])

    context = f"""
Disease label: {expl.disease_label}
URI          : {expl.disease_uri}
Body systems : {systems}
Types        : {types}
Symptoms     : {symptoms}
"""

    user_question = (
        f"Using ONLY the above knowledge graph information, explain in simple terms "
        f"what {expl.disease_label} is, which body system(s) it affects, and how the "
        f"listed symptoms relate to it. Do NOT give any treatment or diagnosis advice. "
        f"End with a clear disclaimer that this is just an educational summary based "
        f"on a knowledge graph, not medical guidance."
    )

    prompt = f"""
We have a medical knowledge graph entry for a disease.

{context}

{user_question}
"""

    return call_ollama_chat(prompt, model=model)


# ================================================================
# 3. CLI demo entry point
# ================================================================


def main() -> None:
    # Load KG once
    g = load_graph_for_llm()

    # DEMO 1: differential diagnosis
    symptoms = ["fever", "cough", "headache"]
    candidates = kg_differential_diagnosis(g, symptoms, limit=10)

    print(f"[LLM] KG candidates for symptoms {symptoms}:")
    for c in candidates:
        print(
            f" - {c.disease_label} (system={c.system_label}, "
            f"matches={len(c.matched_symptoms)}, "
            f"symptoms={sorted(c.matched_symptoms)})"
        )

    print("\n[LLM] Asking Ollama for a differential explanation...\n")
    diff_answer = llm_explain_differential(symptoms, candidates)
    print(diff_answer)
    print("\n  LLM  differential explanation Done..")

    print("\n" + "=" * 80 + "\n")

    # DEMO 2: single disease explanation
    disease_label = "Pneumonia"
    expl = kg_explain_disease(g, disease_label)
    if expl is None:
        print(f"[LLM] Disease '{disease_label}' not found in KG.")
        return

    print(f"[LLM] KG info for disease '{disease_label}':")
    print(f"  URI    : {expl.disease_uri}")
    print(f"  Systems: {', '.join(expl.system_labels or ['Unknown / not classified'])}")
    print(f"  Types  : {', '.join(expl.type_labels or ['med:Disease'])}")
    print(f"  Symptoms: {', '.join(expl.symptom_labels or ['none listed'])}")

    print("\n[LLM] Asking Ollama for a disease explanation...\n")
    disease_answer = llm_explain_disease(expl)
    print(disease_answer)
    print("\n  LLM  disease explanation Done..")


if __name__ == "__main__":
    main()
