"""
Intent parsing for the LARGE medical KG assistant (Streamlit tab).

This module is deliberately *rule-based* and does NOT depend on the mini-project
code or any LLM wrapper. Later, if you want, we can add an LLM layer on top.

It takes a natural language question and converts it into a structured dict:

{
    "intent": "diseases_for_symptoms" | "treatments_for_disease",
    "symptoms": [list of symptom labels from KG],
    "disease": disease label from KG or None,
}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Small helper: normalize text
# ---------------------------------------------------------------------

def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    """
    Very simple tokenization: split on non-letters/digits.
    """
    return [t for t in re.split(r"[^a-z0-9\-]+", _normalize(text)) if t]


# ---------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------

def parse_question_to_slots(
    question: str,
    known_symptoms: List[str],
    known_diseases: List[str],
) -> Dict[str, Any]:
    """
    Convert a free-text question into an intent + slots structure.

    Intent values:
      - "diseases_for_symptoms":
          • User gives symptoms and wants candidate diseases
          • Or user asks for the symptoms of a specific disease
      - "treatments_for_disease":
          • User gives a disease and asks for treatments

    Slots:
      - "symptoms": list[str] – canonical labels of symptoms in the KG
      - "disease": str | None – canonical disease label in the KG (if detected)
    """
    q = question or ""
    q_low = _normalize(q)
    tokens = _tokenize(q)

    # -----------------------------------------------------------------
    # 1) Detect if user is asking about TREATMENT or SYMPTOMS
    # -----------------------------------------------------------------
    is_treatment_query = any(
        kw in q_low
        for kw in [
            "treat", "treatment", "treated", "therapy", "manage", "cure",
            "medication", "drug", "medicine"
        ]
    )

    is_symptom_query_for_disease = any(
        kw in q_low
        for kw in [
            "symptom of", "symptoms of", "signs of", "what are the symptoms",
            "give me information about", "info about", "tell me about"
        ]
    )

    # -----------------------------------------------------------------
    # 2) Match disease names from KG
    # -----------------------------------------------------------------
    disease_label: Optional[str] = None
    disease_labels_norm = {d.lower(): d for d in known_diseases}

    # Try exact label containment (case-insensitive)
    for d_norm, original in disease_labels_norm.items():
        if d_norm in q_low:
            disease_label = original
            break

    # If still None, try word-by-word match for multi-token diseases
    if disease_label is None:
        for d_norm, original in disease_labels_norm.items():
            # e.g., "diabetes type 2" -> ["diabetes", "type", "2"]
            d_tokens = d_norm.split()
            if len(d_tokens) == 1:
                continue
            if all(tok in tokens for tok in d_tokens):
                disease_label = original
                break

    # -----------------------------------------------------------------
    # 3) Match symptom names from KG
    # -----------------------------------------------------------------
    symptom_labels_norm = {s.lower(): s for s in known_symptoms}
    matched_symptoms: List[str] = []

    # Strategy: if a symptom label string appears in the question, we take it.
    # For hyphenated labels (e.g., "loss-of-smell"), we also check tokens.
    for s_norm, original in symptom_labels_norm.items():
        if not s_norm:
            continue

        if s_norm in q_low:
            matched_symptoms.append(original)
            continue

        # handle hyphenated symptoms via tokens
        s_tokens = s_norm.split("-")
        if all(tok in tokens for tok in s_tokens):
            matched_symptoms.append(original)

    # Deduplicate
    matched_symptoms = sorted(set(matched_symptoms))

    # -----------------------------------------------------------------
    # 4) Decide INTENT
    # -----------------------------------------------------------------
    if is_treatment_query and disease_label:
        # Example:
        #   "What are the treatments for Pneumonia?"
        #   "How is COVID-19 treated?"
        intent = "treatments_for_disease"
        symptoms = []
    elif is_symptom_query_for_disease and disease_label:
        # Example:
        #   "What are the symptoms of Common Cold?"
        #   -> we treat this as a symptom-query for a disease.
        #      Downstream code can interpret:
        #         intent="diseases_for_symptoms", disease=<label>, symptoms=[]
        intent = "diseases_for_symptoms"
        symptoms = []
    else:
        # Default assumption: user is giving symptoms and wants diseases.
        # Example:
        #   "I have cough and fever, what could it be?"
        #   "Fever and headache, what might I have?"
        intent = "diseases_for_symptoms"
        symptoms = matched_symptoms

    # -----------------------------------------------------------------
    # 5) Build result dict
    # -----------------------------------------------------------------
    slots: Dict[str, Any] = {
        "intent": intent,
        "symptoms": symptoms,
        "disease": disease_label,
    }

    return slots
