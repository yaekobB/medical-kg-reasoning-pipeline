# src/assistant/explanations.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Sequence

from .kg_api_large import (
    get_all_disease_labels,
    get_all_symptom_labels,
    get_symptoms_for_disease,
    get_treatments_for_disease,
)

# Optional: body-system helper, if it exists in kg_api_large
try:  # type: ignore[assignment]
    from .kg_api_large import get_body_system_for_disease  # type: ignore[import]
except Exception:  # pragma: no cover - optional helper
    get_body_system_for_disease = None  # type: ignore[assignment]

from .intent_parsing import parse_question_to_slots
from .llm_client import run_kg_explanation_prompt

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _fmt_list(items: List[str]) -> str:
    """
    Nicely format a list of strings as a human-readable phrase.
    """
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _safe_get(cand: Any, names: List[str], default: Any) -> Any:
    """
    Robustly extract a field from either:
      - a dict, or
      - an object with attributes,
    trying several possible names.

    Example:
      _safe_get(c, ["disease_label", "label", "disease"], "Unknown disease")
    """
    # dict-style
    if isinstance(cand, dict):
        for n in names:
            if n in cand:
                return cand[n]

    # attribute-style (e.g., dataclass DiseaseMatch)
    for n in names:
        if hasattr(cand, n):
            return getattr(cand, n)

    return default


def _ensure_list(x: Any) -> List[str]:
    """
    Ensure we always get a list of strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [str(x)]

# ---------------------------------------------------------------------
# LLM explanation helpers
# ---------------------------------------------------------------------
def _llm_explain_treatments(
    disease_label: str,
    disease_symptoms: List[str],
    disease_treatments: List[str],
) -> str:
    """
    Use the LLM to explain, in plain language, why the KG links this disease
    to these treatments and symptoms.

    The LLM is strictly constrained: it may only explain KG content,
    not perform diagnosis or give medical advice.
    """
    data = {
        "task": "treatments_for_disease",
        "disease": disease_label,
        "symptoms_from_kg": disease_symptoms,
        "treatments_from_kg": disease_treatments,
    }
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    
    print(f"LLM explain treatments input data_json.....: {data_json}")

    prompt = f"""
You are an assistant that EXPLAINS the contents of a medical knowledge graph.
The knowledge graph may be incomplete or simplified. Your job is ONLY to
explain, in neutral and educational terms, what the graph says and why these
triples might be related.

SAFETY RULES:
- Do NOT perform medical diagnosis.
- Do NOT suggest treatments, dosages, or medications beyond what is already
  explicitly present in the data.
- Do NOT give medical recommendations or tell the user what they should do.
- Emphasize that this is NOT medical advice and is based ONLY on the graph.

Here is the data from the knowledge graph (JSON):
{data_json}

Write 4–6 sentences in plain English that:
- Restate which symptoms and treatments are linked to this disease in the KG.
- Explain, at a high level, why the KG might connect these symptoms and
  treatments (e.g., “these are typical management strategies”).
- Explicitly say that this is KG-based and NOT medical advice.
""".strip()

    try:
        return run_kg_explanation_prompt(prompt)
    except Exception:
        # If the LLM call fails or refuses, higher-level code will provide a fallback.
        return ""


def _llm_explain_differential(
    user_symptoms: List[str],
    candidates: List[Dict[str, Any]],
) -> str:
    """
    Ask the LLM to narratively explain the differential diagnosis ranking,
    based purely on KG matches (no new knowledge, no diagnosis).
    """
    payload = {
        "task": "differential_diagnosis_from_kg",
        "user_symptoms": user_symptoms,
        "candidates": [
            {
                "disease": c["disease_label"],
                "body_system": c["body_system"],
                "num_matches": c["num_matches"],
                "total_symptoms": c["total_symptoms"],
                "matching_symptoms": c["matching_symptoms"],
                "other_symptoms": c["other_symptoms"],
            }
            for c in candidates
        ],
    }
    data_json = json.dumps(payload, ensure_ascii=False, indent=2)
    
    print(f"LLM explain differential input data_json....: {data_json}")

    prompt = f"""
You are explaining the output of a medical knowledge graph.
The graph links diseases to symptoms, and we have computed a ranked list of
candidate diseases based on how many symptoms they share with the user's
reported symptoms.

SAFETY RULES:
- Do NOT diagnose the user.
- Do NOT recommend treatments, medications, or actions.
- Do NOT say that the user "has" a disease.
- You may ONLY explain, in general terms, why these candidates appear in
  the ranking, given the symptom overlaps in the knowledge graph.
- End with a clear statement that this is NOT medical advice.

Here is the KG-based ranking (JSON):
{data_json}

Write 5–7 sentences that:
- Mention which disease(s) appear near the top of the ranking and why
  (which symptoms they share with the user's list).
- Briefly mention that other candidates share fewer symptoms.
- Emphasize that this is only based on KG symptom overlaps and cannot
  substitute for real medical evaluation.
""".strip()

    try:
        return run_kg_explanation_prompt(prompt)
    except Exception:
        return ""


def _llm_explain_symptoms(
    disease_label: str,
    disease_symptoms: List[str],
) -> str:
    """
    Ask the LLM to explain, in natural language, the symptom profile of
    a disease according to the KG.
    """
    data = {
        "task": "symptoms_for_disease",
        "disease": disease_label,
        "symptoms_from_kg": disease_symptoms,
    }
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    
    print(f"LLM explain symptoms input data_json....: {data_json}")

    prompt = f"""
You are explaining how a medical knowledge graph describes a disease.

SAFETY RULES:
- Do NOT diagnose the user.
- Do NOT give medical advice.
- Only restate and interpret what the knowledge graph says.

Here is the KG data (JSON):
{data_json}

Write 3–5 sentences that:
- State that the knowledge graph links this disease to the listed symptoms.
- Treat this as a general, non-personal description (like an encyclopedia),
  not as a diagnosis.
- Explicitly say that this is based on the KG and NOT medical advice.
""".strip()

    try:
        return run_kg_explanation_prompt(prompt)
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Helpers to construct "triples used in reasoning"
# ---------------------------------------------------------------------

def _triples_for_disease_symptoms_and_treatments(
    disease_label: str,
    symptoms: Sequence[str],
    treatments: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Build a list of label-level triples used in reasoning for a single disease.

    We don't need to hit rdflib again here: we already know which symptoms
    and treatments were retrieved from the KG, so we can expose them as
    (subject, predicate, object) triples in a clean, explainable way.
    """
    triples: List[Dict[str, str]] = []

    for s in symptoms:
        triples.append(
            {
                "subject": disease_label,
                "predicate": "med:hasSymptom",
                "object": s,
                "role": "symptom_edge",
            }
        )

    if treatments:
        for t in treatments:
            triples.append(
                {
                    "subject": disease_label,
                    "predicate": "med:treatedWith",
                    "object": t,
                    "role": "treatment_edge",
                }
            )

    return triples

def _triples_for_differential(
    symptoms: Sequence[str],
    candidates: Sequence[Any],
    max_diseases: int = 5,
) -> List[Dict[str, str]]:
    """
    Build a compact triple list showing which disease–symptom edges
    were actually used when computing the differential diagnosis.

    For each of the top `max_diseases` candidates we expose:
      - triples for matching symptoms   (role='matching_symptom')
      - triples for other known symptoms (role='other_symptom')

    Works whether `candidates` are dataclass objects or plain dicts.
    """
    triples: List[Dict[str, str]] = []

    def _get(c: Any, key: str, default=None):
        # Support both dataclass-style objects and dicts
        if hasattr(c, key):
            return getattr(c, key)
        if isinstance(c, dict):
            return c.get(key, default)
        return default

    for cand in candidates[:max_diseases]:
        disease_label = _get(cand, "disease_label", None) or _get(
            cand, "label", "Unknown disease"
        )
        # Ensure we always treat these as lists
        matching = list(_get(cand, "matching_symptoms", []) or [])
        other = list(_get(cand, "other_symptoms", []) or [])

        for s in matching:
            triples.append(
                {
                    "subject": disease_label,
                    "predicate": "med:hasSymptom",
                    "object": s,
                    "role": "matching_symptom",
                }
            )

        for s in other:
            triples.append(
                {
                    "subject": disease_label,
                    "predicate": "med:hasSymptom",
                    "object": s,
                    "role": "other_symptom",
                }
            )

    return triples

# ---------------------------------------------------------------------
# Symbolic explanation helpers
# ---------------------------------------------------------------------
def _build_symbolic_treatments_text(
    disease_label: str,
    disease_symptoms: List[str],
    disease_treatments: List[str],
) -> str:
    """
    Build the symbolic (pure KG-based) explanation text for
    the 'treatments_for_disease' intent.
    """
    lines: List[str] = []
    lines.append(
        f"According to the large medical knowledge graph, '{disease_label}' "
        "is a disease node in the graph."
    )
    if disease_symptoms:
        lines.append(
            f"It is associated with the following symptoms in the KG: "
            f"{_fmt_list(disease_symptoms)}."
        )
    else:
        lines.append("The KG does not list any explicit symptom nodes for this disease.")
    if disease_treatments:
        lines.append(
            "It is linked via med:treatedWith to the following treatments: "
            f"{_fmt_list(disease_treatments)}."
        )
    else:
        lines.append(
            "The KG does not contain any med:treatedWith edges for this disease."
        )
    lines.append(
        "All of this information comes directly from the KG and is intended only "
        "for educational and analytical purposes, not as medical advice."
    )
    return " ".join(lines)


def _build_symbolic_differential_text(
    symptoms: List[str],
    candidates: List[Any],
) -> str:
    """
    Build the symbolic (pure KG-based) explanation text for
    the 'diseases_for_symptoms' intent when user gives symptoms.
    """
    if not candidates:
        return (
            "Based on the large medical knowledge graph, no diseases were found that "
            f"share the reported symptoms: {_fmt_list(symptoms)}. "
            "This may simply reflect gaps in the dataset rather than clinical reality. "
            "This output is derived purely from the structure of the knowledge graph "
            "and is NOT a medical diagnosis or recommendation."
        )

    lines: List[str] = []
    lines.append(
        "Based on the large medical knowledge graph, the following diseases share "
        "your reported symptoms (ranked by number of matching symptoms):"
    )

    for idx, cand in enumerate(candidates, start=1):
        disease_label = _safe_get(cand, ["disease_label", "label", "disease"], "Unknown disease")
        body_system = _safe_get(
            cand,
            ["body_system", "system"],
            "Unknown / not classified",
        )
        num_matches = _safe_get(cand, ["num_matches", "matches", "match_count"], 0)
        total_symptoms = _safe_get(cand, ["total_symptoms", "total"], 0)

        matching = _ensure_list(
            _safe_get(cand, ["matching_symptoms", "matching"], [])
        )
        others = _ensure_list(
            _safe_get(cand, ["other_symptoms", "other"], [])
        )

        matching_str = _fmt_list(matching) if matching else "none"
        others_str = _fmt_list(others) if others else "none"

        lines.append(
            f"{idx}. **{disease_label}:** (system: {body_system}; "
            f"matches: {num_matches}/{total_symptoms} symptoms; "
            f"matching: {matching_str}; other typical symptoms: {others_str})"
        )

    lines.append(
        "This output is derived purely from the structure of the knowledge graph "
        "and is NOT a medical diagnosis or recommendation."
    )

    return "\n".join(lines)


def _build_symbolic_symptoms_text(
    disease_label: str,
    disease_symptoms: List[str],
) -> str:
    """
    Build the symbolic explanation text for disease → symptoms queries
    ('What are the symptoms of X?').
    """
    if not disease_symptoms:
        return (
            f"According to the knowledge graph, there are currently no explicit symptom "
            f"nodes linked to '{disease_label}'. This may reflect incompleteness of the "
            "dataset rather than clinical reality. These facts come directly from the "
            "knowledge graph and are for educational purposes only. They are NOT "
            "medical advice."
        )

    lines: List[str] = []
    lines.append(f"According to the knowledge graph, '{disease_label}' has the following symptoms:")
    for s in disease_symptoms:
        lines.append(f"- {s}")
    lines.append(
        "\nThese facts come directly from the knowledge graph and are for educational "
        "purposes only. They are NOT medical advice."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Differential computation (our own, instead of kg_api_large.diseases_for_symptoms)
# ---------------------------------------------------------------------


def _normalize_symptom_token(s: str) -> str:
    """
    Normalize a symptom string for matching:
    - strip whitespace
    - drop trailing punctuation / brackets
    - lowercase
    """
    if not s:
        return ""
    s = s.strip()
    # remove some noisy trailing characters
    s = s.rstrip(").,; ")
    return s.lower()


def _compute_differential_candidates(
    symptoms: List[str],
    max_results: int = 15,
) -> List[Dict[str, Any]]:
    """
    Compute disease candidates directly from the large KG using:
      - get_all_disease_labels()
      - get_symptoms_for_disease()

    This version:
    - normalizes symptom strings (to avoid 'fatigue' vs 'fatigue)')
    - deduplicates
    - limits the number of 'other typical symptoms' for readability.
    """
    # Keep original user symptoms for display, but normalize for matching
    user_syms_norm = {
        _normalize_symptom_token(s) for s in symptoms if _normalize_symptom_token(s)
    }
    if not user_syms_norm:
        return []

    candidates: List[Dict[str, Any]] = []

    all_diseases = get_all_disease_labels()
    for disease_label in all_diseases:
        raw_syms = get_symptoms_for_disease(disease_label)
        if not raw_syms:
            continue

        # Map normalized → "pretty" (first occurrence)
        norm_to_pretty: Dict[str, str] = {}
        for s in raw_syms:
            norm = _normalize_symptom_token(s)
            if not norm:
                continue
            pretty = s.strip().rstrip(").,; ")
            if norm not in norm_to_pretty:
                norm_to_pretty[norm] = pretty

        disease_syms_norm = set(norm_to_pretty.keys())
        if not disease_syms_norm:
            continue

        # Overlap with user symptoms (in normalized space)
        matching_norm = user_syms_norm.intersection(disease_syms_norm)
        if not matching_norm:
            # no shared symptoms → not a candidate
            continue

        others_norm = disease_syms_norm - user_syms_norm

        # Pretty lists (deduped, cleaned)
        matching_list = [norm_to_pretty[n] for n in sorted(matching_norm)]
        other_list = [norm_to_pretty[n] for n in sorted(others_norm)]

        # Limit length of "other typical symptoms" for readability
        MAX_OTHERS = 6
        if len(other_list) > MAX_OTHERS:
            other_list = other_list[:MAX_OTHERS]

        total = len(disease_syms_norm)

        # Try to get body system if helper is available, otherwise default
        if callable(get_body_system_for_disease):
            try:
                system = get_body_system_for_disease(disease_label)  # type: ignore[misc]
                if not system:
                    system = "Unknown / not classified"
            except Exception:
                system = "Unknown / not classified"
        else:
            system = "Unknown / not classified"

        candidates.append(
            {
                "disease_label": disease_label,
                "body_system": system,
                "num_matches": len(matching_list),
                "total_symptoms": total,
                "matching_symptoms": matching_list,
                "other_symptoms": other_list,
            }
        )

    # Rank:
    #  - more matches first
    #  - then more total symptoms (richer profile)
    #  - then alphabetical by label for stability
    candidates.sort(
        key=lambda c: (
            -c["num_matches"],
            -c["total_symptoms"],
            c["disease_label"],
        )
    )

    return candidates[:max_results]

# ---------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------
def answer_question_with_large_kg(question: str) -> Dict[str, Any]:
    """
    Main entrypoint used by the large project's assistant / UI.

    Given a natural language question, it:
      1) Uses the LLM (via intent_parsing) to classify the intent and slots.
      2) Uses the large medical KG for symbolic reasoning.
      3) Produces:
           - 'symbolic_text': pure KG-based explanation
           - 'llm_text'     : LLM-generated narrative constrained by the KG
           - 'intent'       : detected intent string
           - 'slots'        : raw slot dictionary from the parser
           - 'triples'      : list of KG triples (subject, predicate, object, role)
                              that were actually used in the reasoning
           - 'candidates'   : disease candidates (for differential case)
           - 'mode'         : fixed string "large-kg" (for debug)
    """
    print(f"[LLM] QUESTION: {question}")
    # 1) Gather known labels for intent parsing
    known_symptoms = get_all_symptom_labels()
    known_diseases = get_all_disease_labels()

    print(f"[LLM] Rule-based Intent parsing....:")
    # 2) Let the intent parser (Rule based) interpret the question
    slots = parse_question_to_slots(question, known_symptoms, known_diseases)
    intent = slots.get("intent")
    symptoms = slots.get("symptoms") or []
    disease = slots.get("disease")
    print(f"[LLM] Intent parsing Done....: intent={intent}, disease={disease}, symptoms={symptoms}")
    symbolic_text: Optional[str] = None
    llm_text: Optional[str] = None
    triples_used: List[Dict[str, str]] = []
    candidates: List[Any] = []  # will be filled only in differential case

    # ------------------------------------------------------------------
    # Case A: treatments_for_disease
    # ------------------------------------------------------------------
    if intent == "treatments_for_disease" and disease:
        disease_label = disease
        disease_symptoms = get_symptoms_for_disease(disease_label)
        disease_treatments = get_treatments_for_disease(disease_label)

        symbolic_text = _build_symbolic_treatments_text(
            disease_label, disease_symptoms, disease_treatments
        )

        # NEW: triples used in reasoning (disease + symptoms + treatments)
        triples_used = _triples_for_disease_symptoms_and_treatments(
            disease_label,
            disease_symptoms,
            disease_treatments,
        )

        # LLM narrative (true LLM call, with fallback)
        print(f"[LLM] LLM explanation for treatments....:")
        llm_text = _llm_explain_treatments(
            disease_label=disease_label,
            disease_symptoms=disease_symptoms,
            disease_treatments=disease_treatments,
        )
        print(f"[LLM] LLM treatment explanation Output ....:", llm_text)
        if not llm_text:
            # Fallback to deterministic template if LLM not available
            narrative_lines: List[str] = []
            narrative_lines.append(
                f"From the perspective of the knowledge graph, '{disease_label}' is "
                "characterized by a specific set of symptoms and treatments."
            )
            if disease_symptoms:
                narrative_lines.append(
                    f"Symptom-wise, the KG links '{disease_label}' to: "
                    f"{_fmt_list(disease_symptoms)}."
                )
            else:
                narrative_lines.append(
                    "No explicit symptom nodes are connected to this disease in the graph."
                )
            if disease_treatments:
                narrative_lines.append(
                    "For management, the KG records the following treatments: "
                    f"{_fmt_list(disease_treatments)}. "
                    "These represent typical interventions associated with this disease "
                    "in the dataset, not personalized medical recommendations."
                )
            else:
                narrative_lines.append(
                    "The graph does not contain any treatment edges for this disease."
                )
            narrative_lines.append(
                "This explanation is derived purely from KG triples and is NOT "
                "a medical recommendation."
            )
            llm_text = " ".join(narrative_lines)
        
        print(f"[LLM] LLM treatment explanation Done....:")

    # ------------------------------------------------------------------
    # Case B: diseases_for_symptoms with symptoms given
    # (differential-style query: 'I have fever and cough, what could it be?')
    # ------------------------------------------------------------------
    elif intent == "diseases_for_symptoms" and symptoms:
        # Our own robust differential computation
        candidates = _compute_differential_candidates(symptoms, max_results=15)

        # Symbolic explanation
        symbolic_text = _build_symbolic_differential_text(symptoms, candidates)

        # NEW: triples used in reasoning (matching + other symptoms for top diseases)
        triples_used = _triples_for_differential(symptoms, candidates)

        # LLM narrative (true LLM call, with fallback)
        print(f"[LLM] Getting LLM explanation for differential diagnosis....:")
        llm_text = _llm_explain_differential(symptoms, candidates)
        print(f"[LLM] LLM sympytom explanation Output.... ::", llm_text)
        if not llm_text:
            # Fallback to deterministic narrative, robust to dict OR dataclass
            if candidates:
                top = candidates[0]

                def _get(c: Any, key: str, default: Any = None) -> Any:
                    if hasattr(c, key):
                        return getattr(c, key)
                    if isinstance(c, dict):
                        return c.get(key, default)
                    return default

                disease_label = _get(top, "disease_label", "Unknown disease")
                body_system = _get(top, "body_system", "Unknown / not classified")
                num_matches = _get(top, "num_matches", 0)
                total_symptoms = _get(top, "total_symptoms", 0)
                matching_symptoms = list(_get(top, "matching_symptoms", []) or [])
                other_symptoms = list(_get(top, "other_symptoms", []) or [])

                llm_lines: List[str] = []
                llm_lines.append(
                    "Interpreting your symptoms through the lens of the knowledge graph, "
                    "we can rank candidate diseases by how many of your symptoms they share."
                )
                llm_lines.append(
                    f"The top candidate is '{disease_label}' "
                    f"in the '{body_system}' body system, matching "
                    f"{num_matches} out of {total_symptoms} known symptoms."
                )
                if matching_symptoms:
                    llm_lines.append(
                        "In particular, it shares the symptom(s): "
                        f"{_fmt_list(matching_symptoms)}."
                    )
                if other_symptoms:
                    llm_lines.append(
                        "The KG also indicates other typical symptoms for this disease, "
                        f"such as {_fmt_list(other_symptoms)}."
                    )
                if len(candidates) > 1:
                    others = [
                        _get(c, "disease_label", "Unknown disease")
                        for c in candidates[1:4]
                    ]
                    llm_lines.append(
                        "Other possible candidates with some overlap include: "
                        f"{_fmt_list(others)}."
                    )
                llm_lines.append(
                    "This is only a KG-based differential suggestion, not a diagnosis, "
                    "and cannot replace professional medical evaluation."
                )
                llm_text = " ".join(llm_lines)
            else:
                llm_text = (
                    "Given your reported symptoms, the knowledge graph did not find any "
                    "diseases with matching symptom profiles. This may simply reflect "
                    "gaps in the dataset rather than clinical reality. "
                    "No diagnostic conclusion can be drawn from this."
                )
        print(f"[LLM] LLM sympytom explanation Done....:")

    # ------------------------------------------------------------------
    # Case C: disease is present but interpreted as symptom query
    # (e.g., 'What are the symptoms of Common Cold?')
    # ------------------------------------------------------------------
    elif intent == "diseases_for_symptoms" and disease:
        disease_label = disease
        disease_symptoms = get_symptoms_for_disease(disease_label)

        symbolic_text = _build_symbolic_symptoms_text(disease_label, disease_symptoms)

        # NEW: triples used in reasoning (only symptom edges here)
        triples_used = _triples_for_disease_symptoms_and_treatments(
            disease_label,
            disease_symptoms,
            [],  # no treatments in this query
        )
        # LLM narrative (true LLM call, with fallback)
        print(f"[LLM] LLM explanation for symptoms....:")
        llm_text = _llm_explain_symptoms(disease_label, disease_symptoms)
        print(f"[LLM] LLM explanation Output.... ::", llm_text)
        if not llm_text:
            # Fallback to simple narrative
            if disease_symptoms:
                llm_text = (
                    f"In the medical knowledge graph, '{disease_label}' is described "
                    "by a characteristic set of symptoms. Typical symptoms recorded "
                    f"for this disease include: {_fmt_list(disease_symptoms)}. "
                    "This description is entirely KG-based and does not constitute "
                    "personalized medical guidance."
                )
            else:
                llm_text = (
                    f"In the medical knowledge graph, '{disease_label}' currently has "
                    "no explicit symptom links recorded. This may reflect limitations "
                    "of the dataset rather than clinical reality. "
                    "This is not medical advice."
                )
        print(f"[LLM] LLM explanation Done....:")

    # ------------------------------------------------------------------
    # Fallback: unknown / unsupported intent
    # ------------------------------------------------------------------
    else:
        symbolic_text = (
            "I could not confidently interpret your question in terms of the supported "
            "intents (symptom-based differential or treatments for a disease) using "
            "the large medical knowledge graph."
        )
        llm_text = symbolic_text
        # triples_used stays empty list; candidates stays []

    return {
        "symbolic_text": symbolic_text,
        "llm_text": llm_text,
        "intent": intent,
        "slots": slots,
        "triples": triples_used,   # 👈 for Block 3 in UI
        "candidates": candidates,  # 👈 for debug panel
        "mode": "large-kg",
        # optional: you can later add "raw_llm_slots" here if you extend the parser
    }
