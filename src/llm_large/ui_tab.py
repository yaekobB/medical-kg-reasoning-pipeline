# src/assistant/ui_tab.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from .explanations import answer_question_with_large_kg
from .kg_api_large import (
    get_all_disease_labels,
    get_all_symptom_labels,
)


def _render_examples() -> None:
    """Small helper: show example questions in an expander."""
    with st.expander("Example questions you can try", expanded=False):
        st.markdown(
            """
- **Treatments for a disease**
  - `What are the treatments for Pneumonia?`
  - `How is Tuberculosis treated according to your knowledge graph?`

- **Differential diagnosis from symptoms**
  - `I have fever and cough, what could it be?`
  - `I have chest pain and difficulty breathing, which diseases might match?`

- **Symptoms of a disease**
  - `What are the symptoms of Common Cold?`
  - `Give me information about Pneumonia`
            """.strip()
        )


def _render_kg_summary() -> None:
    """Optional: small sidebar/box showing how big the KG is."""
    try:
        # We only use the labels here; KG is cached inside kg_api_large
        diseases = get_all_disease_labels()
        symptoms = get_all_symptom_labels()
        st.info(
            f"**Large KG summary**  \n"
            f"- Diseases: `{len(diseases)}`  \n"
            f"- Symptoms: `{len(symptoms)}`  \n"
            f"- Back-end graph: `data_medical_large_inferred.ttl`",
            icon="ℹ️",
        )
    except Exception as exc:  # pragma: no cover - purely cosmetic
        st.warning(f"Could not load KG summary: {exc}")


def render_llm_kg_reasoner_large() -> None:
    """
    Render the 'LLM-assisted KG Reasoner (Large)' tab in Streamlit.

    This does **not** alter any pipeline logic. It just:
      - takes a natural language question from the user,
      - calls `answer_question_with_large_kg`,
      - shows symbolic KG explanation + LLM explanation,
      - optionally shows debug info (slots, candidates),
      - and (NEW) shows the concrete KG triples used as evidence.
    """
    st.header("LLM-assisted KG Reasoner – Large Medical KG")

    st.markdown(
        """
This tab demonstrates how a **large medical knowledge graph (KG)** and a
**local LLM** can work together:

1. The **KG** does symbolic reasoning:
   - finds diseases that match given symptoms, or  
   - retrieves symptoms & treatments for a disease.

2. The **LLM** then **explains** *why* those results appear, using only the
   information extracted from the KG.

3. A third block shows the **exact KG triples used as evidence**:
   - concrete `(subject, predicate, object)` edges (e.g. `Pneumonia --med:hasSymptom--> cough`)
   - so you can see precisely which facts the system relied on for its answer.

⚠️ **Important:** everything here is based only on the KG content and is
for **educational** and **analytical** purposes. It is *not* a medical diagnosis
or recommendation.
        """.strip()
    )


    _render_kg_summary()
    _render_examples()

    st.markdown("---")

    question = st.text_area(
        "Ask a question about symptoms or treatments:",
        value="What are the treatments for Pneumonia?",
        height=90,
    )

    col_btn, col_clear = st.columns([1, 1])
    run_clicked = col_btn.button("Run KG + LLM explanation", type="primary")
    clear_clicked = col_clear.button("Clear")

    if clear_clicked:
        # Just reset the text area by rerun – Streamlit will re-render.
        st.rerun()

    if not run_clicked:
        return

    if not question.strip():
        st.warning("Please type a question first.")
        return

    with st.spinner("Querying the knowledge graph and generating explanation..."):
        result: Dict[str, Any] = answer_question_with_large_kg(question.strip())

    symbolic_text: str = (result.get("symbolic_text") or "").strip()
    llm_text: str = (result.get("llm_text") or "").strip()
    slots: Dict[str, Any] = result.get("slots", {})
    raw_llm_slots: Any = result.get("raw_llm_slots")
    candidates: List[Dict[str, Any]] = result.get("candidates", [])
    mode: str = result.get("mode", "unknown")
    # NEW: concrete triples used as evidence
    triples: List[Dict[str, Any]] = result.get("triples", []) or []

    # ------------------------------------------------------------------
    # Block 1 – symbolic KG answer
    # ------------------------------------------------------------------
    st.markdown("### 1. Symbolic KG answer")
    if symbolic_text:
        st.markdown(symbolic_text)
    else:
        st.info("No symbolic explanation was produced for this question.")

    # ------------------------------------------------------------------
    # Block 2 – LLM explanation
    # ------------------------------------------------------------------
    st.markdown("### 2. LLM explanation (based on KG triples)")
    if llm_text:
        st.markdown(llm_text)
    else:
        st.info("No LLM explanation was produced (likely a safety refusal).")

    # ------------------------------------------------------------------
    # Block 3 – KG triples used in reasoning (evidence)
    # ------------------------------------------------------------------
    st.markdown("### 3. KG triples used in reasoning (evidence)")
    st.caption(
        "These are the concrete subject–predicate–object triples from the large "
        "medical KG that were actually used to support the answer above. "
        "They show exactly **which facts** the system relied on."
    )

    if triples:
        # Show only the first N triples to avoid a huge table in the UI
        MAX_SHOW = 80
        shown_triples = triples[:MAX_SHOW]

        # Normalise keys so the table is stable even if some keys are missing
        table_rows: List[Dict[str, Any]] = []
        for t in shown_triples:
            table_rows.append(
                {
                    "role": t.get("role", ""),         # e.g. "symptom-edge", "treatment-edge"
                    "subject": t.get("subject", ""),   # e.g. "Pneumonia"
                    "predicate": t.get("predicate", ""),  # e.g. "hasSymptom", "treatedWith"
                    "object": t.get("object", ""),     # e.g. "fever", "antibiotics"
                }
            )

        st.table(table_rows)

        if len(triples) > MAX_SHOW:
            st.info(
                f"Showing the first {MAX_SHOW} triples out of {len(triples)} total "
                "used for this answer."
            )

        with st.expander("Raw triple list (JSON view)", expanded=False):
            st.json(triples)
    else:
        st.info(
            "No explicit KG triples were recorded for this question. "
            "This usually happens only for unsupported or fallback intents."
        )

    # ------------------------------------------------------------------
    # Debug view (unchanged)
    # ------------------------------------------------------------------
    with st.expander("Debug view – interpreted intent, slots, and candidates", expanded=False):
        st.markdown("**Interpreted slots** (after mapping to KG labels):")
        st.json(
            {
                "mode": mode,
                "slots": slots,
            }
        )

        st.markdown("**Raw LLM slot output** (before post-processing):")
        st.code(str(raw_llm_slots), language="json")

        st.markdown("**Top candidate diseases (from KG):**")
        # Convert DiseaseMatch objects or dicts to plain dicts for JSON display
        serialisable_candidates = []
        for c in candidates:
            # It might be a dataclass or a plain dict; handle both.
            if hasattr(c, "__dict__"):
                serialisable_candidates.append({k: v for k, v in c.__dict__.items()})
            else:
                serialisable_candidates.append(c)
        st.json(serialisable_candidates)
