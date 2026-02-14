import streamlit as st
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
import re
from typing import Any, Dict, Optional

from rdflib.query import Result as SPARQLResult  # rdflib's SPARQL Result typ

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # from src/ui_prototypes to project root
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CSV_PATH = PROCESSED_DIR / "diseases_large.csv"
TTL_RAW_PATH = PROCESSED_DIR / "data_medical_large.ttl"
TTL_INFERRED_PATH = PROCESSED_DIR / "data_medical_large_inferred.ttl"
OWL_RL_TTL_PATH = PROCESSED_DIR / "data_medical_large_owlrl_inferred.ttl"


MED = Namespace("http://example.org/medkg#")

from typing import List, Optional

from src.sparql.sparql_queries_llm import (
    load_graph_for_llm,
    kg_differential_diagnosis,
    kg_explain_disease,
)
from src.llm_simple.ollama_integration import (
    llm_explain_differential,
    llm_explain_disease,
)

from src.llm_large.ui_tab import render_llm_kg_reasoner_large
from src.sparql.sparql_ui_demo_queries import DEMO_QUERY_BUILDERS



# ------------------------------
# Caching loaders
# ------------------------------
@st.cache_resource
def load_graph_raw() -> Graph:
    g = Graph()
    g.parse(TTL_RAW_PATH)
    return g


@st.cache_resource
def load_graph_inferred() -> Graph:
    g = Graph()
    g.parse(TTL_INFERRED_PATH)
    return g

@st.cache_resource
def load_graph_owlrl(ttl_path) -> Graph | None:
    ttl_path = Path(ttl_path)
    if not ttl_path.exists():
        return None
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    return g

@st.cache_resource
def load_diseases_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def get_llm_graph():
    """Load the inferred KG once and reuse it across LLM calls."""
    return load_graph_for_llm()

# ------------------------------        
# Simple intent detection
# -----------------------------

class IntentResult:
    def __init__(
        self,
        intent: str,
        symptoms: Optional[List[str]] = None,
        disease_label: Optional[str] = None,
        raw: str = "",
    ):
        self.intent = intent               # "diff_diagnosis" | "explain_disease" | "unknown"
        self.symptoms = symptoms or []
        self.disease_label = disease_label
        self.raw = raw


def parse_user_question(question: str) -> IntentResult:
    """
    Very simple, rule-based intent detection.

    Supported for now:
      - Differential diagnosis from symptoms:
          "fever, cough, headache"
          "Patient has fever and cough, what could it be?"
      - Disease explanation:
          "Explain Pneumonia"
          "What is Tuberculosis?"
          "What are the symptoms of pneumonia?"
    """
    q = (question or "").strip()
    q_low = q.lower()

    if not q:
        return IntentResult(intent="unknown", raw=question)

    # --------------------------------------------------
    # 1) Disease explanation intent
    #    (including "what are the symptoms of X?")
    # --------------------------------------------------
    explanation_patterns = [
        "explain ",
        "what is ",
        "what's ",
        "tell me about ",
        "info about ",
        "information about ",
        "what are the symptoms of ",
        "what are the main symptoms of ",
        "symptoms of ",
    ]

    for pat in explanation_patterns:
        if pat in q_low:
            idx = q_low.find(pat)
            after = q_low[idx + len(pat):]

            # Clean trailing punctuation
            after = after.replace("?", "").replace(".", "")

            # Special case: if pattern contains "symptoms of", strip leading "of"
            if after.startswith("of "):
                after = after[3:]

            # Remove leading articles ("a pneumonia" -> "pneumonia")
            for art in ("a ", "an ", "the "):
                if after.startswith(art):
                    after = after[len(art):]

            disease_candidate = after.strip(" :,-")

            if disease_candidate:
                disease_label = " ".join(
                    w.capitalize() for w in disease_candidate.split()
                )
                return IntentResult(
                    intent="explain_disease",
                    disease_label=disease_label,
                    raw=question,
                )

    # --------------------------------------------------
    # 2) Differential diagnosis intent (symptoms)
    # --------------------------------------------------
    # Heuristic: symptom-style questions with commas / "and"
    if ("," in q or " and " in q_low or "symptom" in q_low):
        tmp = q_low

        # Remove some noisy phrases
        noise_phrases = [
            "patient has",
            "patient have",
            "patient is",
            "has",
            "have",
            "having",
            "with",
            "symptoms of",
            "symptoms are",
            "symptom of",
            "symptom is",
            "symptoms include",
            "what could it be",
            "what could this be",
            "what diseases could it be",
            "what disease could it be",
        ]
        for phrase in noise_phrases:
            tmp = tmp.replace(phrase, " ")

        # Normalize "and" to commas
        tmp = tmp.replace(" and ", ",")

        parts = [p.strip() for p in tmp.split(",")]

        # Filter out obviously non-symptom fragments
        raw_symptoms = [
            p for p in parts
            if p and 1 < len(p) <= 40 and not p.endswith("?")
        ]

        # Remove common question words from each fragment
        blacklist = {"what", "could", "this", "be", "the", "a", "an"}
        symptoms: list[str] = []
        for frag in raw_symptoms:
            tokens = [t for t in frag.split() if t not in blacklist]
            if tokens:
                symptoms.append(" ".join(tokens))

        if symptoms:
            return IntentResult(
                intent="diff_diagnosis",
                symptoms=symptoms,
                raw=question,
            )

    # --------------------------------------------------
    # 3) Fallback: short comma-separated list -> symptoms
    # --------------------------------------------------
    if "," in q and len(q) < 80:
        symptoms = [s.strip().lower() for s in q.split(",") if s.strip()]
        if symptoms:
            return IntentResult(
                intent="diff_diagnosis",
                symptoms=symptoms,
                raw=question,
            )

    # --------------------------------------------------
    # 4) Unknown intent
    # --------------------------------------------------
    return IntentResult(intent="unknown", raw=question)

def sparql_escape_literal(text: str) -> str:
    """
    Escape a Python string so it is safe inside a SPARQL double-quoted literal "...".
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\\", "\\\\")   # escape backslashes first
    s = s.replace('"', '\\"')     # escape quotes
    s = s.replace("\n", " ").replace("\r", " ")  # avoid breaking the query
    return s

# ------------------------------
# Helper: symptom-based ranking
# ------------------------------
def rank_diseases_by_symptoms(graph: Graph, symptom_terms, limit=50):
    """
    Very simple ranking:
    - symptom_terms: list of lowercase strings ["fever", "cough"]
    - For each disease, count how many of these substrings appear in its symptom labels.
    - Also collect all matching symptom labels and treatments.
    """
    symptom_terms = [t.strip().lower() for t in symptom_terms if t.strip()]
    if not symptom_terms:
        return []

    # key -> {matched_terms, symptoms, treatments}
    matches = {}

    for term in symptom_terms:
        q = f"""
        PREFIX med: <{MED}>
        PREFIX rdfs: <{RDFS}>

        SELECT DISTINCT ?d ?d_label ?sys_label ?sym_label ?t_label
        WHERE {{
          ?d a med:Disease ;
             rdfs:label ?d_label ;
             med:hasSymptom ?s ;
             med:affectsSystem ?sys .
          ?sys rdfs:label ?sys_label .
          ?s a med:Symptom ;
             rdfs:label ?sym_label .

          OPTIONAL {{
            ?d med:treatedWith ?t .
            ?t rdfs:label ?t_label .
          }}

          FILTER(CONTAINS(LCASE(STR(?sym_label)), "{term}"))
        }}
        """
        # Execute query and process results
        for row in graph.query(q):
            d_uri, d_label, sys_label, sym_label, t_label = row
            key = (str(d_uri), str(d_label), str(sys_label))
            if key not in matches:
                matches[key] = {
                    "matched_terms": set(),
                    "symptoms": set(),
                    "treatments": set(),
                }
            matches[key]["matched_terms"].add(term)
            matches[key]["symptoms"].add(str(sym_label))
            if t_label is not None:
                matches[key]["treatments"].add(str(t_label))

    results = []
    for (d_uri, d_label, sys_label), info in matches.items():
        results.append({
            "disease_uri": d_uri,
            "disease": d_label,
            "body_system": sys_label,
            "num_matching_symptoms": len(info["matched_terms"]),
            "matched_symptoms": ", ".join(sorted(info["symptoms"])),
            "treatments": ", ".join(sorted(info["treatments"])) if info["treatments"] else "",
        })

    results = sorted(results, key=lambda x: x["num_matching_symptoms"], reverse=True)
    return results[:limit]


# ------------------------------
# Body-system helpers
# ------------------------------
def get_all_body_system_labels(graph: Graph):
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT DISTINCT ?sys_label
    WHERE {{
      ?sys a med:BodySystem ;
           rdfs:label ?sys_label .
    }}
    ORDER BY ?sys_label
    """
    return [str(r[0]) for r in graph.query(q)]


def get_diseases_by_body_system(graph: Graph, system_label: str):
    
    safe_label = sparql_escape_literal(system_label)
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT DISTINCT ?d ?d_label ?sys_label
    WHERE {{
      ?d a med:Disease ;
         rdfs:label ?d_label ;
         med:affectsSystem ?sys .
      ?sys rdfs:label ?sys_label .
      FILTER(STR(?sys_label) = "{safe_label}")
    }}
    ORDER BY ?d_label
    """
    rows = list(graph.query(q))
    return [
        {"disease_uri": str(r[0]), "disease": str(r[1]), "body_system": str(r[2])}
        for r in rows
    ]

# ------------------------------
# Disease details (for detail panel)
# ------------------------------
def get_disease_details(graph: Graph, disease_label: str):
    
    safe_disease = sparql_escape_literal(disease_label)
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?d ?sys_label ?sym_label ?t_label
    WHERE {{
      ?d a med:Disease ;
         rdfs:label ?d_label .
      FILTER(STR(?d_label) = "{safe_disease}")
      OPTIONAL {{
        ?d med:affectsSystem ?sys .
        ?sys rdfs:label ?sys_label .
      }}
      OPTIONAL {{
        ?d med:hasSymptom ?s .
        ?s rdfs:label ?sym_label .
      }}
      OPTIONAL {{
        ?d med:treatedWith ?t .
        ?t rdfs:label ?t_label .
      }}
    }}
    """
    rows = list(graph.query(q))
    if not rows:
        return None

    d_uri = str(rows[0][0])
    system = rows[0][1]
    system_label = str(system) if system is not None else "Unknown / not classified"

    symptoms = sorted({str(r[2]) for r in rows if r[2] is not None})
    treatments = sorted({str(r[3]) for r in rows if r[3] is not None})

    return {
        "disease_uri": d_uri,
        "disease": disease_label,
        "body_system": system_label,
        "symptoms": symptoms,
        "treatments": treatments,
    }

# ------------------------------
# Extra helpers for Disease/Treatment/SPARQL tabs
# ------------------------------
from rdflib.plugins.sparql.processor import SPARQLResult

def get_all_disease_labels(graph: Graph) -> list[str]:
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>
    SELECT DISTINCT ?label WHERE {{
      ?d a med:Disease ;
         rdfs:label ?label .
    }}
    ORDER BY ?label
    """
    return [str(r[0]) for r in graph.query(q)]

def get_all_treatment_labels(graph: Graph) -> list[str]:
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>
    SELECT DISTINCT ?label WHERE {{
      ?t a med:Treatment ;
         rdfs:label ?label .
    }}
    ORDER BY ?label
    """
    return [str(r[0]) for r in graph.query(q)]

def get_diseases_by_treatment(graph: Graph, treatment_label: str) -> list[dict]:
    
    safe_treatment = sparql_escape_literal(treatment_label)
    q = f"""
    PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT DISTINCT ?d ?d_label ?sys_label ?t_label
    WHERE {{
      ?d a med:Disease ;
         rdfs:label ?d_label ;
         med:treatedWith ?t .
      ?t rdfs:label ?t_label .
      OPTIONAL {{
        ?d med:affectsSystem ?sys .
        ?sys rdfs:label ?sys_label .
      }}
      FILTER(STR(?t_label) = "{safe_treatment}")
    }}
    ORDER BY ?d_label
    """
    rows = list(graph.query(q))
    out = []
    for r in rows:
        d_uri = str(r[0])
        d_label = str(r[1])
        sys_label = str(r[2]) if r[2] is not None else "Unknown / not classified"
        t_label = str(r[3])
        out.append({
            "disease_uri": d_uri,
            "disease": d_label,
            "body_system": sys_label,
            "treatment": t_label,
        })
    return out

def _safe_merge_with_csv(sys_df: pd.DataFrame, df_csv: pd.DataFrame) -> pd.DataFrame:
    """
    Join KG rows to CSV metadata safely:
    KG 'disease' label -> CSV 'disease_name'
    """
    if sys_df.empty:
        return sys_df.copy()

    if "disease_name" in df_csv.columns and "disease" in sys_df.columns:
        joined = sys_df.merge(
            df_csv,
            left_on="disease",
            right_on="disease_name",
            how="left",
            suffixes=("", "_csv"),
        )
        return joined
    return sys_df.copy()

# -----------------------------
# Query form detection helpers
# -----------------------------
_FORM_RE = re.compile(
    r"""(?isx)
    ^\s*
    (?:\#.*\n\s*)*                                   # leading comments
    (?:(?:BASE\s+<[^>]*>\s*)|(?:PREFIX\s+\w*:\s*<[^>]*>\s*))*  # BASE/PREFIX lines
    (?:\#.*\n\s*)*
    (SELECT|ASK|CONSTRUCT|DESCRIBE)\b
    """
)

# Block SPARQL Update / dataset-mutation keywords (UI safety)
_UPDATE_RE = re.compile(
    r"""(?isx)
    \b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY)\b
    """
)

def _query_form(q: str) -> Optional[str]:
    m = _FORM_RE.match(q or "")
    return m.group(1).upper() if m else None

def _is_safe_readonly(q: str) -> bool:
    # Reject SPARQL Update patterns
    return not bool(_UPDATE_RE.search(q or ""))


# -----------------------------
# Main runner (multi-form)
# -----------------------------
def run_sparql_query(g: Graph, query: str, limit_rows: int = 200) -> Dict[str, Any]:
    """
    Run SPARQL and return a UI-friendly dict:
      - SELECT    -> {"form":"SELECT", "df": DataFrame}
      - ASK       -> {"form":"ASK", "ask": bool}
      - CONSTRUCT -> {"form":"CONSTRUCT", "triples_df": DataFrame, "ttl": str}
      - DESCRIBE  -> {"form":"DESCRIBE", "triples_df": DataFrame, "ttl": str}

    limit_rows limits displayed rows/triples (not the engine execution).
    """
    q = (query or "").strip()
    if not q:
        return {"form": "EMPTY"}

    form = _query_form(q)
    if form is None:
        raise ValueError("Unsupported query. Use SELECT / ASK / CONSTRUCT / DESCRIBE (PREFIX/BASE allowed).")

    if not _is_safe_readonly(q):
        raise ValueError("SPARQL Update queries are disabled in this UI (INSERT/DELETE/LOAD/CLEAR/DROP/...).")

    res = g.query(q)

    # ---- SELECT ----
    if form == "SELECT":
        if not isinstance(res, SPARQLResult):
            # rdflib should return SPARQLResult here; keep defensive
            raise ValueError("Unexpected result type for SELECT query.")

        vars_ = [str(v) for v in res.vars]
        rows = []
        for i, row in enumerate(res):
            if i >= limit_rows:
                break
            # row is tuple aligned with vars
            rows.append([str(row[j]) if row[j] is not None else "" for j in range(len(vars_))])

        return {"form": "SELECT", "df": pd.DataFrame(rows, columns=vars_)}

    # ---- ASK ----
    if form == "ASK":
        # rdflib returns SPARQLResult with .askAnswer
        if isinstance(res, SPARQLResult) and hasattr(res, "askAnswer"):
            return {"form": "ASK", "ask": bool(res.askAnswer)}
        # Fallback (defensive)
        try:
            return {"form": "ASK", "ask": bool(res)}
        except Exception:
            raise ValueError("Unexpected result type for ASK query.")

    # ---- CONSTRUCT / DESCRIBE ----
    # rdflib often returns a Graph directly for CONSTRUCT/DESCRIBE
    out_graph = None
    if isinstance(res, Graph):
        out_graph = res
    elif isinstance(res, SPARQLResult) and hasattr(res, "graph") and isinstance(res.graph, Graph):
        out_graph = res.graph

    if out_graph is None:
        raise ValueError(f"Unexpected result type for {form} query (expected an RDF graph).")

    # Convert graph -> triples dataframe (limited)
    triples = []
    for i, (s, p, o) in enumerate(out_graph.triples((None, None, None))):
        if i >= limit_rows:
            break
        triples.append([str(s), str(p), str(o)])

    triples_df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
    ttl = out_graph.serialize(format="turtle")

    return {"form": form, "triples_df": triples_df, "ttl": ttl}

# ------------------------------
# KG + LLM Assistant UI
# ------------------------------

def render_llm_tab():
    st.header("KG + LLM Assistant (prototype)")

    st.markdown(
        """
This tab lets you **ask questions in natural language**, then:

1. We detect your intent (rule-based).
2. We query the **RDF knowledge graph (KG)**.
3. We call the **local LLM via Ollama** to generate a friendly explanation.

**Supported for now:**

- **Differential diagnosis from symptoms**  
  Examples:  
  - `fever, cough, headache`  
  - `Patient has fever and cough, what could it be?`  

- **Disease explanation**  
  Examples:  
  - `Explain Pneumonia`  
  - `What is Tuberculosis?`
        """
    )

    default_q = "Patient has fever, cough, headache. What diseases could it be?"
    question = st.text_area("Your question", value=default_q, height=120)

    if st.button("Ask assistant"):
        if not question.strip():
            st.warning("Please type a question first.")
            return
        print(f"[User] Question: {question}")
        intent_res = parse_user_question(question)
        print(f"[Intent] Parsed intent: {intent_res.intent}, symptoms={intent_res.symptoms}, disease_label={intent_res.disease_label}")
        if intent_res.intent == "unknown":
            st.error("Sorry, I can't understand this kind of question yet.")
            st.info(
                "Try for example:\n"
                "- `fever, cough, headache`\n"
                "- `Explain Pneumonia`\n"
                "- `What is Tuberculosis?`"
            )
            return

        # Load KG once (cached)
        g = get_llm_graph()

        # ---------------------------------------------------------
        # Intent: differential diagnosis from symptoms
        # ---------------------------------------------------------
        if intent_res.intent == "diff_diagnosis":
            symptoms = [s.lower() for s in intent_res.symptoms]
            st.write("**Detected intent:** differential diagnosis")
            st.write(f"**Parsed symptoms:** `{symptoms}`")

            candidates = kg_differential_diagnosis(g, symptoms, limit=10)

            if not candidates:
                st.warning("No diseases found in the KG for these symptoms.")
                return

            # Show KG candidates as a small table
            df = pd.DataFrame(
                [
                    {
                        "Disease": c.disease_label,
                        "Body system": c.system_label,
                        "#Matching symptoms": len(c.matched_symptoms),
                        "Matching symptoms": ", ".join(sorted(c.matched_symptoms)),
                    }
                    for c in candidates
                ]
            )

            st.subheader("KG candidate diseases")
            st.dataframe(df)

            # Ask LLM for explanation
            st.subheader("LLM explanation")

            try:
                answer = llm_explain_differential(symptoms, candidates)
                st.markdown(answer)
            except Exception as e:
                st.error(f"LLM backend error: {e}")
                st.info("Is Ollama running? Try `ollama serve` in another terminal.")

        # ---------------------------------------------------------
        # Intent: explain a disease
        # ---------------------------------------------------------
        elif intent_res.intent == "explain_disease":
            st.write("**Detected intent:** disease explanation")
            st.write(f"**Disease label:** `{intent_res.disease_label}`")
            print(f"[LLM Tab] Explaining disease: {intent_res.disease_label}")
            expl = kg_explain_disease(g, intent_res.disease_label)
            
            print(f"[LLM Tab] KG explanation retrieved: {expl is not None}")

            if expl is None:
                st.warning(
                    f"No disease found in the KG with label '{intent_res.disease_label}'."
                )
                return

            # Show raw KG facts
            st.subheader("KG facts")

            st.write(f"**URI**: `{expl.disease_uri}`")
            st.write(
                "**Body system(s)**: "
                + ", ".join(expl.system_labels or ["Unknown / not classified"])
            )
            st.write("**Type(s)**: " + ", ".join(expl.type_labels or ["med:Disease"]))
            st.write("**Symptoms**: " + ", ".join(expl.symptom_labels or ["(none)"]))

            # Ask LLM for explanation
            st.subheader("LLM explanation")

            try:
                answer = llm_explain_disease(expl)
                print(f"[LLM Tab] LLM explanation Done.\n{answer}")
                st.markdown(answer)
            except Exception as e:
                st.error(f"LLM backend error: {e}")
                st.info("Is Ollama running? Try `ollama serve` in another terminal.")



# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Medical Knowledge-Graph Reasoning - Medium UI",
        layout="wide",
    )
    st.title("Medical Knowledge-Graph Reasoning – Dashboard ")

    g_raw = load_graph_raw() # Load KG once
    g_inf = load_graph_inferred() # load infered KG once
    g_owl = load_graph_owlrl(OWL_RL_TTL_PATH) # load OWL RL KG once

    df = load_diseases_df() # Load diseases CSV once
    

    tab_overview, tab_symptom, tab_body, tab_disease, tab_treatment, tab_sparql, tab_vis, tab_llm, tab_llm_large, tab_limits = st.tabs(
    [
        "Overview",
        "Symptom Explorer",
        "BodySystem Explorer",
        "Disease Explorer",
        "Treatment Explorer",
        "SPARQL Playground",
        "Visualizations",
        "LLM-assisted(Simple)",
        "LLM-assisted(Large)",
        "RDFS Limitations",
    ]
)

    # --------------------------
    # Tab 1: Overview
    # --------------------------
    with tab_overview:
        st.subheader("Project Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Diseases (CSV)",
                df["disease_id"].nunique() if "disease_id" in df.columns else len(df),
            )
        with col2:
            st.metric("Rows in processed CSV", len(df))
        with col3:
            st.metric("Triples in RAW KG", len(g_raw))
        with col4:
            st.metric("Triples in inferred KG", len(g_inf))

        # --------------------------
        # SHACL reports
        # --------------------------
        st.markdown("### Data quality (SHACL)")
        with st.expander("Show SHACL validation reports"):
            rep_raw_txt = PROCESSED_DIR / "shacl_report_data_medical_large.txt"
            rep_inf_txt = PROCESSED_DIR / "shacl_report_data_medical_large_inferred.txt"

            if rep_raw_txt.exists():
                st.markdown("**RAW KG report (text):**")
                st.code(rep_raw_txt.read_text(encoding="utf-8")[:5000])
            else:
                st.info("RAW SHACL report not found yet.")

            if rep_inf_txt.exists():
                st.markdown("**INFERRED KG report (text):**")
                st.code(rep_inf_txt.read_text(encoding="utf-8")[:5000])
            else:
                st.info("INFERRED SHACL report not found yet.")

        # --------------------------
        # NEW: KG type summary (RAW vs INFERRED)
        # --------------------------
        st.markdown("### KG summary (RAW vs INFERRED) — reasoning impact")

        from rdflib import Namespace
        from rdflib.namespace import RDF

        MEDNS = Namespace(MED)

        def _instances_of(g, cls_uri):
            return set(g.subjects(RDF.type, cls_uri))

        def _count(g, cls_uri):
            return len(_instances_of(g, cls_uri))

        # base sets
        raw_disease = _instances_of(g_raw, MEDNS.Disease)
        raw_chronic = _instances_of(g_raw, MEDNS.ChronicDisease)
        raw_infect  = _instances_of(g_raw, MEDNS.InfectiousDisease)

        inf_disease = _instances_of(g_inf, MEDNS.Disease)
        inf_chronic = _instances_of(g_inf, MEDNS.ChronicDisease)
        inf_infect  = _instances_of(g_inf, MEDNS.InfectiousDisease)

        # subset checks
        raw_chronic_and_disease = raw_chronic & raw_disease
        raw_chronic_not_disease = raw_chronic - raw_disease
        raw_infect_and_disease  = raw_infect & raw_disease
        raw_infect_not_disease  = raw_infect - raw_disease

        inf_chronic_and_disease = inf_chronic & inf_disease
        inf_chronic_not_disease = inf_chronic - inf_disease
        inf_infect_and_disease  = inf_infect & inf_disease
        inf_infect_not_disease  = inf_infect - inf_disease

        summary_rows = [
            ("Total med:Disease",            len(raw_disease), len(inf_disease)),
            ("Total med:ChronicDisease",     len(raw_chronic), len(inf_chronic)),
            ("Total med:InfectiousDisease",  len(raw_infect),  len(inf_infect)),

            ("Chronic ∧ Disease",            len(raw_chronic_and_disease), len(inf_chronic_and_disease)),
            ("Chronic ∧ ¬Disease",           len(raw_chronic_not_disease), len(inf_chronic_not_disease)),

            ("Infectious ∧ Disease",         len(raw_infect_and_disease),  len(inf_infect_and_disease)),
            ("Infectious ∧ ¬Disease",        len(raw_infect_not_disease),  len(inf_infect_not_disease)),
        ]

        summary_df = pd.DataFrame(summary_rows, columns=["Row", "RAW", "INFERRED"])
        summary_df["Δ (INFERRED-RAW)"] = summary_df["INFERRED"] - summary_df["RAW"]

        st.dataframe(summary_df, width="stretch")

        st.caption(
            "Interpretation: in RAW, many individuals are typed only as ChronicDisease/InfectiousDisease. "
            "After RDFS reasoning, those become med:Disease too (because ChronicDisease ⊑ Disease, InfectiousDisease ⊑ Disease)."
        )

        # --------------------------
        # CSV dataset summary
        # --------------------------
        st.markdown("### Dataset summary (from CSV)")
        st.write("**Counts by category (Chronic / Infectious / Disease)**")

        if "category" in df.columns:
            st.dataframe(
                df["category"]
                .value_counts()
                .rename_axis("category")
                .reset_index(name="count"),
                width="stretch",
            )
        else:
            st.info("Column 'category' not found in CSV (no category stats).")

        st.markdown("### Body system distribution (from CSV)")
        if "body_system" in df.columns:
            body_counts = (
                df["body_system"]
                .value_counts()
                .rename_axis("body_system")
                .reset_index(name="count")
            )
            st.dataframe(body_counts, width="stretch")
            st.bar_chart(body_counts.set_index("body_system"), width="stretch")
        else:
            st.info("Column 'body_system' not found in CSV.")


    # --------------------------
    # Tab 2: Symptom Explorer
    # --------------------------
    with tab_symptom:
        st.subheader("Symptom-based Disease Explorer")

        st.write(
            "Enter symptom keywords (e.g., `fever, cough, headache`). "
            "We search symptom labels in the KG and rank diseases by how many of the symptoms they match."
        )

        sym_input = st.text_input(
            "Symptoms (comma-separated):", value="fever, cough, headache"
        )
        min_matches = st.slider("Minimum matching symptoms", 1, 3, 1)
        max_results = st.slider("Max results", 10, 100, 30)

        kg_choice = st.radio(
            "KG version",
            ["Reasoned (after RDFS closure)", "Raw (no reasoning)"],
            horizontal=True,
        )
        g_used = g_inf if kg_choice.startswith("Reasoned") else g_raw

        expected_cols = [
            "disease",
            "body_system",
            "num_matching_symptoms",
            "matched_symptoms",
            "treatments",
        ]

        # Persist results across reruns (so dropdown works after button click)
        if "symptom_res_df" not in st.session_state:
            st.session_state.symptom_res_df = None
        if "symptom_kg_choice" not in st.session_state:
            st.session_state.symptom_kg_choice = None
        if "symptom_min_matches" not in st.session_state:
            st.session_state.symptom_min_matches = None

        # Action: compute results only when button is clicked
        if st.button("Find candidate diseases"):
            terms = [s.strip() for s in sym_input.split(",") if s.strip()]
            results = rank_diseases_by_symptoms(g_used, terms, limit=max_results)

            if not results:
                st.session_state.symptom_res_df = pd.DataFrame(columns=expected_cols)
            else:
                filtered = [r for r in results if r["num_matching_symptoms"] >= min_matches]
                st.session_state.symptom_res_df = pd.DataFrame(filtered).reindex(columns=expected_cols)

            st.session_state.symptom_kg_choice = kg_choice
            st.session_state.symptom_min_matches = min_matches

        # Render results (outside the button block) so selection changes refresh correctly
        res_df = st.session_state.symptom_res_df

        if res_df is not None:
            # Informational message consistent with your original intention
            st.write(
                f"Using **{st.session_state.symptom_kg_choice}** – "
                f"found {len(res_df)} diseases with ≥ {st.session_state.symptom_min_matches} matching symptoms."
            )

            if res_df.empty:
                st.warning(
                    "No diseases satisfied the minimum matching symptoms threshold. "
                    "Try lowering 'Minimum matching symptoms' or changing the symptoms."
                )
                st.dataframe(pd.DataFrame(columns=expected_cols), width="stretch")
            else:
                st.dataframe(res_df, width="stretch")

                # Download button
                csv_data = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    csv_data,
                    "candidate_diseases.csv",
                    "text/csv",
                )

                # Disease details panel
                st.markdown("### Disease details")
                disease_names = res_df["disease"].dropna().tolist()

                selected_disease = st.selectbox(
                    "Select a disease to inspect",
                    disease_names,
                    key="symptom_selected_disease",  # stable key fixes “stuck on first”
                )

                if selected_disease:
                    # Use reasoned KG for richer details (same as your original intention)
                    details = get_disease_details(g_inf, selected_disease)
                    if not details:
                        st.warning("Could not find details for this disease in the KG.")
                    else:
                        st.markdown(f"#### {details['disease']}")
                        st.write(f"Body system: **{details['body_system']}**")

                        col_s, col_t = st.columns(2)
                        with col_s:
                            st.markdown("**Symptoms**")
                            st.write(details["symptoms"] if details["symptoms"] else "_No symptoms recorded in KG._")
                        with col_t:
                            st.markdown("**Treatments**")
                            st.write(details["treatments"] if details["treatments"] else "_No treatments recorded in KG._")

    # --------------------------
    # Tab 3: Body System Explorer
    # --------------------------
    with tab_body:
        st.subheader("Body System Explorer")

        all_systems = get_all_body_system_labels(g_inf)
        if not all_systems:
            st.warning("No med:BodySystem instances found in KG.")
        else:
            col_left, col_right = st.columns([1, 2])

            with col_left:
                selected_sys = st.selectbox("Select body system", all_systems)

                st.markdown("**Filters (from CSV metadata)**")
                only_chronic = st.checkbox("Only chronic diseases", value=False)
                only_infectious = st.checkbox("Only infectious diseases", value=False)

            with col_right:
                sys_diseases = get_diseases_by_body_system(g_inf, selected_sys)
                sys_df = pd.DataFrame(sys_diseases)

                if sys_df.empty:
                    st.info(f"No diseases found for **{selected_sys}** in the KG.")
                else:
                    joined = _safe_merge_with_csv(sys_df, df)

                    # Apply filters only if columns exist
                    if only_chronic:
                        if "chronic" in joined.columns:
                            joined = joined[joined["chronic"] == True]
                        else:
                            st.warning("CSV column 'chronic' not found — cannot apply chronic filter.")

                    if only_infectious:
                        if "contagious" in joined.columns:
                            joined = joined[joined["contagious"] == True]
                        else:
                            st.warning("CSV column 'contagious' not found — cannot apply infectious filter.")

                    st.write(f"**{selected_sys}** – {len(joined)} diseases after filters.")

                    cols_to_show = ["disease"]
                    for c in ["category", "body_system", "chronic", "contagious"]:
                        if c in joined.columns and c not in cols_to_show:
                            cols_to_show.append(c)

                    st.dataframe(joined[cols_to_show], width="stretch")

                    csv_body = joined[cols_to_show].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download diseases in this system (CSV)",
                        csv_body,
                        file_name="body_system_diseases.csv",
                        mime="text/csv",
                    )

    
    # --------------------------
    # Tab 4: Disease Explorer
    # --------------------------
    with tab_disease:
        st.subheader("Disease Explorer")

        disease_labels = get_all_disease_labels(g_inf)
        if not disease_labels:
            st.warning("No med:Disease individuals found in inferred KG.")
        else:
            selected_disease = st.selectbox("Select a disease", disease_labels)

            details = get_disease_details(g_inf, selected_disease)
            if not details:
                st.warning("Disease not found in KG details query.")
            else:
                # Join CSV metadata for this disease (if exists)
                meta = None
                if "disease_name" in df.columns:
                    tmp = df[df["disease_name"] == selected_disease]
                    if not tmp.empty:
                        meta = tmp.iloc[0].to_dict()

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    st.markdown("### KG facts")
                    st.write(f"**URI:** `{details['disease_uri']}`")
                    st.write(f"**Body system:** **{details['body_system']}**")
                    st.markdown("**Symptoms (from KG):**")
                    st.write(details["symptoms"] if details["symptoms"] else "_None in KG._")

                with col_b:
                    st.markdown("### Treatments")
                    st.write(details["treatments"] if details["treatments"] else "_None in KG._")

                    if meta is not None:
                        st.markdown("### CSV metadata")
                        cols = ["category", "body_system", "chronic", "contagious"]
                        for c in cols:
                            if c in meta:
                                st.write(f"**{c}:** {meta[c]}")

                # Download a compact “disease card”
                disease_card = {
                    "disease": details["disease"],
                    "disease_uri": details["disease_uri"],
                    "body_system": details["body_system"],
                    "symptoms": details["symptoms"],
                    "treatments": details["treatments"],
                }
                st.download_button(
                    "Download disease card (JSON)",
                    data=pd.Series(disease_card).to_json(),
                    file_name="disease_card.json",
                    mime="application/json",
                )

    # --------------------------
    # Tab 5: Treatment Explorer
    # --------------------------
    with tab_treatment:
        st.subheader("Treatment Explorer")

        treatments = get_all_treatment_labels(g_inf)
        if not treatments:
            st.warning("No med:Treatment individuals found in inferred KG.")
        else:
            selected_treat = st.selectbox("Select a treatment", treatments)

            rows = get_diseases_by_treatment(g_inf, selected_treat)
            tdf = pd.DataFrame(rows)

            st.write(f"**{selected_treat}** is linked to **{len(tdf)}** diseases in the KG.")

            if tdf.empty:
                st.info("No diseases found for this treatment.")
            else:
                # Optional: join CSV metadata (category/chronic/contagious)
                joined = _safe_merge_with_csv(tdf, df)

                cols_to_show = ["disease", "body_system", "treatment"]
                for c in ["category", "chronic", "contagious"]:
                    if c in joined.columns and c not in cols_to_show:
                        cols_to_show.append(c)

                st.dataframe(joined[cols_to_show], width="stretch")

                csv_bytes = joined[cols_to_show].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download treatment → diseases (CSV)",
                    csv_bytes,
                    file_name="treatment_diseases.csv",
                    mime="text/csv",
                )

    # --------------------------
    # Tab 6: SPARQL Playground (FIXED: editor + dropdown)
    # --------------------------
    with tab_sparql:
        st.subheader("SPARQL Playground")
        st.write("Run **SELECT** queries on the KG. (UI limits output rows to keep it fast.)")

        editor_key = "sparql_query_editor"
        pending_editor_key = "sparql_pending_editor_text"

        demo_key = "sparql_demo_choice"
        pending_demo_key = "sparql_pending_demo_choice"

        custom_label = "(Custom / write your own)"

        # Choose which KG to query
        choices = ["Reasoned (after RDFS closure)", "Raw (no reasoning)"]

        # Only show OWL-RL option if the file exists / graph loaded
        if g_owl is not None:
            choices.insert(0, "OWL-RL (for RDFS limitation demo)")

        kg_choice = st.radio(
            "KG version",
            choices,
            horizontal=True,
            key="sparql_kg_choice",
        )

        if kg_choice.startswith("Raw"):
            g_used = g_raw
        elif kg_choice.startswith("OWL-RL"):
            g_used = g_owl
        else:
            g_used = g_inf


        # Default query (safe fallback)
        default_query = f"""PREFIX med: <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?d_label ?sys_label
    WHERE {{
    ?d a med:Disease ;
        rdfs:label ?d_label ;
        med:affectsSystem ?sys .
    ?sys rdfs:label ?sys_label .
    }}
    LIMIT 20
    """.strip()

        # ------------------------------------------------------------
        # IMPORTANT: init/apply pending state BEFORE widgets exist
        # ------------------------------------------------------------
        if editor_key not in st.session_state:
            st.session_state[editor_key] = default_query
        if pending_editor_key not in st.session_state:
            st.session_state[pending_editor_key] = None

        if demo_key not in st.session_state:
            st.session_state[demo_key] = custom_label
        if pending_demo_key not in st.session_state:
            st.session_state[pending_demo_key] = None

        # Apply pending dropdown choice BEFORE selectbox is created
        if st.session_state[pending_demo_key] is not None:
            st.session_state[demo_key] = st.session_state[pending_demo_key]
            st.session_state[pending_demo_key] = None

        # Apply pending editor text BEFORE text_area is created
        if st.session_state[pending_editor_key] is not None:
            st.session_state[editor_key] = st.session_state[pending_editor_key]
            st.session_state[pending_editor_key] = None

        # ---- Demo query support ----
        st.markdown("### Demo queries (Q1–Q9)")
        demo_names = [custom_label] + list(DEMO_QUERY_BUILDERS.keys())

        st.checkbox(
            "Auto-fill editor with selected demo query",
            value=True,
            key="sparql_autofill",
        )

        def build_demo_query(choice: str) -> str:
            if choice == custom_label:
                return ""

            if choice.startswith("Q1"):
                limit = int(st.session_state.get("q1_limit", 20))
                return DEMO_QUERY_BUILDERS[choice](limit=limit)

            if choice.startswith("Q2"):
                symptom_label = st.session_state.get("q2_sym", "fever")
                return DEMO_QUERY_BUILDERS[choice](symptom_label=symptom_label)

            if choice.startswith("Q3"):
                raw = st.session_state.get("q3_syms", "fever, cough, headache")
                symptoms = [x.strip() for x in raw.split(",") if x.strip()]
                return DEMO_QUERY_BUILDERS[choice](symptoms=symptoms)

            if choice.startswith("Q5"):
                limit = int(st.session_state.get("q5_limit", 30))
                return DEMO_QUERY_BUILDERS[choice](limit=limit)

            if choice.startswith("Q6"):
                body_system_label = st.session_state.get("q6_sys", "Respiratory system")
                min_symptoms = int(st.session_state.get("q6_min", 3))
                return DEMO_QUERY_BUILDERS[choice](
                    body_system_label=body_system_label,
                    min_symptoms=min_symptoms,
                )

            if choice.startswith("Q7"):
                treatment_text = st.session_state.get("q7_treat", "antibiotic")
                return DEMO_QUERY_BUILDERS[choice](treatment_text=treatment_text)

            if choice.startswith("Q9"):
                disease_label = st.session_state.get("q9_dis", "Pneumonia")
                return DEMO_QUERY_BUILDERS[choice](disease_label=disease_label)

            return DEMO_QUERY_BUILDERS[choice]()  # Q4, Q8 etc.

        def maybe_autofill():
            """Callback-safe: never call st.rerun() here."""
            if st.session_state.get("sparql_autofill", True):
                choice = st.session_state.get(demo_key, custom_label)
                demo_q = build_demo_query(choice)
                if demo_q:
                    st.session_state[pending_editor_key] = demo_q

        demo_choice = st.selectbox(
            "Choose a demo query",
            demo_names,
            key=demo_key,
            on_change=maybe_autofill,
        )

        # ---- Parameter widgets ----
        if demo_choice != custom_label:
            if demo_choice.startswith("Q1"):
                st.number_input("Limit", 1, 500, 20, key="q1_limit", on_change=maybe_autofill)

            elif demo_choice.startswith("Q2"):
                st.text_input("Symptom label", "fever", key="q2_sym", on_change=maybe_autofill)

            elif demo_choice.startswith("Q3"):
                st.text_input("Symptoms (comma-separated)", "fever, cough, headache", key="q3_syms", on_change=maybe_autofill)

            elif demo_choice.startswith("Q5"):
                st.number_input("Limit", 1, 500, 30, key="q5_limit", on_change=maybe_autofill)

            elif demo_choice.startswith("Q6"):
                st.text_input("Body system label", "Respiratory system", key="q6_sys", on_change=maybe_autofill)
                st.number_input("Min symptoms", 1, 50, 3, key="q6_min", on_change=maybe_autofill)

            elif demo_choice.startswith("Q7"):
                st.text_input("Treatment contains", "antibiotic", key="q7_treat", on_change=maybe_autofill)

            elif demo_choice.startswith("Q9"):
                st.text_input("Disease label", "Pneumonia", key="q9_dis", on_change=maybe_autofill)

            # Manual apply (if autofill is OFF)
            if not st.session_state.get("sparql_autofill", True):
                if st.button("Apply demo query to editor", width="stretch"):
                    q_demo = build_demo_query(demo_choice)
                    if q_demo:
                        st.session_state[pending_editor_key] = q_demo
                        st.rerun()

        # ---- Editor ----
        q = st.text_area(
            "SPARQL query (editable)",
            height=260,
            key=editor_key,
        )

        limit_rows = st.slider("Max rows to display", 50, 500, 200, key="sparql_limit_rows")

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_btn = st.button("Run query", width="stretch")

        with col_clear:
            if st.button("Clear to default", width="stretch"):
                # Queue BOTH changes, then rerun
                st.session_state[pending_editor_key] = default_query
                st.session_state[pending_demo_key] = custom_label
                st.rerun()

        if run_btn:
            try:
                result = run_sparql_query(g_used, q, limit_rows=limit_rows)

                if result["form"] == "EMPTY":
                    st.info("Please paste a query first.")
                    st.stop()

                if result["form"] == "SELECT":
                    out_df = result["df"]
                    if out_df.empty:
                        st.info("Query returned 0 rows.")
                    else:
                        st.dataframe(out_df, width="stretch")
                        st.download_button(
                            "Download results (CSV)",
                            out_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"sparql_select_{kg_choice}_results.csv",
                            mime="text/csv",
                            width="stretch",
                        )

                elif result["form"] == "ASK":
                    ans = result["ask"]
                    if ans:
                        st.success("ASK result: TRUE")
                    else:
                        st.warning("ASK result: FALSE")

                elif result["form"] in ("CONSTRUCT", "DESCRIBE"):
                    triples_df = result["triples_df"]
                    st.info(f"{result['form']} produced an RDF subgraph (showing up to {len(triples_df)} triples).")
                    st.dataframe(triples_df, width="stretch")

                    st.download_button(
                        "Download triples (CSV)",
                        triples_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"sparql_{result['form'].lower()}_{kg_choice}_triples.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                    st.download_button(
                        "Download RDF (Turtle .ttl)",
                        result["ttl"].encode("utf-8"),
                        file_name=f"sparql_{result['form'].lower()}_{kg_choice}_result.ttl",
                        mime="text/turtle",
                        width="stretch",
                    )

                    with st.expander("Show Turtle preview"):
                        st.code(result["ttl"], language="turtle")

                else:
                    st.error(f"Unhandled result form: {result['form']}")

            except Exception as e:
                st.error(f"SPARQL error: {e}")

    # --------------------------
    # Tab 7: Visualizations (with expanders + your exact filenames)
    # --------------------------
    with tab_vis:
        st.subheader("Visualizations")

        vis_dir = DATA_DIR / "visualizations"
        if not vis_dir.exists():
            st.warning("Visualization directory not found. Run Step 5 to generate plots.")
            st.stop()

        # Helper: render 1 or many images nicely
        def show_images(items, ncols=3):
            """
            items: list of dicts:
            {"file": "name.png", "caption": "...", "desc": "..."}
            """
            existing = []
            missing = []
            for it in items:
                p = vis_dir / it["file"]
                if p.exists():
                    existing.append((p, it))
                else:
                    missing.append(it["file"])

            if missing:
                st.info("Missing images: " + ", ".join(missing))

            if not existing:
                st.warning("No images available in this section.")
                return

            # If only 1 image -> full width
            if len(existing) == 1:
                p, it = existing[0]
                st.image(str(p), caption=it.get("caption", it["file"]), width="stretch")
                if it.get("desc"):
                    st.caption(it["desc"])
                return

            # Multiple -> grid
            for i in range(0, len(existing), ncols):
                row = existing[i : i + ncols]
                cols = st.columns(len(row))
                for col, (p, it) in zip(cols, row):
                    with col:
                        st.image(str(p), caption=it.get("caption", it["file"]), width="stretch")
                        if it.get("desc"):
                            st.caption(it["desc"])

        # --------------------------
        # Expander 1: Graphviz & Python-generated plots
        # --------------------------
        with st.expander("📌 RDF / Graphviz / Python-generated visualizations", expanded=True):
            st.write(
                "These plots are generated from the RDF schema/graph and offline analysis "
                "to explain the ontology structure and KG topology."
            )

            # Ontology (Graphviz)
            show_images(
                [
                    {
                        "file": "ontology_graphviz_large.png",
                        "caption": "Ontology schema (Graphviz)",
                        "desc": "Schema-level view: main classes and properties in the ontology.",
                    }
                ],
                ncols=1,
            )

            st.markdown("#### Graph structure views (samples)")
            show_images(
                [
                    {
                        "file": "instance_kg_large_sample.png",
                        "caption": "Instance KG sample",
                        "desc": "A small extracted subgraph showing Diseases connected to Symptoms/Treatments/BodySystems.",
                    },
                    {
                        "file": "disease_symptom_network_sample.png",
                        "caption": "Disease–Symptom network sample",
                        "desc": "Bipartite-style view: diseases linked to symptoms (structure overview).",
                    },
                    {
                        "file": "disease_similarity_network_sample.png",
                        "caption": "Disease similarity network sample",
                        "desc": "Diseases connected if they share symptoms (similarity by overlap).",
                    },
                ],
                ncols=2,
            )

            st.markdown("#### Distributions & degree statistics")
            show_images(
                [
                    {
                        "file": "body_system_distribution.png",
                        "caption": "Body system distribution",
                        "desc": "Counts of diseases per body system category (dataset/instances overview).",
                    },
                    {
                        "file": "symptom_frequency_top40.png",
                        "caption": "Top-40 symptom frequency",
                        "desc": "Most frequent symptoms across all diseases (frequency analysis).",
                    },
                    {
                        "file": "disease_symptom_degree_hist.png",
                        "caption": "Disease → symptom degree histogram",
                        "desc": "How many symptoms each disease has (degree distribution).",
                    },
                    {
                        "file": "symptom_disease_degree_hist.png",
                        "caption": "Symptom → disease degree histogram",
                        "desc": "How many diseases each symptom appears in (degree distribution).",
                    },
                ],
                ncols=2,
            )

        # --------------------------
        # Expander 2: Neo4j screenshots / exports
        # --------------------------
        with st.expander("🧠 Neo4j visualizations (screenshots / graph exploration)", expanded=False):
            st.write(
                "These are Neo4j Browser/Bloom screenshots showing interactive exploration "
                "of the graph (query results, neighborhoods, schema-level view)."
            )

            st.markdown("#### Neo4j schema / ontology views")
            show_images(
                [
                    {
                        "file": "neo4j_schema_level.png",
                        "caption": "Neo4j schema-level view",
                        "desc": "High-level overview of node labels and relationship types.",
                    },
                    {
                        "file": "neo4j_ontology.png",
                        "caption": "Neo4j ontology view",
                        "desc": "Ontology/schema graph as explored in Neo4j (classes/properties).",
                    },
                ],
                ncols=2,
            )

            st.markdown("#### Neo4j instance exploration (examples)")
            show_images(
                [
                    {
                        "file": "neo4j_disease_symptom.png",
                        "caption": "Disease → Symptom neighborhood",
                        "desc": "Example query result showing a disease connected to its symptoms.",
                    },
                    {
                        "file": "neo4j_disease_treatment.png",
                        "caption": "Disease → Treatment neighborhood",
                        "desc": "Example query result showing a disease linked to treatments.",
                    },
                    {
                        "file": "neo4j_disease_bodysystem.png",
                        "caption": "Disease → BodySystem neighborhood",
                        "desc": "Example query result showing the body system affected by a disease.",
                    },
                ],
                ncols=2,
            )

            st.markdown("#### Neo4j focused demos (for professor)")
            show_images(
                [
                    {
                        "file": "neo4j_single_disease_syptom.png",
                        "caption": "Single disease with symptoms",
                        "desc": "Focused subgraph for one disease: symptom connections.",
                    },
                    {
                        "file": "neo4j_single_disease_treatment.png",
                        "caption": "Single disease with treatments",
                        "desc": "Focused subgraph for one disease: treatment connections.",
                    },
                    {
                        "file": "neo4j_single_disease_syptom_treamtment.png",
                        "caption": "Single disease with symptoms + treatments",
                        "desc": "Combined neighborhood view (symptoms + treatments).",
                    },
                ],
                ncols=2,
            )

            st.markdown("#### Neo4j query-based patterns")
            show_images(
                [
                    {
                        "file": "neo4j_fever_more_disease.png",
                        "caption": "Symptom 'fever' linked to many diseases",
                        "desc": "Example: exploring a common symptom and its connected diseases.",
                    },
                    {
                        "file": "neo4j_treatment_with_morethan_5_diease.png",
                        "caption": "Treatment linked to >5 diseases",
                        "desc": "Example: exploring treatments that apply to many diseases.",
                    },
                ],
                ncols=2,
            )


                            
    
    # --------------------------
    # Tab 8: KG + LLM Assistant
    # --------------------------
    with tab_llm:
        render_llm_tab()    
    
    # --------------------------
    # Tab 9: LLM-assisted KG Reasoner (Large)   
    # --------------------------
    with tab_llm_large:
        render_llm_kg_reasoner_large()
    
    # 
    # --------------------------
    # NEW TAB: RDFS vs OWL-RL Limitations
    # --------------------------
    with tab_limits:
        st.subheader("RDFS vs OWL-RL – Limitations (Evidence from our KG)")

        st.markdown(
            """
    **Quick summary**
    - **RDFS reasoning** can infer taxonomy-level facts (e.g., `subClassOf`, `subPropertyOf`, `domain`, `range`) and propagate types.
    - **RDFS cannot** use OWL semantics such as:
    - **logical class definitions** (`owl:equivalentClass`, `owl:Restriction`)
    - **inverse properties** (`owl:inverseOf`)
    - **OWL-RL** can materialize these inferences (rule-based OWL profile, still scalable).
    """
        )

        # Load OWL-RL inferred graph (created by pipeline Step 7)
        g_owl = load_graph_owlrl(OWL_RL_TTL_PATH)

        if g_owl is None:
            st.warning(
                "OWL-RL inferred KG not found. Run the pipeline with Step 7 enabled to generate:\n"
                f"{OWL_RL_TTL_PATH}"
            )
            st.stop()

        # Small helper: run SELECT and return DataFrame
        def _run_select_df(g: Graph, query: str, limit_rows: int = 500) -> pd.DataFrame:
            res = g.query(query)
            cols = [str(v) for v in res.vars]
            rows = []
            for i, r in enumerate(res):
                if i >= limit_rows:
                    break
                row = {}
                for c, v in zip(cols, r):
                    row[c] = str(v) if v is not None else None
                rows.append(row)
            return pd.DataFrame(rows)

        st.markdown("---")

        # ==========================================================
        # Limitation #1: OWL-defined class membership (Q10) + CONTROL
        # ==========================================================
        st.markdown("## Limitation #1 — OWL-defined class membership (equivalentClass / restriction)")

        # --- CONTROL query (facts-based) ---
        q10_control = f"""
        PREFIX med:  <{MED}>
        PREFIX rdfs: <{RDFS}>

        SELECT ?d ?label
        WHERE {{
        ?d a med:ChronicDisease ;
            med:affectsSystem med:RespiratorySystem ;
            rdfs:label ?label .
        }}
        ORDER BY ?label
        """.strip()

        # --- LIMITATION query (OWL-defined class) ---
        q10_limitation = f"""
        PREFIX med:  <{MED}>
        PREFIX rdfs: <{RDFS}>

        SELECT ?d ?label
        WHERE {{
        ?d a med:RespiratoryChronicDisease ;
            rdfs:label ?label .
        }}
        ORDER BY ?label
        """.strip()

        with st.expander("Show Limitation #1 queries (Control + Limitation)"):
            st.markdown("**Q10-CONTROL (facts-based): ChronicDisease + affectsSystem med:RespiratorySystem**")
            st.code(q10_control, language="sparql")
            st.markdown("**Q10 (LIMITATION): OWL-defined class membership (med:RespiratoryChronicDisease)**")
            st.code(q10_limitation, language="sparql")

        # Run on both graphs
        df_ctrl_rdfs = _run_select_df(g_inf, q10_control, limit_rows=2000)
        df_ctrl_owl  = _run_select_df(g_owl, q10_control, limit_rows=2000)

        df_lim_rdfs  = _run_select_df(g_inf, q10_limitation, limit_rows=2000)
        df_lim_owl   = _run_select_df(g_owl, q10_limitation, limit_rows=2000)

        st.markdown("### Q10-CONTROL (facts-based) — should match in both reasoners")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RDFS count (Q10-CONTROL)", int(len(df_ctrl_rdfs)))
            st.dataframe(df_ctrl_rdfs, width="stretch")
        with c2:
            st.metric("OWL-RL count (Q10-CONTROL)", int(len(df_ctrl_owl)))
            st.dataframe(df_ctrl_owl, width="stretch")

        st.markdown("### Q10 (OWL-defined class) — RDFS fails, OWL-RL succeeds")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RDFS count (Q10)", int(len(df_lim_rdfs)))
            if df_lim_rdfs.empty:
                st.info("RDFS returns 0 rows because it cannot materialize OWL class definitions (equivalentClass / restriction).")
            else:
                st.dataframe(df_lim_rdfs, width="stretch")
        with c2:
            st.metric("OWL-RL count (Q10)", int(len(df_lim_owl)))
            st.dataframe(df_lim_owl, width="stretch")


        # ==========================================================
        # Limitation #2: Inverse properties (owl:inverseOf)
        # ==========================================================
        st.markdown("## Limitation #2 — Inverse properties (owl:inverseOf)")

        symptom_label = st.text_input("Symptom for inverse demo", value="fever", key="limits_symptom_label").strip().lower()

        q11_a_control = f"""
    PREFIX med:  <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?d ?label
    WHERE {{
    ?d a med:Disease ;
        med:hasSymptom ?s ;
        rdfs:label ?label .
    ?s rdfs:label ?sl .
    FILTER(LCASE(STR(?sl)) = "{symptom_label}")
    }}
    ORDER BY ?label
    """.strip()

        q11_b_inverse = f"""
    PREFIX med:  <{MED}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?d ?label
    WHERE {{
    ?s rdfs:label ?sl ;
        med:symptomOf ?d .
    FILTER(LCASE(STR(?sl)) = "{symptom_label}")
    ?d rdfs:label ?label .
    }}
    ORDER BY ?label
    """.strip()

        with st.expander("Show inverse demo queries (Q11-A control and Q11-B limitation)"):
            st.markdown("**Q11-A (CONTROL): asserted direction `Disease → hasSymptom → Symptom`**")
            st.code(q11_a_control, language="sparql")
            st.markdown("**Q11-B (LIMITATION): inverse direction `Symptom → symptomOf → Disease`**")
            st.code(q11_b_inverse, language="sparql")

        df_a_rdfs = _run_select_df(g_inf, q11_a_control, limit_rows=2000)
        df_a_owl  = _run_select_df(g_owl, q11_a_control, limit_rows=2000)

        df_b_rdfs = _run_select_df(g_inf, q11_b_inverse, limit_rows=2000)
        df_b_owl  = _run_select_df(g_owl, q11_b_inverse, limit_rows=2000)

        st.markdown("### Q11-A (Control) — should match in both reasoners")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RDFS count (Q11-A)", int(len(df_a_rdfs)))
            st.dataframe(df_a_rdfs, width="stretch")
        with c2:
            st.metric("OWL-RL count (Q11-A)", int(len(df_a_owl)))
            st.dataframe(df_a_owl, width="stretch")

        st.markdown("### Q11-B (Inverse) — RDFS fails, OWL-RL succeeds")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RDFS count (Q11-B)", int(len(df_b_rdfs)))
            if df_b_rdfs.empty:
                st.info("RDFS returns 0 rows because it does not infer `owl:inverseOf` relations.")
            else:
                st.dataframe(df_b_rdfs, width="stretch")
        with c2:
            st.metric("OWL-RL count (Q11-B)", int(len(df_b_owl)))
            st.dataframe(df_b_owl, width="stretch")

        st.markdown("---")

        # Optional: show your saved limitation report file
        rep = PROCESSED_DIR / "rdfs_limitations_report.txt"
        if rep.exists():
            with st.expander("Show saved limitation report (from pipeline)"):
                st.code(rep.read_text(encoding="utf-8")[:12000])

if __name__ == "__main__":
    main()
