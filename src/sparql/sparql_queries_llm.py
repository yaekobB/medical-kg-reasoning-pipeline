from __future__ import annotations

"""
SPARQL helpers for LLM integration.

This module focuses on:
  - Loading the inferred KG
  - Differential diagnosis from symptoms
  - Disease explanation (URI, system, types, symptoms)

It is used both:
  - from the CLI (python -m src.sparql.sparql_queries_llm)
  - from the Streamlit UI (LLM Assistant tab)
"""

from dataclasses import dataclass
from typing import List, Set, Optional

from rdflib import Graph, Namespace
from src import config


MED = Namespace("http://example.org/medkg#")


# ================================================================
# Data classes for structured results
# ================================================================


@dataclass
class DifferentialCandidate:
    disease_uri: str
    disease_label: str
    system_label: str
    matched_symptoms: Set[str]


@dataclass
class DiseaseExplanation:
    disease_uri: str
    disease_label: str
    system_labels: List[str]
    type_labels: List[str]
    symptom_labels: List[str]


# ================================================================
# Graph loading
# ================================================================


def load_graph_for_llm() -> Graph:
    """
    Load the *inferred* KG (data_medical_large_inferred.ttl).

    This is used by:
      - CLI demo
      - Streamlit LLM tab
    """
    g = Graph()
    ttl_path = config.KG_INFERRED_TTL
    print(f"[LLM] Loading inferred KG from: {ttl_path}")
    g.parse(ttl_path, format="turtle")
    print(f"[LLM] Loaded graph with {len(g)} triples.")
    return g


# ================================================================
# KG query: differential diagnosis from symptoms
# ================================================================


def kg_differential_diagnosis(
    g: Graph,
    symptoms: List[str],
    limit: int = 10,
) -> List[DifferentialCandidate]:
    """
    Given a list of symptom labels (e.g. ["fever", "cough"]),
    find diseases that have ANY of them, and rank by number of matches.

    This is the same logic as Q3 in sparql_queries_large, but packaged
    as a reusable function for LLM integration and UI.
    """
    if not symptoms:
        return []

    target_symptoms = {s.lower() for s in symptoms}
    
    print(f"[LLM] Querying: Differential diagnosis from symptoms: {symptoms}")
    
    # 1) Get all disease–symptom pairs (plus system) from the KG
    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?dLabel ?sLabel ?systemLabel
    WHERE {
        ?d a med:Disease ;
           med:hasSymptom ?s ;
           rdfs:label ?dLabel .

        ?s rdfs:label ?sLabel .

        OPTIONAL {
            ?d med:affectsSystem ?sys .
            ?sys rdfs:label ?systemLabel .
        }
    }
    """

    results = list(g.query(query))

    by_disease = {}  # d_uri -> {"label": ..., "system": ..., "matched": set([...])}

    for row in results:
        d_uri, d_label, s_label, sys_label = row
        s_label_str = str(s_label)
        s_lower = s_label_str.lower()

        if s_lower not in target_symptoms:
            continue

        if d_uri not in by_disease:
            by_disease[d_uri] = {
                "label": str(d_label),
                "system": str(sys_label) if sys_label is not None else "Unknown / not classified",
                "matched": set(),
            }

        by_disease[d_uri]["matched"].add(s_lower)

    if not by_disease:
        return []

    ranked = sorted(
        by_disease.items(),
        key=lambda kv: (-len(kv[1]["matched"]), kv[1]["label"]),
    )

    candidates: List[DifferentialCandidate] = []
    for d_uri, info in ranked[:limit]:
        candidates.append(
            DifferentialCandidate(
                disease_uri=str(d_uri),
                disease_label=info["label"],
                system_label=info["system"],
                matched_symptoms=set(info["matched"]),
            )
        )

    return candidates


# ================================================================
# KG query: explain a disease by label
# ================================================================


def kg_explain_disease(g: Graph, target_label: str) -> Optional[DiseaseExplanation]:
    """
    Given a disease label (case-insensitive), return a structured
    explanation object (URI, systems, types, symptoms).

    Uses a single SPARQL query, then aggregates in Python.
    """
    if not target_label:
        return None

    print(f"[LLM] Explain disease from KG: {target_label}")

    sparql = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?d_label ?system_label ?sym_label ?t ?t_label
    WHERE {
      ?d a med:Disease ;
         rdfs:label ?d_label .
      FILTER(LCASE(STR(?d_label)) = LCASE(?target_label))

      OPTIONAL {
        ?d med:affectsSystem ?sys .
        ?sys rdfs:label ?system_label .
      }

      OPTIONAL {
        ?d med:hasSymptom ?s .
        ?s rdfs:label ?sym_label .
      }

      OPTIONAL {
        ?d a ?t .
        VALUES ?t { med:Disease med:ChronicDisease med:InfectiousDisease }
        OPTIONAL { ?t rdfs:label ?t_label . }
      }
    }
    """

    # Bind target label via string replacement
    sparql = sparql.replace("?target_label", f'"{target_label}"')
    print(f"[LLM] Running SPARQL query to explain disease...")
    rows = list(g.query(sparql))

    if not rows:
        return None

    systems: Set[str] = set()
    symptoms: Set[str] = set()
    types: Set[str] = set()
    disease_uri: Optional[str] = None
    disease_name: Optional[str] = None

    for row in rows:
        d_uri, d_label, sys_label, sym_label, t_uri, t_label = row

        if disease_uri is None:
            disease_uri = str(d_uri)
        if disease_name is None and d_label is not None:
            disease_name = str(d_label)

        if sys_label:
            systems.add(str(sys_label))
        if sym_label:
            symptoms.add(str(sym_label))

        if t_uri:
            if t_label:
                types.add(str(t_label))
            else:
                local = str(t_uri).split("#")[-1]
                types.add(local)

    if disease_uri is None or disease_name is None:
        return None

    return DiseaseExplanation(
        disease_uri=disease_uri,
        disease_label=disease_name,
        system_labels=sorted(systems) if systems else [],
        type_labels=sorted(types) if types else [],
        symptom_labels=sorted(symptoms) if symptoms else [],
    )


# ================================================================
# CLI demo (so python -m src.sparql.sparql_queries_llm still works)
# ================================================================


def main() -> None:
    g = load_graph_for_llm()

    # Demo 1: differential diagnosis
    symptoms = ["fever", "cough", "headache"]
    print(f"\n[LLM DEMO] Differential diagnosis for: {symptoms}")
    candidates = kg_differential_diagnosis(g, symptoms, limit=5)
    for c in candidates:
        print(
            f"- {c.disease_label} | system={c.system_label} | "
            f"matches={len(c.matched_symptoms)} | "
            f"symptoms={sorted(c.matched_symptoms)}"
        )

    # Demo 2: explain Pneumonia
    print("\n[LLM DEMO] Explain Pneumonia")
    expl = kg_explain_disease(g, "Pneumonia")
    if expl is None:
        print("No disease 'Pneumonia' found in KG.")
    else:
        print(f"Disease: {expl.disease_label}")
        print(f"URI    : {expl.disease_uri}")
        print("Systems:", ", ".join(expl.system_labels or ["Unknown / not classified"]))
        print("Types  :", ", ".join(expl.type_labels or ["med:Disease"]))
        print("Symptoms:")
        for s in expl.symptom_labels:
            print(f" - {s}")


if __name__ == "__main__":
    main()
