# src/sparql/demo_queries.py
from __future__ import annotations

from typing import Dict, List

# ---------- small helper to safely insert user strings into SPARQL literals ----------
def sparql_escape_literal(s: str) -> str:
    """
    Escape a Python string so it can be safely inserted inside "..." in SPARQL.
    """
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    return s


# ---------- Q1–Q9 query builders (return SPARQL strings) ----------

def q1(limit: int = 20) -> str:
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?disease_label ?type_label ?system_label
WHERE {{
  ?d a med:Disease ;
     rdfs:label ?disease_label .

  OPTIONAL {{
    ?d a ?t .
    VALUES ?t {{ med:Disease med:ChronicDisease med:InfectiousDisease }}
    ?t rdfs:label ?type_label .
  }}

  OPTIONAL {{
    ?d med:affectsSystem ?sys .
    ?sys rdfs:label ?system_label .
  }}
}}
LIMIT {int(limit)}
""".strip()


def q2(symptom_label: str = "fever") -> str:
    sym = sparql_escape_literal(symptom_label)
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?disease_label ?system_label
WHERE {{
  ?s a med:Symptom ;
     rdfs:label "{sym}"@en .

  ?d a med:Disease ;
     med:hasSymptom ?s ;
     rdfs:label ?disease_label .

  OPTIONAL {{
    ?d med:affectsSystem ?sys .
    ?sys rdfs:label ?system_label .
  }}
}}
ORDER BY ?disease_label
""".strip()


def q3(symptoms: List[str]) -> str:
    # use VALUES to pass multiple symptoms
    safe = [f"\"{sparql_escape_literal(x)}\"@en" for x in symptoms if str(x).strip()]
    if not safe:
        safe = ['"fever"@en']

    values = " ".join(safe)
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?disease_label (COUNT(DISTINCT ?s) AS ?matches)
WHERE {{
  VALUES ?symLabel {{ {values} }}

  ?s a med:Symptom ;
     rdfs:label ?symLabel .

  ?d a med:Disease ;
     med:hasSymptom ?s ;
     rdfs:label ?disease_label .
}}
GROUP BY ?disease_label
ORDER BY DESC(?matches) ?disease_label
""".strip()


def q4() -> str:
    return """
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?system_label
       (COUNT(DISTINCT ?d) AS ?total)
       (COUNT(DISTINCT ?dc) AS ?chronic_total)
WHERE {
  ?d a med:Disease ;
     med:affectsSystem ?sys .
  ?sys rdfs:label ?system_label .

  OPTIONAL {
    ?dc a med:ChronicDisease ;
        med:affectsSystem ?sys .
  }
}
GROUP BY ?system_label
ORDER BY DESC(?total)
""".strip()


def q5(limit: int = 30) -> str:
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?d1_label ?d2_label ?sym_label
WHERE {{
  ?d1 a med:Disease ; med:hasSymptom ?s ; rdfs:label ?d1_label .
  ?d2 a med:Disease ; med:hasSymptom ?s ; rdfs:label ?d2_label .
  ?s rdfs:label ?sym_label .
  FILTER(?d1 != ?d2)
}}
LIMIT {int(limit)}
""".strip()


def q6(body_system_label: str = "Respiratory system", min_symptoms: int = 3) -> str:
    syslab = sparql_escape_literal(body_system_label)
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?disease_label (COUNT(DISTINCT ?s) AS ?numSymptoms)
WHERE {{
  ?sys a med:BodySystem ;
       rdfs:label ?sys_label .

  # Match body system label regardless of language tag (@en) or no tag
  FILTER(LCASE(STR(?sys_label)) = LCASE("{syslab}"))

  ?d a med:ChronicDisease ;
     med:affectsSystem ?sys ;
     rdfs:label ?disease_label ;
     med:hasSymptom ?s .
}}
GROUP BY ?disease_label
HAVING (COUNT(DISTINCT ?s) >= {int(min_symptoms)})
ORDER BY DESC(?numSymptoms) ?disease_label
""".strip()



def q7(treatment_text: str = "antibiotic") -> str:
    t = sparql_escape_literal(treatment_text).lower()
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?disease_label ?system_label ?treat_label
WHERE {{
  ?d a med:Disease ;
     med:treatedWith ?tr ;
     rdfs:label ?disease_label .

  ?tr a med:Treatment ;
      rdfs:label ?treat_label .

  FILTER(CONTAINS(LCASE(STR(?treat_label)), "{t}"))

  OPTIONAL {{
    ?d med:affectsSystem ?sys .
    ?sys rdfs:label ?system_label .
  }}
}}
ORDER BY ?disease_label
""".strip()


def q8() -> str:
    # Example: fixed set fever+cough+headache like your console demo
    return q3(["fever", "cough", "headache"])


def q9(disease_label: str = "Pneumonia") -> str:
    d = sparql_escape_literal(disease_label)
    return f"""
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?type_label ?system_label ?sym_label
WHERE {{
  ?d a med:Disease ;
     rdfs:label "{d}"@en .

  OPTIONAL {{
    ?d a ?t .
    VALUES ?t {{ med:Disease med:ChronicDisease med:InfectiousDisease }}
    ?t rdfs:label ?type_label .
  }}

  OPTIONAL {{
    ?d med:affectsSystem ?sys .
    ?sys rdfs:label ?system_label .
  }}

  OPTIONAL {{
    ?d med:hasSymptom ?s .
    ?s rdfs:label ?sym_label .
  }}
}}
""".strip()


# ---------- registry used by the UI dropdown ----------
DEMO_QUERY_BUILDERS: Dict[str, object] = {
    "Q1 – Sample diseases with type & system": q1,
    "Q2 – Diseases by one symptom": q2,
    "Q3 – Rank diseases by symptom set": q3,
    "Q4 – Counts per body system (and chronic subset)": q4,
    "Q5 – Disease pairs sharing a symptom": q5,
    "Q6 – Chronic diseases in a system with >= N symptoms": q6,
    "Q7 – Diseases treated with treatment containing text": q7,
    "Q8 – Differential (fever+cough+headache)": q8,
    "Q9 – Explain a disease (types/system/symptoms)": q9,
}
