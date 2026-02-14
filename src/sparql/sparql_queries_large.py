"""
sparql_queries_large.py

Task 5: SPARQL analytics on the LARGE inferred medical KG.

Input graph:
    data/processed/data_medical_large_inferred.ttl

This script runs several queries that demonstrate:
  - How we use the reasoned KG (RDFS closure from Task 4)
  - How SPARQL can implement "reasoning-like" analytics

Queries:

  Q1: Overview – list a sample of diseases, with category and body system.
  Q2: Given a symptom label, find diseases that have that symptom.
  Q3: Given a list of symptoms, rank diseases by how many of them they match.
  Q4: Count diseases per body system, and how many of those are chronic.
  Q5: Find pairs of diseases that share at least one symptom.

  Q10: ASK – boolean check (e.g., does a given disease have a given symptom?).
  Q11: CONSTRUCT – build a small RDF subgraph for a disease (types, system, symptoms, treatments).
  Q12: DESCRIBE – fetch an implementation-defined description of a disease resource.

You can run the full script:

    python -m src.sparql.sparql_queries_large

or import individual functions in a notebook or another script.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from rdflib import Graph, Namespace, Literal, RDF, RDFS


# -------------------------------------------------------------------
# Paths and namespaces
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # .../medical-kg-reasoning
DATA_PROCESSED = BASE_DIR / "data" / "processed"
INFERRED_FILE = DATA_PROCESSED / "data_medical_large_inferred.ttl"

MED = Namespace("http://example.org/medkg#")


# -------------------------------------------------------------------
# Load graph
# -------------------------------------------------------------------

def load_graph() -> Graph:
    """
    Load the inferred KG from disk into an rdflib Graph.

    We expect that Task 4 (reasoning_large.py) has already been run and
    produced data_medical_large_inferred.ttl.
    """
    if not INFERRED_FILE.exists():
        raise FileNotFoundError(
            f"Inferred KG not found: {INFERRED_FILE}\n"
            "Run Task 4 (reasoning_large.py) first."
        )

    g = Graph() # Create empty graph
    print(f"[INFO] Loading inferred KG from: {INFERRED_FILE}")
    g.parse(INFERRED_FILE, format="turtle") # Parse in the data
    g.bind("med", MED) # Bind namespace for pretty output
    return g


# -------------------------------------------------------------------
# Q1: Overview – diseases with category and body system
# -------------------------------------------------------------------

def q1_list_diseases_with_category_and_system(g: Graph, limit: int = 30) -> None:
    """
    Show a sample of diseases, their main category (Chronic/Infectious/Generic)
    and the body system they affect.

    This uses the types created in Task 3 (instance building) + Task 4 reasoning.
    """

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?label ?category ?systemLabel
    WHERE {
        ?d a med:Disease ;
           rdfs:label ?label .

        # Determine a simple category label:
        OPTIONAL { ?d a med:ChronicDisease .
                   BIND(true AS ?isChronic) }
        OPTIONAL { ?d a med:InfectiousDisease .
                   BIND(true AS ?isInfectious) }

        BIND(
            IF(BOUND(?isChronic),
               "ChronicDisease",
               IF(BOUND(?isInfectious),
                  "InfectiousDisease",
                  "Disease")
            ) AS ?category
        )

        OPTIONAL {
            ?d med:affectsSystem ?sys .
            ?sys rdfs:label ?systemLabel .
        }
    }
    ORDER BY ?label
    LIMIT %d
    """ % limit

    print("\n=== Q1: Sample of diseases with category and body system ===")
    for row in g.query(query):
        _d_uri, label, category, system_label = row
        sys_str = str(system_label) if system_label is not None else "(no system)"
        print(f"- {label}  |  type: {category}  |  system: {sys_str}")


# -------------------------------------------------------------------
# Q2: Diseases that have a given symptom
# -------------------------------------------------------------------

def q2_diseases_by_symptom(g: Graph, symptom_label: str) -> None:
    """
    Given a symptom label (e.g. "fever"), list diseases that have that symptom.

    Matching is case-insensitive on the Symptom rdfs:label.
    """

    print(f"\n=== Q2: Diseases that have symptom: '{symptom_label}' ===")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?dLabel ?symLabel ?systemLabel
    WHERE {
        ?d a med:Disease ;
           med:hasSymptom ?s ;
           rdfs:label ?dLabel .

        ?s rdfs:label ?symLabel .

        BIND(LCASE(STR(?symLabel)) AS ?symLabelLower)
        BIND(LCASE(STR(?symSearch)) AS ?symSearchLower)
        FILTER(?symLabelLower = ?symSearchLower)

        OPTIONAL {
            ?d med:affectsSystem ?sys .
            ?sys rdfs:label ?systemLabel .
        }
    }
    ORDER BY ?dLabel
    """

    results = list(
        g.query(query, initBindings={"symSearch": Literal(symptom_label)})
    )

    if not results:
        print("No diseases found with this symptom label.")
        return

    for row in results:
        d_label, sym_label, sys_label = row
        sys_str = str(sys_label) if sys_label is not None else "(no system)"
        print(f"- {d_label}  | symptom: {sym_label}  | system: {sys_str}")


# -------------------------------------------------------------------
# Q3: Diseases that match a set of symptoms (ranked)
# -------------------------------------------------------------------

def q3_rank_diseases_by_symptom_set(g: Graph, symptoms: List[str]) -> None:
    """
    Given a list of symptom labels (e.g. ["fever", "cough", "headache"]),
    find diseases that have ANY of them and rank by how many match.

    Here we keep SPARQL simple (get all disease–symptom pairs), and then
    we do the matching + counting in Python. This avoids fragile string
    filters inside SPARQL and is very robust.
    """

    print(f"\n=== Q3: Diseases that match symptoms: {symptoms} ===")

    # Normalize target symptoms to lowercase once
    target_symptoms = {s.lower() for s in symptoms}

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

    # 2) Aggregate in Python: for each disease, count how many target
    #    symptoms it has, and record which ones.
    by_disease = {}  # d_uri -> {"label": ..., "system": ..., "matched": set([...])}

    for row in results:
        d_uri, d_label, s_label, sys_label = row
        s_label_str = str(s_label)
        s_lower = s_label_str.lower()

        if s_lower not in target_symptoms:
            continue  # symptom is not in the user-provided list

        if d_uri not in by_disease:
            by_disease[d_uri] = {
                "label": str(d_label),
                "system": str(sys_label) if sys_label is not None else "(no system)",
                "matched": set(),
            }

        by_disease[d_uri]["matched"].add(s_lower)

    # 3) Prepare ranking: disease -> match count
    if not by_disease:
        print("No diseases found matching any of the given symptoms.")
        return

    ranked = sorted(
        by_disease.values(),
        key=lambda x: (-len(x["matched"]), x["label"]),
    )

    # 4) Print results (top 20 for readability)
    print("Disease candidates (ranked by number of matching symptoms):")
    for i, info in enumerate(ranked[:20], start=1):
        label = info["label"]
        system = info["system"]
        matched = sorted(info["matched"])
        print(f"{i:2d}. {label}  | matches: {len(matched)}  | system: {system}  | symptoms: {matched}")



# -------------------------------------------------------------------
# Q4: Counts per body system and chronicity
# -------------------------------------------------------------------

def q4_counts_by_system_and_chronic(g: Graph) -> None:
    """
    Count how many diseases affect each body system, and within that,
    how many are chronic.

    This uses med:affectsSystem and the med:ChronicDisease subclass.
    """

    print("\n=== Q4: Counts of diseases per body system (and chronic subset) ===")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?systemLabel
           (COUNT(DISTINCT ?d) AS ?total)
           (COUNT(DISTINCT ?dChronic) AS ?chronicCount)
    WHERE {
        ?d a med:Disease ;
           med:affectsSystem ?sys .
        ?sys rdfs:label ?systemLabel .

        OPTIONAL {
            ?d a med:ChronicDisease .
            BIND(?d AS ?dChronic)
        }
    }
    GROUP BY ?systemLabel
    ORDER BY DESC(?total)
    """

    for row in g.query(query):
        system_label, total, chronic = row
        print(f"- {system_label}: total={int(total)}, chronic={int(chronic)}")


# -------------------------------------------------------------------
# Q5: Disease pairs sharing at least one symptom
# -------------------------------------------------------------------

def q5_disease_pairs_sharing_symptom(g: Graph, limit: int = 30) -> None:
    """
    Find pairs of diseases that share at least one symptom.

    We only show a limited number of pairs for readability.
    """

    print(f"\n=== Q5: Disease pairs that share at least one symptom (limit {limit}) ===")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?dLabel1 ?dLabel2 ?symLabel
    WHERE {
        ?d1 a med:Disease ;
            med:hasSymptom ?s ;
            rdfs:label ?dLabel1 .

        ?d2 a med:Disease ;
            med:hasSymptom ?s ;
            rdfs:label ?dLabel2 .

        ?s rdfs:label ?symLabel .

        # Avoid duplicates: (A,B) and (B,A)
        FILTER(STR(?d1) < STR(?d2))
    }
    ORDER BY ?dLabel1 ?dLabel2 ?symLabel
    LIMIT %d
    """ % limit

    results = list(g.query(query))
    if not results:
        print("No disease pairs found sharing symptoms (unexpected).")
        return

    for row in results:
        d1_label, d2_label, sym_label = row
        print(f"- {d1_label}  &  {d2_label}  share symptom: {sym_label}")


# -------------------------------------------------------------------
# Q6: Chronic diseases in a body system with
# -------------------------------------------------------------------
from rdflib import Graph, Literal

def q6_query_chronic_by_body_system(
    g: Graph,
    body_system_label: str,
    min_symptoms: int = 3,
):
    """
    Q6: Given a body system label and a minimum number of symptoms,
        list chronic diseases in that system with >= min_symptoms
        associated symptoms.

    This runs on the REASONED graph (data_medical_large_inferred.ttl).
    """
    print("\n=== Q6: Chronic diseases in a body system with >= "
          f"{min_symptoms} symptoms ===")
    print(f"Body system filter: {body_system_label!r}\n")

    q = f"""
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d_label (COUNT(DISTINCT ?sym) AS ?num_symptoms)
    WHERE {{
      ?d a med:ChronicDisease .
      ?d med:affectsSystem ?s .
      ?s rdfs:label ?system_label .
      ?d rdfs:label ?d_label .
      ?d med:hasSymptom ?sym .

      FILTER(LCASE(STR(?system_label)) =
             LCASE("{body_system_label}"))
    }}
    GROUP BY ?d_label
    HAVING (COUNT(DISTINCT ?sym) >= {min_symptoms})
    ORDER BY DESC(?num_symptoms) ?d_label
    """

    results = g.query(q)

    if not results:
        print("No chronic diseases found for this body system "
              f"with >= {min_symptoms} symptoms.\n")
        return

    print(f"{'Disease':60s}  #Symptoms")
    print("-" * 75)
    for row in results:
        d_label: Literal = row.d_label
        n_sym:   Literal = row.num_symptoms
        print(f"{str(d_label):60s}  {int(n_sym.toPython()):9d}")
    print()

# -------------------------------------------------------------------
# Q7: Diseases treated by a given treatment (substring match)
# -------------------------------------------------------------------

def q7_query_diseases_by_treatment(
    g: Graph,
    treatment_text: str,
):
    """
    Q7: Given a treatment name (substring), list diseases that are
        treated with a treatment whose label contains that text.

    Example use:
        treatment_text = "antibiotic"
        treatment_text = "Ibuprofen"
    """
    print("\n=== Q7: Diseases treated with a given treatment ===")
    print(f"Treatment text filter (substring, case-insensitive): "
          f"{treatment_text!r}\n")

    q = f"""
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?d_label ?system_label ?t_label
    WHERE {{
      ?d med:treatedWith ?t .
      ?d rdfs:label ?d_label .
      ?t rdfs:label ?t_label .

      OPTIONAL {{
        ?d med:affectsSystem ?s .
        ?s rdfs:label ?system_label .
      }}

      FILTER(CONTAINS(
                LCASE(STR(?t_label)),
                LCASE("{treatment_text}")
             ))
    }}
    ORDER BY ?d_label ?t_label
    """

    results = g.query(q)

    if not results:
        print("No diseases found for this treatment filter.\n")
        return

    print(f"{'Disease':50s}  {'Body system':35s}  {'Treatment'}")
    print("-" * 110)
    for row in results:
        d_label: Literal      = row.d_label
        system_label: Literal = row.system_label
        t_label: Literal      = row.t_label

        d_s = str(d_label)
        s_s = str(system_label) if system_label else "Unknown / not classified"
        t_s = str(t_label)

        print(f"{d_s:50s}  {s_s:35s}  {t_s}")
    print()

# -------------------------------------------------------------------
# Q8: Differential diagnosis (fever, cough, headache)
# -------------------------------------------------------------------

def q8_differential_diagnosis(g: Graph) -> None:
    """
    Q8: Differential diagnosis for a fixed symptom set:
        ['fever', 'cough', 'headache'].

    Pure SPARQL: count how many of these symptoms each disease has.
    """
    print("\n=== Q8: Differential diagnosis (fever, cough, headache) ===")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?d_label ?system_label (COUNT(DISTINCT ?s) AS ?matchCount)
    WHERE {
      ?d a med:Disease ;
         med:hasSymptom ?s ;
         rdfs:label ?d_label .

      ?s rdfs:label ?s_label .

      OPTIONAL {
        ?d med:affectsSystem ?sys .
        ?sys rdfs:label ?system_label .
      }

      FILTER( LCASE(STR(?s_label)) IN ("fever", "cough", "headache") )
    }
    GROUP BY ?d ?d_label ?system_label
    HAVING(COUNT(DISTINCT ?s) > 0)
    ORDER BY DESC(?matchCount) ?d_label
    LIMIT 20
    """

    results = g.query(query)

    print("Disease                                   System                              #Matches")
    print("-------------------------------------------------------------------------------------------------")
    for row in results:
        d_label = str(row.d_label)
        system_label = str(row.system_label) if row.system_label else "Unknown / not classified"
        match_count = int(row.matchCount.toPython())
        print(f"{d_label:<40} {system_label:<35} {match_count}")


# -------------------------------------------------------------------
# Q9: Explanation for disease (more optimized SPARQL)
# -------------------------------------------------------------------

def q9_explain_disease(g: Graph, target_label: str = "Pneumonia") -> None:
    """
    Q9 (SPARQL-based, optimized):

    Given a disease label (e.g. 'Pneumonia'), explain it:
      - class types (Disease / ChronicDisease / InfectiousDisease)
      - body system
      - symptoms

    Strategy:
      1) SPARQL #1: Find the disease URI by exact label + fetch its types and system.
      2) SPARQL #2: Given that URI, fetch all symptom labels.

    This avoids FILTER with string functions and keeps queries small.
    """

    print(f"\n=== Q9: Explanation for disease: {target_label} ===")

    # We know labels are created as Literal(..., lang="en"), so we match "@en"
    safe_label = target_label.replace('"', '\\"')

    # -------- Query 1: get disease URI + types + system --------
    query1 = f"""
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?system_label ?type_label ?d_label
    WHERE {{
      ?d a med:Disease ;
         rdfs:label "{safe_label}"@en .

      OPTIONAL {{
        ?d med:affectsSystem ?sys .
        ?sys rdfs:label ?system_label .
      }}

      OPTIONAL {{
        ?d a ?t .
        VALUES ?t {{ med:Disease med:ChronicDisease med:InfectiousDisease }}
        ?t rdfs:label ?type_label .
      }}

      # ?d_label is just to print the canonical label we matched
      BIND("{safe_label}"@en AS ?d_label)
    }}
    """

    rows1 = list(g.query(query1))

    if not rows1:
        print(f"No disease found with label '{target_label}'.")
        return

    # Aggregate type(s) and system(s) from query1
    types = set()
    systems = set()
    d_uri = None
    d_name = None

    for row in rows1:
        if row.d:
            d_uri = row.d
        if row.d_label:
            d_name = str(row.d_label)
        if row.system_label:
            systems.add(str(row.system_label))
        if row.type_label:
            types.add(str(row.type_label))

    # Safety check
    if d_uri is None:
        print(f"Internal error: disease URI not found for '{target_label}'.")
        return

    # -------- Query 2: get symptoms for that specific disease --------
    query2 = f"""
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?sym_label
    WHERE {{
      <{d_uri}> med:hasSymptom ?s .
      ?s rdfs:label ?sym_label .
    }}
    """

    rows2 = list(g.query(query2))
    symptoms = sorted({str(r.sym_label) for r in rows2 if r.sym_label})

    # -------- Pretty print the explanation --------
    # Remove '@en' if any (since we forced label with lang="en")
    if d_name and d_name.endswith("@en"):
        d_name = d_name[:-3]

    print(f"Disease: {d_name or target_label}")

    if types:
        print("Types  : " + ", ".join(sorted(types)))
    else:
        print("Types  : (no explicit type labels found, but it is a med:Disease)")

    if systems:
        print("System : " + ", ".join(sorted(systems)))
    else:
        print("System : Unknown / not classified")

    if symptoms:
        print("\nSymptoms linked in KG:")
        for s in symptoms:
            print(f" - {s}")
    else:
        print("\nSymptoms linked in KG: none")



# -------------------------------------------------------------------
# Q10: ASK query – boolean check
# -------------------------------------------------------------------

def q10_ask_disease_has_symptom(
    g: Graph,
    disease_label: str = "Pneumonia",
    symptom_label: str = "fever",
) -> None:
    """
    Q10 (ASK):

    Demonstrate ASK queries, which return a single boolean answer.

    Example question:
      "Does <disease_label> have symptom <symptom_label>?"
    """
    print("\n=== Q10: ASK – does disease have a given symptom? ===")
    print(f"Disease : {disease_label!r}")
    print(f"Symptom : {symptom_label!r}\n")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    ASK {
      ?d a med:Disease ;
         rdfs:label ?dLabel ;
         med:hasSymptom ?s .

      ?s rdfs:label ?sLabel .

      # Case-insensitive match on labels (works for plain strings or langStrings)
      FILTER(LCASE(STR(?dLabel)) = LCASE(STR(?dSearch)))
      FILTER(LCASE(STR(?sLabel)) = LCASE(STR(?sSearch)))
    }
    """

    res = g.query(
        query,
        initBindings={
            "dSearch": Literal(disease_label),
            "sSearch": Literal(symptom_label),
        },
    )

    answer = bool(getattr(res, "askAnswer", False))
    print(f"ASK result: {answer}\n")


# -------------------------------------------------------------------
# Q11: CONSTRUCT query – build a small RDF subgraph
# -------------------------------------------------------------------

def q11_construct_disease_subgraph(
    g: Graph,
    disease_label: str = "Pneumonia",
    max_preview_lines: int = 40,
) -> None:
    """
    Q11 (CONSTRUCT):

    Demonstrate CONSTRUCT queries, which return an RDF graph as output.
    Here we build a compact subgraph describing a disease:
      - selected types (Disease / ChronicDisease / InfectiousDisease)
      - affectsSystem (and system label)
      - hasSymptom (and symptom labels)
      - treatedWith (and treatment labels)

    The result is printed as a Turtle snippet for readability.
    """
    print("\n=== Q11: CONSTRUCT – build a disease subgraph (RDF output) ===")
    print(f"Target disease label: {disease_label!r}\n")

    query = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    CONSTRUCT {
      ?d a ?t ;
         rdfs:label ?dLabel .

      ?d med:affectsSystem ?sys .
      ?sys rdfs:label ?sysLabel .

      ?d med:hasSymptom ?s .
      ?s rdfs:label ?sLabel .

      ?d med:treatedWith ?trt .
      ?trt rdfs:label ?trtLabel .
    }
    WHERE {
      ?d a med:Disease ;
         rdfs:label ?dLabel .

      FILTER(LCASE(STR(?dLabel)) = LCASE(STR(?targetLabel)))

      OPTIONAL {
        ?d a ?t .
        VALUES ?t { med:Disease med:ChronicDisease med:InfectiousDisease }
      }

      OPTIONAL {
        ?d med:affectsSystem ?sys .
        OPTIONAL { ?sys rdfs:label ?sysLabel . }
      }

      OPTIONAL {
        ?d med:hasSymptom ?s .
        OPTIONAL { ?s rdfs:label ?sLabel . }
      }

      OPTIONAL {
        ?d med:treatedWith ?trt .
        OPTIONAL { ?trt rdfs:label ?trtLabel . }
      }
    }
    """

    res = g.query(query, initBindings={"targetLabel": Literal(disease_label)})
    
    #print("\nConstructed RDF subgraph:", res)

    out_g = getattr(res, "graph", None)
    if out_g is None:
        try:
            out_g = res
        except Exception:
            out_g = None

    if out_g is None:
        print("CONSTRUCT returned no graph output (unexpected).\n")
        return

    # Bind namespaces for pretty Turtle output
    try:
        out_g.bind("med", MED)
        out_g.bind("rdfs", RDFS)
        out_g.bind("rdf", RDF)
    except Exception:
        pass

    ttl = out_g.serialize(format="turtle")
    lines = ttl.splitlines()
    print(f"Constructed triples: {len(out_g)}")
    print("Turtle preview:")
    print("-" * 75)
    for ln in lines[:max_preview_lines]:
        print(ln)
    if len(lines) > max_preview_lines:
        print(f"... ({len(lines) - max_preview_lines} more lines)")
    print()



# -------------------------------------------------------------------
# Q12: DESCRIBE query – resource description (implementation-defined)
# -------------------------------------------------------------------

def _find_disease_uri_by_label(g: Graph, label: str):
    """
    Helper: find a disease URI by (case-insensitive) rdfs:label match.
    Returns an rdflib term (URIRef) or None.
    """
    q = """
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?d ?dLabel
    WHERE {
      ?d a med:Disease ;
         rdfs:label ?dLabel .
      FILTER(LCASE(STR(?dLabel)) = LCASE(STR(?targetLabel)))
    }
    LIMIT 1
    """
    rows = list(g.query(q, initBindings={"targetLabel": Literal(label)}))
    if not rows:
        return None
    return rows[0].d


def q12_describe_disease(
    g: Graph,
    disease_label: str = "Pneumonia",
    max_preview_lines: int = 40,
) -> None:
    """
    Q12 (DESCRIBE):

    Demonstrate DESCRIBE queries, which return an RDF graph describing
    a resource. The exact content is store/engine-dependent, so we treat
    this as a *convenience* query rather than a fully deterministic one.

    We first resolve the disease URI by label, then DESCRIBE that URI.
    """
    print("\n=== Q12: DESCRIBE – describe a disease resource (RDF output) ===")
    print(f"Target disease label: {disease_label!r}\n")

    d_uri = _find_disease_uri_by_label(g, disease_label)
    # print(f"Resolved disease URI: {d_uri}\n")
    if d_uri is None:
        print(f"No disease found with label {disease_label!r}.\n")
        return

    query = f"""
    PREFIX med:  <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    DESCRIBE <{d_uri}>
    """

    res = g.query(query)
    out_g = getattr(res, "graph", None) or res

    ttl = out_g.serialize(format="turtle")
    lines = ttl.splitlines()
    print(f"Disease URI        : {d_uri}")
    print(f"Described triples  : {len(out_g)}")
    print("Turtle preview:")
    print("-" * 75)
    
    # Print the first max_preview_lines lines of the Turtle output
    for ln in lines[:max_preview_lines]:
        print(ln)
    if len(lines) > max_preview_lines:
        print(f"... ({len(lines) - max_preview_lines} more lines)")
    print()


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main() -> None:
    """
    Convenience entry point to run all queries in sequence.

    For the exam/demo you can:
      - run everything once (for exploration), or
      - comment/uncomment individual queries to focus on a subset.
    """
    g = load_graph()

    # Q1: overview of diseases, categories, and systems
    q1_list_diseases_with_category_and_system(g, limit=10)

    # Q2: single-symptom lookup (you can change "fever" to test other labels)
    q2_diseases_by_symptom(g, symptom_label="fever")

    # Q3: multi-symptom diagnosis-like ranking
    q3_rank_diseases_by_symptom_set(g, symptoms=["fever", "cough", "headache"])

    # Q4: body-system statistics
    q4_counts_by_system_and_chronic(g)

    # Q5: examples of diseases that share symptoms
    q5_disease_pairs_sharing_symptom(g, limit=20)

    # Q6: chronic diseases in a body system with min symptoms
    q6_query_chronic_by_body_system(
        g,
        body_system_label="Respiratory system",
        min_symptoms=3,
    )

    # Q7: diseases treated with treatments whose label contains text
    q7_query_diseases_by_treatment(
        g,
        treatment_text="antibiotic",
    )

    # Q8: differential diagnosis
    q8_differential_diagnosis(g)

    # Q9: explain a disease
    q9_explain_disease(g, target_label="Pneumonia")  # Tuberculosis

    # Q10: ASK (boolean check)
    q10_ask_disease_has_symptom(g, disease_label="Pneumonia", symptom_label="fever")

    # Q11: CONSTRUCT (RDF subgraph output)
    q11_construct_disease_subgraph(g, disease_label="Pneumonia")

    # Q12: DESCRIBE (RDF description output)
    q12_describe_disease(g, disease_label="Pneumonia")


if __name__ == "__main__":
    main()
