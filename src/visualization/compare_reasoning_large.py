from __future__ import annotations

"""
Compare RAW vs REASONED knowledge graphs for the large medical KG.

Run from the project root:

    python -m src.visualization.compare_reasoning_large

This script will:

  1. Load:
       - RAW      : data/processed/data_medical_large.ttl
       - REASONED : data/processed/data_medical_large_inferred.ttl

  2. Compare global statistics:
       - #triples
       - #instances of key classes:
           med:Disease
           med:ChronicDisease
           med:InfectiousDisease
           med:Symptom
           med:Treatment
           med:BodySystem
       - For each class, how many instances appear
         ONLY in the reasoned graph (i.e. inferred).

  3. Compare object properties:
       - med:hasSymptom
       - med:treatedWith
       - med:affectsSystem

  4. Inspect **rdf:type** assertions:
       - total number of rdf:type triples in RAW vs REASONED
       - #distinct types used in RAW vs REASONED

  5. Inspect inferred-only triples (REASONED minus RAW):
       - how many triples are added by reasoning
       - which predicates appear most often
       - a small human-readable sample of inferred triples

  6. Disease category subset checks:
       - counts of Disease / ChronicDisease / InfectiousDisease
       - overlaps (Chronic ∧ Disease, Infectious ∧ Disease)
       - violations (Chronic ∧ ¬Disease, Infectious ∧ ¬Disease)

  7. Run reasoning-sensitive SPARQL:
       - Body-system distribution via ?d a med:Disease

  8. Run reasoning-insensitive SPARQL:
       - Diseases that have symptom 'fever' (label contains 'fever')

The goal is to show clearly:
  - what RDFS reasoning ADDS
  - which analyses change, and which do not.
"""

from pathlib import Path
from typing import Dict, Set, Tuple
from collections import Counter

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal

# ---------------------------------------------------------------------------
# Paths and namespaces
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TTL = PROCESSED_DIR / "data_medical_large.ttl"
INF_TTL = PROCESSED_DIR / "data_medical_large_inferred.ttl"

MED = Namespace("http://example.org/medkg#")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_graph(path: Path) -> Graph:
    """Load an RDF graph from a Turtle file."""
    g = Graph()
    print(f"[INFO] Loading graph from: {path}")
    g.parse(path, format="turtle")
    print(f"[INFO] Loaded {len(g)} triples.\n")
    return g


def get_instances(g: Graph, cls: URIRef) -> Set[URIRef]:
    """Return the set of resources that have rdf:type = cls."""
    return set(g.subjects(RDF.type, cls))


def summarize_types(
    g_raw: Graph, g_inf: Graph
) -> Dict[URIRef, Tuple[int, int, int]]:
    """
    For each key class, compute:

      raw_count, inf_count, inferred_only

    where:
      inferred_only = #instances that are typed in the inferred graph
                      but NOT in the raw graph (for that class).
    """
    classes = [
        MED.Disease,
        MED.ChronicDisease,
        MED.InfectiousDisease,
        MED.Symptom,
        MED.Treatment,
        MED.BodySystem,
    ]

    summary: Dict[URIRef, Tuple[int, int, int]] = {}

    for cls in classes:
        raw_inst = get_instances(g_raw, cls)
        inf_inst = get_instances(g_inf, cls)

        raw_count = len(raw_inst)
        inf_count = len(inf_inst)
        inferred_only = len(inf_inst - raw_inst)

        summary[cls] = (raw_count, inf_count, inferred_only)

    return summary


def count_property_edges(g: Graph, prop: URIRef) -> int:
    """Count how many triples use the given property."""
    return sum(1 for _ in g.triples((None, prop, None)))


# ---------------------------------------------------------------------------
# SPARQL queries for comparison
# ---------------------------------------------------------------------------


def body_system_distribution_by_disease(g: Graph) -> Dict[str, int]:
    """
    Count diseases per body system, using:

        ?d a med:Disease ;
           med:affectsSystem ?s .
        ?s rdfs:label ?label .

    This query is intended to be reasoning-sensitive, because many diseases
    can become 'med:Disease' via rdfs:subClassOf / domain inference.
    """
    q = """
    PREFIX med: <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?system_label (COUNT(DISTINCT ?d) AS ?n)
    WHERE {
      ?d a med:Disease .
      ?d med:affectsSystem ?s .
      ?s rdfs:label ?system_label .
    }
    GROUP BY ?system_label
    ORDER BY DESC(?n)
    """

    results = g.query(q)
    counts: Dict[str, int] = {}
    for row in results:
        label: Literal = row.system_label
        n: Literal = row.n
        counts[str(label)] = int(n.toPython())
    return counts


def diseases_with_fever(g: Graph) -> Set[str]:
    """
    Retrieve disease labels that have a symptom whose label contains 'fever'
    (case-insensitive). This query is mostly independent of reasoning,
    because it relies on the explicit hasSymptom links.
    """
    q = """
    PREFIX med: <http://example.org/medkg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?d_label
    WHERE {
      ?d med:hasSymptom ?s .
      ?d rdfs:label ?d_label .
      ?s rdfs:label ?s_label .
      FILTER(CONTAINS(LCASE(STR(?s_label)), "fever"))
    }
    ORDER BY ?d_label
    """

    results = g.query(q)
    return {str(row.d_label) for row in results}


def rdf_type_counts(g: Graph) -> Tuple[int, int]:
    """
    Return:
      total_count, distinct_type_count
    for rdf:type assertions in the given graph.
    """
    q_total = """
    SELECT (COUNT(*) AS ?n)
    WHERE {
      ?s a ?c .
    }
    """
    q_distinct = """
    SELECT (COUNT(DISTINCT ?c) AS ?n)
    WHERE {
      ?s a ?c .
    }
    """

    res_total = g.query(q_total)
    total = int(next(iter(res_total))[0].toPython())

    res_distinct = g.query(q_distinct)
    distinct = int(next(iter(res_distinct))[0].toPython())

    return total, distinct


# ---------------------------------------------------------------------------
# Inferred-only triple analysis
# ---------------------------------------------------------------------------


def short_iri(term) -> str:
    """
    Produce a short, readable representation of a URIRef or Literal.
    """
    if isinstance(term, URIRef):
        s = str(term)
        if s.startswith(str(MED)):
            return "med:" + s.split("#", 1)[-1]
        if s == str(RDF.type):
            return "rdf:type"
        if s == str(RDFS.subClassOf):
            return "rdfs:subClassOf"
        if s == str(RDFS.label):
            return "rdfs:label"
        return s
    elif isinstance(term, Literal):
        txt = str(term)
        if len(txt) > 40:
            txt = txt[:37] + "..."
        return f'"{txt}"'
    else:
        return str(term)


def analyze_inferred_triples(g_raw: Graph, g_inf: Graph) -> None:
    """
    Compute and print statistics about triples that exist only
    in the REASONED graph.
    """
    raw_triples = set(g_raw)
    inf_triples = set(g_inf)

    inferred_only = inf_triples - raw_triples

    print("=== Inferred-only triples (REASONED minus RAW) ===")
    print(f"Total inferred-only triples: {len(inferred_only)}\n")

    if not inferred_only:
        print("No triples are unique to the REASONED graph.\n")
        return

    # Count predicates
    pred_counts = Counter(p for (_, p, _) in inferred_only)

    print("Top predicates among inferred-only triples:")
    max_preds = 10
    for pred, cnt in pred_counts.most_common(max_preds):
        print(f"- {short_iri(pred):25s}: {cnt}")
    if len(pred_counts) > max_preds:
        print(f"... (total distinct predicates: {len(pred_counts)})")
    print()

    # Show a small sample of inferred triples
    #print("Sample inferred-only triples:")
    #sample_size = 15
    # sort for stable, readable output
    #sorted_triples = sorted(
    #    inferred_only,
    #    key=lambda t: (str(t[1]), str(t[0]), str(t[2])),
    #)
    #for i, (s, p, o) in enumerate(sorted_triples[:sample_size], start=1):
    #    print(f"{i:2d}. {short_iri(s)}  {short_iri(p)}  {short_iri(o)}")
    #if len(inferred_only) > sample_size:
    #    print(f"... ({len(inferred_only) - sample_size} more triples)")
    #print()


# ---------------------------------------------------------------------------
# Disease category / subset checks
# ---------------------------------------------------------------------------


def disease_category_stats(g: Graph) -> Dict[str, int]:
    """
    Compute counts for Disease / Chronic / Infectious and their overlaps.
    Returns a dict with keys:
      - disease
      - chronic
      - infectious
      - chronic_and_disease
      - infectious_and_disease
      - chronic_not_disease
      - infectious_not_disease
    """
    def run_scalar(q: str) -> int:
        res = g.query(q)
        val = next(iter(res))[0]
        return int(val.toPython())

    q_disease = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE { ?d a med:Disease . }
    """

    q_chronic = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE { ?d a med:ChronicDisease . }
    """

    q_infectious = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE { ?d a med:InfectiousDisease . }
    """

    q_chronic_and_disease = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE {
      ?d a med:ChronicDisease .
      ?d a med:Disease .
    }
    """

    q_infectious_and_disease = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE {
      ?d a med:InfectiousDisease .
      ?d a med:Disease .
    }
    """

    q_chronic_not_disease = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE {
      ?d a med:ChronicDisease .
      FILTER NOT EXISTS { ?d a med:Disease . }
    }
    """

    q_infectious_not_disease = """
    PREFIX med: <http://example.org/medkg#>
    SELECT (COUNT(DISTINCT ?d) AS ?n)
    WHERE {
      ?d a med:InfectiousDisease .
      FILTER NOT EXISTS { ?d a med:Disease . }
    }
    """

    return {
        "disease": run_scalar(q_disease),
        "chronic": run_scalar(q_chronic),
        "infectious": run_scalar(q_infectious),
        "chronic_and_disease": run_scalar(q_chronic_and_disease),
        "infectious_and_disease": run_scalar(q_infectious_and_disease),
        "chronic_not_disease": run_scalar(q_chronic_not_disease),
        "infectious_not_disease": run_scalar(q_infectious_not_disease),
    }


def print_disease_subset_checks(g_raw: Graph, g_inf: Graph) -> None:
    """
    Print a table comparing Disease / Chronic / Infectious + overlaps
    for RAW vs REASONED graphs, including an overlap column
    (percentage of Chronic/ Infectious that are also Disease).
    """
    stats_raw = disease_category_stats(g_raw)
    stats_inf = disease_category_stats(g_inf)

    def safe_pct(num: int, denom: int) -> str:
        if denom == 0:
            return "n/a"
        return f"{(num / denom) * 100:.1f}%"

    # Precompute overlaps (as %)
    chronic_ov_raw = safe_pct(
        stats_raw["chronic_and_disease"],
        stats_raw["chronic"],
    )
    chronic_ov_inf = safe_pct(
        stats_inf["chronic_and_disease"],
        stats_inf["chronic"],
    )

    infectious_ov_raw = safe_pct(
        stats_raw["infectious_and_disease"],
        stats_raw["infectious"],
    )
    infectious_ov_inf = safe_pct(
        stats_inf["infectious_and_disease"],
        stats_inf["infectious"],
    )

    print("=== Disease category subset checks (RAW vs REASONED) ===")
    print(
        f"{'Row':40s}  {'RAW':>6s}  {'REASONED':>9s}  {'Overlap (RAW/REAS)':>18s}"
    )
    print("-" * 80)

    rows = [
        ("Total med:Disease", "disease", ""),
        ("Total med:ChronicDisease", "chronic", "chronic"),
        ("Total med:InfectiousDisease", "infectious", "infectious"),
        ("Chronic ∧ Disease", "chronic_and_disease", None),
        ("Infectious ∧ Disease", "infectious_and_disease", None),
        ("Chronic ∧ ¬Disease", "chronic_not_disease", None),
        ("Infectious ∧ ¬Disease", "infectious_not_disease", None),
    ]

    for label, key, overlap_kind in rows:
        r = stats_raw[key]
        i = stats_inf[key]

        if overlap_kind == "chronic":
            overlap_str = f"{chronic_ov_raw} / {chronic_ov_inf}"
        elif overlap_kind == "infectious":
            overlap_str = f"{infectious_ov_raw} / {infectious_ov_inf}"
        else:
            overlap_str = ""

        print(f"{label:40s}  {r:6d}  {i:9d}  {overlap_str:>18s}")
    print()



# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------


def print_type_summary(summary: Dict[URIRef, Tuple[int, int, int]]) -> None:
    """Nicely print raw vs reasoned type counts."""
    print("=== Type counts: RAW vs REASONED (and inferred-only) ===")
    print(f"{'Class':35s}  {'RAW':>6s}  {'INF':>6s}  {'INF_ONLY':>9s}")
    print("-" * 65)

    label_map = {
        MED.Disease: "Disease",
        MED.ChronicDisease: "ChronicDisease",
        MED.InfectiousDisease: "InfectiousDisease",
        MED.Symptom: "Symptom",
        MED.Treatment: "Treatment",
        MED.BodySystem: "BodySystem",
    }

    for cls, (raw, inf, inferred_only) in summary.items():
        name = label_map.get(cls, cls.split("#")[-1])
        print(f"{name:35s}  {raw:6d}  {inf:6d}  {inferred_only:9d}")
    print()


def print_body_system_distribution(
    name: str, dist: Dict[str, int], max_rows: int = 15
) -> None:
    """Print body-system counts for a graph."""
    print(f"=== Body system distribution via med:Disease ({name}) ===")
    if not dist:
        print("No med:Disease instances returned by the query.\n")
        return

    # sort by count desc
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    for label, n in items[:max_rows]:
        print(f"- {label}: {n}")
    if len(items) > max_rows:
        print(f"... (total systems: {len(items)})")
    print()


def print_fever_comparison(raw_set: Set[str], inf_set: Set[str]) -> None:
    """Compare diseases with 'fever' symptom in raw vs reasoned graph."""
    print("=== Diseases that have a symptom containing 'fever' ===")
    print(f"RAW graph   : {len(raw_set)} diseases")
    print(f"REASONED    : {len(inf_set)} diseases")

    only_raw = sorted(raw_set - inf_set)
    only_inf = sorted(inf_set - raw_set)

    if not only_raw and not only_inf:
        print("→ The sets are identical (reasoning does not change this query).\n")
        return

    if only_raw:
        print("Diseases only in RAW:")
        for d in only_raw:
            print(f"  - {d}")
    if only_inf:
        print("Diseases only in REASONED:")
        for d in only_inf:
            print(f"  - {d}")
    print()


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Load graphs
    print("=======================================================")
    print("Comparing RAW vs REASONED graphs (large medical KG)")
    print("=======================================================\n")

    g_raw = load_graph(RAW_TTL)
    g_inf = load_graph(INF_TTL)

    # 2. Global triple counts
    print("=== Triple counts ===")
    print(f"RAW graph      : {len(g_raw):6d} triples")
    print(f"REASONED graph : {len(g_inf):6d} triples")
    print(
        f"→ Increase of {len(g_inf) - len(g_raw)} triples "
        f"({(len(g_inf) - len(g_raw)) / len(g_raw) * 100:.1f}% extra)\n"
    )

    # 3. Type counts & inferred-only instances
    type_summary = summarize_types(g_raw, g_inf)
    print_type_summary(type_summary)

    # 4. Property edge counts
    print("=== Object property edge counts ===")
    for prop, label in [
        (MED.hasSymptom, "hasSymptom"),
        (MED.treatedWith, "treatedWith"),
        (MED.affectsSystem, "affectsSystem"),
    ]:
        raw_n = count_property_edges(g_raw, prop)
        inf_n = count_property_edges(g_inf, prop)
        print(f"{label:15s} RAW={raw_n:5d}   REASONED={inf_n:5d}")
    print(
        "Note: counts should usually be identical; "
        "reasoning mostly adds rdf:type triples, not new edges.\n"
    )

    # 5. rdf:type assertion statistics
    total_raw, distinct_raw = rdf_type_counts(g_raw)
    total_inf, distinct_inf = rdf_type_counts(g_inf)

    print("=== rdf:type assertion counts ===")
    print(
        f"RAW      : {total_raw:6d} rdf:type triples, "
        f"{distinct_raw:4d} distinct types"
    )
    print(
        f"REASONED : {total_inf:6d} rdf:type triples, "
        f"{distinct_inf:4d} distinct types"
    )
    print(
        f"→ Reasoning adds {total_inf - total_raw} rdf:type triples "
        f"and {distinct_inf - distinct_raw} additional distinct types "
        f"(if positive).\n"
    )

    # 6. Inferred-only triples (REASONED minus RAW)
    analyze_inferred_triples(g_raw, g_inf)

    # 7. Disease category subset checks
    print_disease_subset_checks(g_raw, g_inf)

    # 8. Reasoning-sensitive SPARQL: body-system distribution
    dist_raw = body_system_distribution_by_disease(g_raw)
    dist_inf = body_system_distribution_by_disease(g_inf)

    print_body_system_distribution("RAW", dist_raw)
    print_body_system_distribution("REASONED", dist_inf)

    # 9. Reasoning-insensitive SPARQL: diseases with 'fever'
    fever_raw = diseases_with_fever(g_raw)
    fever_inf = diseases_with_fever(g_inf)
    print_fever_comparison(fever_raw, fever_inf)

    print("Comparison done.")
    


if __name__ == "__main__":
    main()
