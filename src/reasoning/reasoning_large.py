"""
reasoning_large.py

Apply RDFS reasoning to the LARGE medical knowledge graph.

Input:
    - data/ontology/schema_medical_large.ttl
    - data/processed/data_medical_large.ttl   (instances from build_instances_large.py)

Output:
    - data/processed/data_medical_large_inferred.ttl

The goal is to:
    * expand the graph with all RDFS-entailable triples
      (e.g., subclasses, type propagation, domain/range)
    * show how many new triples we get
    * print some summary statistics by class

This script is the "large" counterpart of the mini-project reasoning demo,
but it is written to be stand-alone and work directly on the large dataset.
"""

from __future__ import annotations

from pathlib import Path
import os

from rdflib import Graph, Namespace, RDF, RDFS
from owlrl import DeductiveClosure, RDFS_Semantics


# -------------------------------------------------------------------
# Paths and namespaces
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # .../medical-kg-reasoning

DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_ONTOLOGY = DATA_DIR / "ontology"

SCHEMA_FILE = DATA_ONTOLOGY / "schema_medical_large.ttl"  # Ontology/schema
DATA_FILE = DATA_PROCESSED / "data_medical_large.ttl"  # RDF graph built 
OUT_FILE = DATA_PROCESSED / "data_medical_large_inferred.ttl"

MED = Namespace("http://example.org/medkg#")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _count_instances(g: Graph, class_uri) -> int:
    """
    Count how many individuals in the graph have rdf:type class_uri.
    """
    return sum(1 for _ in g.subjects(RDF.type, class_uri))


def _print_type_summary(g: Graph) -> None:
    """
    Print a small summary of key classes after reasoning.

    This helps to see the effect of reasoning at a high level.
    """
    print("\n=== Type summary after reasoning ===")

    disease_count = _count_instances(g, MED.Disease)
    chronic_count = _count_instances(g, MED.ChronicDisease)
    infectious_count = _count_instances(g, MED.InfectiousDisease)
    symptom_count = _count_instances(g, MED.Symptom)
    treatment_count = _count_instances(g, MED.Treatment)
    body_system_count = _count_instances(g, MED.BodySystem)

    print(f"  med:Disease             : {disease_count}")
    print(f"  med:ChronicDisease      : {chronic_count}")
    print(f"  med:InfectiousDisease   : {infectious_count}")
    print(f"  med:Symptom             : {symptom_count}")
    print(f"  med:Treatment           : {treatment_count}")
    print(f"  med:BodySystem          : {body_system_count}")


def _print_sample_diseases(g: Graph, limit: int = 10) -> None:
    """
    Print a small sample of Disease individuals with their labels,
    just to inspect what reasoning produced.

    We show at most `limit` examples.
    """
    print(f"\n=== Sample med:Disease individuals (max {limit}) ===")
    count = 0
    for s in g.subjects(RDF.type, MED.Disease):
        label = g.value(s, RDFS.label)
        print(f"  - {s} | label: {label}")
        count += 1
        if count >= limit:
            break


# -------------------------------------------------------------------
# Main reasoning function
# -------------------------------------------------------------------

def run_reasoning_large() -> None:
    """
    Load schema + instance data, apply RDFS reasoning, and save the
    inferred graph to Turtle.

    This function is designed to be run as a script or imported and
    called from elsewhere if needed.
    """
    # 1. Basic checks ------------------------------------------------
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Instance data file not found: {DATA_FILE}")

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    # 2. Load schema + data into one graph ---------------------------
    g_raw = Graph() 
    g_raw.bind("med", MED)

    print(f"[INFO] Loading schema from: {SCHEMA_FILE}")
    g_raw.parse(SCHEMA_FILE, format="turtle") # Load ontology/schema

    print(f"[INFO] Loading instance data from: {DATA_FILE}")
    g_raw.parse(DATA_FILE, format="turtle")

    print(f"[INFO] Triples BEFORE reasoning: {len(g_raw)}")

    # 3. Apply RDFS reasoning using owlrl ----------------------------
    print("[INFO] Running RDFS closure (this may take a moment)...")
    DeductiveClosure(RDFS_Semantics).expand(g_raw) # it runs RDFS rules and adds inferred triples into the same graph.

    print(f"[INFO] Triples AFTER reasoning : {len(g_raw)}")

    # 4. Summary of what we got --------------------------------------
    _print_type_summary(g_raw)
    _print_sample_diseases(g_raw, limit=10)

    # 5. Serialize inferred graph ------------------------------------
    g_raw.serialize(destination=str(OUT_FILE), format="turtle")
    print(f"\n[INFO] Inferred graph written to: {OUT_FILE}")


if __name__ == "__main__":
    run_reasoning_large()
