# !/usr/bin/env python3
# reading_owlrl_large.py

from __future__ import annotations

from pathlib import Path

from rdflib import Graph
from owlrl import DeductiveClosure, OWLRL_Semantics

from src import config


def main() -> None:
    print("=======================================================")
    print("OWL-RL Reasoning (large medical KG)")
    print("=======================================================\n")

    schema_path: Path = config.SCHEMA_TTL
    data_path: Path = config.KG_RAW_TTL

    ext_path: Path = config.ONTOLOGY_DIR / "owlrl_extensions.ttl"
    out_path: Path = config.PROCESSED_DIR / "data_medical_large_owlrl_inferred.ttl"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Instance data not found: {data_path}")
    if not ext_path.exists():
        raise FileNotFoundError(
            f"OWL extension file not found: {ext_path}\n"
            "Create it at: data/ontology/owlrl_extensions.ttl"
        )

    # Load schema + extensions + instance data into one graph
    g = Graph()
    print(f"[INFO] Loading schema from: {schema_path}")
    g.parse(str(schema_path), format="turtle")

    print(f"[INFO] Loading OWL extensions from: {ext_path}")
    g.parse(str(ext_path), format="turtle")

    print(f"[INFO] Loading instance data from: {data_path}")
    g.parse(str(data_path), format="turtle")

    print(f"[INFO] Triples BEFORE OWL-RL: {len(g)}")
    print("[INFO] Running OWL-RL closure (this may take a moment)...")

    DeductiveClosure(OWLRL_Semantics).expand(g)

    print(f"[INFO] Triples AFTER  OWL-RL: {len(g)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    print(f"[INFO] OWL-RL inferred KG written to: {out_path}")

    # Quick sanity check: does OWL-defined class appear?
    q = """
PREFIX med:  <http://example.org/medkg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT (COUNT(?d) AS ?n)
WHERE {
  ?d a med:RespiratoryChronicDisease .
}
"""
    try:
        rows = list(g.query(q))
        n = int(rows[0][0]) if rows else 0
        print(f"[CHECK] Instances of med:RespiratoryChronicDisease (OWL-RL): {n}")
    except Exception as e:
        print(f"[WARN] Could not run sanity query: {e}")


if __name__ == "__main__":
    main()
