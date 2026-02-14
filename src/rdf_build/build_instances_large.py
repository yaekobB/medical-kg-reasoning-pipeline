"""
build_instances_large.py

Build the large medical knowledge graph (instances) from the processed CSV.

Input:
    - data/ontology/schema_medical_large.ttl  (ontology/schema)
    - data/processed/diseases_large.csv       (processed dataset from preprocess_large.py)

Output:
    - data/processed/data_medical_large.ttl   (RDF graph: schema + instances)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import os
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # .../medical-kg-reasoning

DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_ONTOLOGY = DATA_DIR / "ontology"

ONTOLOGY_FILE = DATA_ONTOLOGY / "schema_medical_large.ttl"
CSV_FILE = DATA_PROCESSED / "diseases_large.csv"
OUT_TTL_FILE = DATA_PROCESSED / "data_medical_large.ttl"

MED = Namespace("http://example.org/medkg#")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _slugify(s: str) -> str:
    """
    Turn a label into a safe URI suffix: lowercase, hyphens, no weird chars.
    """
    if not s:
        return ""

    s = str(s).strip().lower()
    s = s.replace(" ", "-")
    for ch in [",", ";", ":", "(", ")", "[", "]", "/", "\\", "'", '"']:
        s = s.replace(ch, "")
    return s


# body_system string -> BodySystem individual name in ontology
BODY_SYSTEM_URI_MAP: Dict[str, str] = {
    "cardiovascular": "CardiovascularSystem",
    "respiratory": "RespiratorySystem",
    "endocrine-metabolic": "EndocrineMetabolicSystem",
    "digestive-hepatic": "DigestiveHepaticSystem",
    "nervous": "NervousSystem",
    "musculoskeletal": "MusculoskeletalSystem",
    "dermatologic": "DermatologicSystem",
    "genitourinary-reproductive": "GenitourinaryReproductiveSystem",
    "ophthalmologic": "OphthalmologicSystem",
    "ent": "ENTSystem",
    "hematologic-oncologic": "HematologicOncologicSystem",
    "immune-rheumatologic": "ImmuneRheumatologicSystem",
    "psychiatric": "PsychiatricSystem",
    "infectious-systemic": "InfectiousSystemicCategory",
    "toxicologic": "ToxicologicCategory",
    "oral-dental": "OralDentalSystem",
    "systemic-other": "SystemicOtherCategory",
    "unknown": "UnknownSystem",
}


# -------------------------------------------------------------------
# Main build function
# -------------------------------------------------------------------

def build_instances() -> None:
    """
    Load ontology + processed CSV and create the large instance graph.
    """

    # ---- 1. Load ontology ----
    if not ONTOLOGY_FILE.exists():
        raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_FILE}")
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Processed CSV not found: {CSV_FILE}")

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    g = Graph() # Create RDF graph (empty initially)
    print(f"[INFO] Loading ontology from: {ONTOLOGY_FILE}")
    g.parse(ONTOLOGY_FILE, format="turtle") # Load ontology into graph
    g.bind("med", MED) # Bind the MED namespace i.e. It tells rdflib: When you output Turtle, use the prefix med: for anything in the MED namespace.
                                   # So instead of printing ugly full URIs like: <http://example.org/medkg#Disease_D216>, it prints med:Disease_D216

    # ---- 2. Load processed CSV ----
    print(f"[INFO] Loading processed dataset from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    expected_cols = {
        "disease_id",
        "disease_name",
        "contagious",
        "chronic",
        "category",
        "body_system",
        "symptoms",
        "treatments",
        "num_symptoms",
        "num_treatments",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Processed CSV is missing expected columns: {missing}")

    print(f"[INFO] Processed rows: {len(df)}")

    # caches: symptom/treatment string -> URI suffix
    symptom_cache: Dict[str, str] = {}
    treatment_cache: Dict[str, str] = {}

    disease_count = 0
    symptom_edge_count = 0
    treatment_edge_count = 0
    system_edge_count = 0

    # ---- 3. Iterate over diseases ----
    for _, row in df.iterrows():
        disease_id = str(row["disease_id"]).strip()
        disease_name = str(row["disease_name"]).strip()

        contagious = bool(row["contagious"])
        chronic = bool(row["chronic"])
        category = str(row["category"]).strip()
        body_system = str(row["body_system"]).strip().lower()

        symptoms_str = "" if pd.isna(row["symptoms"]) else str(row["symptoms"])
        treatments_str = "" if pd.isna(row["treatments"]) else str(row["treatments"])

        # --- Disease individual ---
        disease_uri = MED[f"Disease_{disease_id}"]

        # IMPORTANT:
        # We do NOT always assert med:Disease.
        # For ChronicDisease / InfectiousDisease we let RDFS infer med:Disease
        # via rdfs:subClassOf in the ontology.
        if category == "ChronicDisease":
            # Only the subclass type; med:Disease will be inferred.
            g.add((disease_uri, RDF.type, MED.ChronicDisease))
        elif category == "InfectiousDisease":
            # Only the subclass type; med:Disease will be inferred.
            g.add((disease_uri, RDF.type, MED.InfectiousDisease))
        else:
            # Plain Disease (or unknown) – here we assert med:Disease directly.
            g.add((disease_uri, RDF.type, MED.Disease))
        
        # Add disease attributes (label + datatype properties)
        g.add((disease_uri, RDFS.label, Literal(disease_name, lang="en")))
        g.add((disease_uri, MED.diseaseCode, Literal(disease_id, datatype=XSD.string)))
        g.add((disease_uri, MED.isChronic, Literal(chronic, datatype=XSD.boolean)))
        g.add((disease_uri, MED.isContagious, Literal(contagious, datatype=XSD.boolean)))

        # --- Body system link ---
        if body_system:
            bs_key = body_system.lower()
            if bs_key in BODY_SYSTEM_URI_MAP:
                bs_ind_name = BODY_SYSTEM_URI_MAP[bs_key]
            else:
                bs_ind_name = BODY_SYSTEM_URI_MAP["unknown"]

            bs_uri = MED[bs_ind_name]

            # Safety: assert that this node is a BodySystem individual
            g.add((bs_uri, RDF.type, MED.BodySystem))

            g.add((disease_uri, MED.affectsSystem, bs_uri)) # Add affectsSystem edge, the triple
            system_edge_count += 1  # Increment affectsSystem edge count

        # --- Symptoms ---
        if symptoms_str.strip():
            for sym in symptoms_str.split(","):
                sym_clean = sym.strip()
                if not sym_clean:
                    continue

                if sym_clean in symptom_cache:
                    sym_suffix = symptom_cache[sym_clean]
                else:
                    sym_suffix = _slugify(sym_clean)
                    if not sym_suffix:
                        continue
                    symptom_uri = MED[f"Symptom_{sym_suffix}"]
                    g.add((symptom_uri, RDF.type, MED.Symptom))
                    g.add(
                        (symptom_uri, RDFS.label,
                         Literal(sym_clean.replace("-", " "), lang="en"))
                    )
                    symptom_cache[sym_clean] = sym_suffix

                symptom_uri = MED[f"Symptom_{symptom_cache[sym_clean]}"]
                g.add((disease_uri, MED.hasSymptom, symptom_uri)) # Add hasSymptom edge, the triple
                symptom_edge_count += 1  # Increment hasSymptom edge count

        # --- Treatments ---
        if treatments_str.strip():
            for tr in treatments_str.split(","):
                tr_clean = tr.strip()
                if not tr_clean:
                    continue

                if tr_clean in treatment_cache:
                    tr_suffix = treatment_cache[tr_clean]
                else:
                    tr_suffix = _slugify(tr_clean)
                    if not tr_suffix:
                        continue
                    treatment_uri = MED[f"Treatment_{tr_suffix}"]
                    g.add((treatment_uri, RDF.type, MED.Treatment)) # Add Treatment type triple
                    g.add(
                        (treatment_uri, RDFS.label,
                         Literal(tr_clean.replace("-", " "), lang="en")) 
                    )
                    treatment_cache[tr_clean] = tr_suffix

                treatment_uri = MED[f"Treatment_{treatment_cache[tr_clean]}"] 
                g.add((disease_uri, MED.treatedWith, treatment_uri)) # Add treatedWith edge, the triple
                treatment_edge_count += 1   # Increment treatedWith edge count

        disease_count += 1 # Increment disease count

    # ---- 4. Serialize ----
    #print(f"[INFO] Built RDF graph with {len(g)} triples.")
    #print(f"[INFO] Diseases             : {disease_count}")
    #print(f"[INFO] hasSymptom edges     : {symptom_edge_count}")
    #print(f"[INFO] treatedWith edges    : {treatment_edge_count}")
    #print(f"[INFO] affectsSystem edges  : {system_edge_count}")
    
        # ---- 4. Serialize ----
    def count_edges(prop):
        return sum(1 for _ in g.triples((None, prop, None)))

    true_symptom_edges = count_edges(MED.hasSymptom)
    true_treatment_edges = count_edges(MED.treatedWith)
    true_system_edges = count_edges(MED.affectsSystem)

    print(f"[INFO] Built RDF graph with {len(g)} triples.")
    print(f"[INFO] Diseases             : {disease_count}")
    print(f"[INFO] hasSymptom edges     : {true_symptom_edges}")
    print(f"[INFO] treatedWith edges    : {true_treatment_edges}")
    print(f"[INFO] affectsSystem edges  : {true_system_edges}")


    g.serialize(destination=str(OUT_TTL_FILE), format="turtle")
    print(f"[INFO] Written instance KG to: {OUT_TTL_FILE}")


if __name__ == "__main__":
    build_instances()
