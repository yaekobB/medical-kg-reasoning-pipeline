from __future__ import annotations

"""
Global configuration for the medical-kg-reasoning project.

Centralizes all important paths so they are consistent across scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------

# src/ directory
SRC_DIR = Path(__file__).resolve().parent

# project root: .../medical-kg-reasoning
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ONTOLOGY_DIR = DATA_DIR / "ontology"
VIS_DIR = DATA_DIR / "visualizations"

# Make sure these directories exist (idempotent)
for p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ONTOLOGY_DIR, VIS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Canonical dataset files
# ---------------------------------------------------------------------

# Main raw dataset (you already use this name)
RAW_DISEASES_CSV = RAW_DIR / "diseases_symptoms_main.csv"

# Processed dataset used for the large KG
PROCESSED_DISEASES_CSV = PROCESSED_DIR / "diseases_large.csv"

# Instance KG (before reasoning)
KG_RAW_TTL = PROCESSED_DIR / "data_medical_large.ttl"

# Inferred KG (after reasoning)
KG_INFERRED_TTL = PROCESSED_DIR / "data_medical_large_inferred.ttl"

# Ontology schema
SCHEMA_TTL = ONTOLOGY_DIR / "schema_medical_large.ttl"

# Visualizations output directory
PLOTS_DIR = VIS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# OWL-RL inferred KG (for comparison)
KG_OWLRL_TTL = PROCESSED_DIR / "data_medical_large_owlrl_inferred.ttl"



def print_config_summary() -> None:
    """Small helper to debug paths if needed."""
    print("=== Config summary ===")
    print(f"BASE_DIR            : {BASE_DIR}")
    print(f"DATA_DIR            : {DATA_DIR}")
    print(f"RAW_DIR             : {RAW_DIR}")
    print(f"PROCESSED_DIR       : {PROCESSED_DIR}")
    print(f"ONTOLOGY_DIR        : {ONTOLOGY_DIR}")
    print(f"VIS_DIR             : {VIS_DIR}")
    print(f"RAW_DISEASES_CSV    : {RAW_DISEASES_CSV}")
    print(f"PROCESSED_DISEASES_CSV: {PROCESSED_DISEASES_CSV}")
    print(f"KG_RAW_TTL          : {KG_RAW_TTL}")
    print(f"KG_INFERRED_TTL     : {KG_INFERRED_TTL}")
    print(f"SCHEMA_TTL          : {SCHEMA_TTL}")
    print(f"KG_OWLRL_TTL        : {KG_OWLRL_TTL}")
    print("======================")
