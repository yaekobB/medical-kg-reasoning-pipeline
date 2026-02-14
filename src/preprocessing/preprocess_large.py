"""
preprocess_large.py

Preprocess the raw Kaggle disease–symptom–treatment dataset into a
canonical form for RDF/RDFS knowledge-graph construction.

Input  (raw):
    data/raw/diseases_symptoms_main.csv

Expected columns (from Kaggle):
    Name, Symptoms, Treatments, Disease_Code, Contagious, Chronic

Output (processed):
    data/processed/diseases_large.csv

Columns:
    disease_id      : string (from Disease_Code)
    disease_name    : string (from Name, stripped)
    contagious      : bool   (from Contagious: TRUE/FALSE → True/False)
    chronic         : bool   (from Chronic: TRUE/FALSE → True/False)
    category        : string (ChronicDisease / InfectiousDisease / Disease)
    body_system     : string (coarse system: respiratory, cardiovascular, ...)
    symptoms        : string (normalized, comma-separated list)
    treatments      : string (normalized, comma-separated list)
    num_symptoms    : int    (number of distinct symptoms)
    num_treatments  : int    (number of distinct treatments)
"""

from __future__ import annotations

import os
import re
import pandas as pd
from pathlib import Path
from typing import List

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # medical-kg-reasoning/
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

RAW_FILE = DATA_RAW / "diseases_symptoms_main.csv"
OUT_FILE = DATA_PROCESSED / "diseases_large.csv"


# -------------------------------------------------------------------
# Helper: split + normalize lists (symptoms, treatments)
# -------------------------------------------------------------------

_CONTINUATION_STARTS = {
    "particularly", "especially", "including", "such", "e.g.", "eg", "i.e.", "ie",
    "in", "on", "of", "with", "without", "and", "or", "when", "during", "after",
    "at", "around", "near", "associated", "often", "like", "typically"
}

_TRAILING_PUNCT = " \t\r\n;:.!?"

def _looks_like_phrase(s: str) -> bool:
    # phrase-like continuation, not a simple symptom word
    return len(s.split()) >= 3

def _smart_split_commas(text: str) -> List[str]:
    raw_parts = [p.strip() for p in str(text).split(",")]
    raw_parts = [p for p in raw_parts if p]

    merged: List[str] = []
    for part in raw_parts:
        token0 = part.strip().split(" ", 1)[0].lower()

        is_continuation = (
            (token0 in _CONTINUATION_STARTS) or
            (part[:1].islower() and _looks_like_phrase(part))
        )

        if merged and is_continuation:
            merged[-1] = merged[-1].rstrip(_TRAILING_PUNCT) + ", " + part
        else:
            merged.append(part)

    return merged

def _split_and_normalize_list(text: str) -> List[str]:
    if pd.isna(text) or not str(text).strip():
        return []

    parts = _smart_split_commas(text)

    items: List[str] = []
    for clean in parts:
        clean = clean.strip()
        clean = re.sub(r"\s+", " ", clean)          # collapse spaces
        clean = clean.strip().strip(").,;:")        # stronger punctuation strip
        clean = clean.strip()

        if not clean:
            continue

        clean_norm = clean.lower().replace(" ", "-")
        items.append(clean_norm)

    seen = set()
    unique_items: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            unique_items.append(it)

    return unique_items


def _bool_from_string(val: str) -> bool:
    """
    Convert TRUE/FALSE or True/False or 1/0 into Python bool.
    Any non-empty, non-FALSE-like string is treated as True.
    """
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", ""):
        return False
    # Fallback: treat unknown non-empty strings as False
    return False


def _derive_category(contagious: bool, chronic: bool) -> str:
    """
    Derive a coarse-grained disease category that will be mapped
    to RDF classes:

        - if chronic      → ChronicDisease
        - elif contagious → InfectiousDisease
        - else            → Disease
    """
    if chronic:
        return "ChronicDisease"
    if contagious:
        return "InfectiousDisease"
    return "Disease"


# -------------------------------------------------------------------
# Body-system mapping (manual + keyword-based)
# -------------------------------------------------------------------

BODY_SYSTEM_MANUAL = {
    # explicit mappings for common diseases (you can extend later)
    "gestational cholestasis": "digestive-hepatic",
    "scabies": "dermatologic",
    "congenital glaucoma": "ophthalmologic",
    "gastroesophageal reflux disease (gerd)": "digestive-hepatic",
    "common cold": "respiratory",
    "pulmonary fibrosis": "respiratory",
    "type 2 diabetes": "endocrine-metabolic",
    "type 1 diabetes": "endocrine-metabolic",
    "asthma": "respiratory",
    "hypertension": "cardiovascular",
    "hypertensive heart disease": "cardiovascular",
    "atrial fibrillation": "cardiovascular",
    "coronary atherosclerosis": "cardiovascular",
    "osteoporosis": "musculoskeletal",
    "osteoarthritis": "musculoskeletal",
    "rheumatoid arthritis": "immune-rheumatologic",
    "migraine": "nervous",
    "chronic migraine": "nervous",
    "parkinson disease": "nervous",
    "psoriasis": "dermatologic",
    "eczema": "dermatologic",
    "urticaria (hives)": "dermatologic",
    "chronic kidney disease": "genitourinary-reproductive",
    "acute kidney injury": "genitourinary-reproductive",
    "kidney stone": "genitourinary-reproductive",
    "urinary tract infection (uti)": "genitourinary-reproductive",
    "endometriosis": "genitourinary-reproductive",
    "uterine fibroids": "genitourinary-reproductive",
    "ovarian torsion": "genitourinary-reproductive",
    "testicular cancer": "genitourinary-reproductive",
    "vaginal yeast infection": "genitourinary-reproductive",
    "autism": "nervous",
    "bipolar disorder": "psychiatric",
    "panic disorder": "psychiatric",
    "panic attack": "psychiatric",
    "anxiety": "psychiatric",
    "dysthymic disorder": "psychiatric",
    "postpartum depression": "psychiatric",
    "attention deficit hyperactivity disorder (adhd)": "psychiatric",
    "chronic fatigue syndrome": "systemic-other",
    "fibromyalgia": "musculoskeletal",
    "plantar fasciitis": "musculoskeletal",
    "carpal tunnel syndrome": "musculoskeletal",
    "rotator cuff injury": "musculoskeletal",
    "sciatica": "musculoskeletal",
    "labyrinthitis": "ent",
    "otitis media": "ent",
    "acute otitis media": "ent",
    "chronic otitis media": "ent",
    "meniere disease": "ent",
    "pharyngitis": "ent",
    "acute sinusitis": "ent",
    "chronic sinusitis": "ent",
    "conjunctivitis due to allergy": "ophthalmologic",
    "retinal detachment": "ophthalmologic",
    "diabetic retinopathy": "ophthalmologic",
    "macular degeneration": "ophthalmologic",
    "vitreous hemorrhage": "ophthalmologic",
    "optic neuritis": "ophthalmologic",
    "pneumonia": "respiratory",
    "tuberculosis": "respiratory",
    "cystic fibrosis": "respiratory",
    "emphysema": "respiratory",
    "pulmonary congestion": "respiratory",
    "pulmonary eosinophilia": "respiratory",
    "gastroenteritis (stomach flu)": "digestive-hepatic",
    "acute pancreatitis": "digestive-hepatic",
    "cirrhosis": "digestive-hepatic",
    "nonalcoholic liver disease (nash)": "digestive-hepatic",
    "liver cancer": "digestive-hepatic",
    "lymphoma": "hematologic-oncologic",
    "leukemia": "hematologic-oncologic",
    "anemia": "hematologic-oncologic",
    "aplastic anemia": "hematologic-oncologic",
    "anemia of chronic disease": "hematologic-oncologic",
    "anemia due to malignancy": "hematologic-oncologic",
    "thrombocytopenia": "hematologic-oncologic",
    "hemophilia": "hematologic-oncologic",
    "von willebrand disease": "hematologic-oncologic",
}

BODY_SYSTEM_KEYWORDS = [
    ("cardiovascular", ["heart", "cardio", "coronary", "aortic", "atrial", "hypertensive", "valve"]),
    ("respiratory", ["lung", "pulmonary", "bronch", "pneumonia", "respiratory", "asthma"]),
    ("endocrine-metabolic", ["diabetes", "thyroid", "pituitary", "parathyroid", "hypercalcemia", "hypocalcemia", "hyperkalemia", "hypoglycemia"]),
    ("digestive-hepatic", ["hepat", "pancreat", "colitis", "bowel", "intestinal", "esophageal", "stomach", "liver", "bile", "chole", "hernia", "gastr", "reflux"]),
    ("nervous", ["neuropathy", "epilepsy", "seizure", "parkinson", "sclerosis", "myasthenia", "syringomyelia", "stroke", "ischemic", "hemorrhage", "headache", "migraine", "narcolepsy"]),
    ("musculoskeletal", ["arthritis", "fracture", "spondyl", "bursitis", "tendon", "ligament", "epicondylitis", "spasm", "myositis", "bone", "osteochondrosis", "sprain"]),
    ("dermatologic", ["dermatitis", "eczema", "psoriasis", "rash", "urticaria", "pemphigus", "acne", "scabies", "fungal infection of the skin", "actinic keratosis", "alopecia", "hyperhidrosis", "warts"]),
    ("genitourinary-reproductive", ["uterine", "ovarian", "endometr", "fibroid", "prostate", "testicle", "kidney", "renal", "bladder", "urinary", "pyelonephritis", "hydrocele", "cryptorchidism", "preeclampsia", "placenta", "pregnancy"]),
    ("ophthalmologic", ["eye", "retina", "retinal", "glaucoma", "macular", "optic", "cornea", "conjunctiv", "cataract"]),
    ("ent", ["ear", "otitis", "meniere", "pharyngitis", "tonsil", "sinusitis", "laryng", "nasal"]),
    ("hematologic-oncologic", ["anemia", "lymphoma", "leukemia", "myeloma", "thrombocyto", "hemophilia", "cancer", "carcinoma", "adenoma", "sarcoma"]),
    ("immune-rheumatologic", ["lupus", "sjögren", "sjÃ¶gren", "vasculitis", "scleroderma", "amyloidosis", "polymyalgia", "rheumatica"]),
    ("psychiatric", ["depression", "schizophrenia", "bipolar", "anxiety", "panic", "autism", "adhd", "insomnia", "eating disorder", "anorexia", "bulimia", "orthorexia"]),
    ("infectious-systemic", ["sepsis", "infection", "infectious", "virus", "viral", "bacterial", "parasitic", "fungal", "fever"]),
    ("toxicologic", ["poison", "overdose", "tox", "intoxication"]),
    ("oral-dental", ["dental", "tooth", "teeth", "gingivitis", "gum disease", "caries", "oral", "mouth"]),
]


def _assign_body_system(disease_name: str) -> str:
    """
    Assign a coarse body_system label to a disease name using:
      1) explicit manual mappings
      2) keyword-based rules
      3) fallback "unknown"
    """
    if not disease_name:
        return "unknown"

    name = disease_name.strip().lower()

    # 1) Manual exact mapping
    if name in BODY_SYSTEM_MANUAL:
        return BODY_SYSTEM_MANUAL[name]

    # 2) Keyword rules
    for system, keywords in BODY_SYSTEM_KEYWORDS:
        for kw in keywords:
            if kw in name:
                return system

    # 3) Fallback
    return "unknown"


# -------------------------------------------------------------------
# Main preprocessing
# -------------------------------------------------------------------

def preprocess():
    """
    Load the raw Kaggle file, normalize, and write the canonical CSV.
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    print(f"[INFO] Loading raw dataset from: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    expected_cols = {"Name", "Symptoms", "Treatments", "Disease_Code", "Contagious", "Chronic"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")

    print(f"[INFO] Raw rows: {len(df)}")

    # Drop duplicate Disease_Code if any
    df = df.drop_duplicates(subset=["Disease_Code"]).reset_index(drop=True)
    print(f"[INFO] Rows after dropping duplicates by Disease_Code: {len(df)}")

    # Build processed DataFrame
    processed = pd.DataFrame()
    processed["disease_id"] = df["Disease_Code"].astype(str).str.strip()
    processed["disease_name"] = df["Name"].astype(str).str.strip()

    # Boolean flags
    processed["contagious"] = df["Contagious"].apply(_bool_from_string)
    processed["chronic"] = df["Chronic"].apply(_bool_from_string)

    # Derived category
    processed["category"] = processed.apply(
        lambda row: _derive_category(row["contagious"], row["chronic"]),
        axis=1,
    )

    # NEW: derive body_system from disease_name
    processed["body_system"] = processed["disease_name"].apply(_assign_body_system)

    # Normalize symptoms
    normalized_symptoms = df["Symptoms"].apply(_split_and_normalize_list)
    processed["symptoms"] = normalized_symptoms.apply(lambda lst: ",".join(lst))
    processed["num_symptoms"] = normalized_symptoms.apply(len)

    # Normalize treatments
    normalized_treatments = df["Treatments"].apply(_split_and_normalize_list)
    processed["treatments"] = normalized_treatments.apply(lambda lst: ",".join(lst))
    processed["num_treatments"] = normalized_treatments.apply(len)

    # Optional: drop rows with zero symptoms
    before = len(processed)
    processed = processed[processed["num_symptoms"] > 0].reset_index(drop=True)
    after = len(processed)
    if after < before:
        print(f"[INFO] Dropped {before - after} rows with no symptoms.")

    # Save
    processed.to_csv(OUT_FILE, index=False)
    print(f"[INFO] Written processed dataset to: {OUT_FILE}")

    # Summary
    print("\n=== Processed dataset summary (first 5 rows) ===")
    print(processed.head(5))
    print("\nCounts by category:")
    print(processed["category"].value_counts())
    print("\nCounts by body_system (top 10):")
    print(processed["body_system"].value_counts().head(10))
    print("\ncontagious=True:", processed["contagious"].sum())
    print("chronic=True   :", processed["chronic"].sum())
    print("Total rows     :", len(processed))


if __name__ == "__main__":
    preprocess()
