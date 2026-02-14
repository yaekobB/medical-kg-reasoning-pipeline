"""
KG API for the LARGE medical knowledge graph.

This module is intentionally independent from the mini-project code.
It provides a small, read-only API that the LLM+KG assistant tab can use.

It operates over the REASONED graph:
    data/processed/data_medical_large_inferred.ttl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdflib import Graph, Namespace, RDF, RDFS, URIRef

# ---------------------------------------------------------------------
# Paths & Namespaces
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
KG_INFERRED_PATH = PROCESSED_DIR / "data_medical_large_inferred.ttl"

MED = Namespace("http://example.org/medkg#")


# ---------------------------------------------------------------------
# Dataclasses for structured results
# ---------------------------------------------------------------------

@dataclass
class DiseaseMatch:
    uri: URIRef
    label: str
    matched_symptoms: List[str]
    total_symptoms: int
    body_system: Optional[str] = None


# ---------------------------------------------------------------------
# Graph loading (lazy singleton)
# ---------------------------------------------------------------------

_graph: Optional[Graph] = None


def get_graph() -> Graph:
    """
    Load the large inferred KG once and cache it.
    """
    global _graph
    if _graph is None:
        g = Graph()
        print(f"[KG-API-LARGE] Loading inferred KG from: {KG_INFERRED_PATH}")
        g.parse(str(KG_INFERRED_PATH), format="turtle")
        print(f"[KG-API-LARGE] Loaded graph with {len(g)} triples.")
        _graph = g
    return _graph


# ---------------------------------------------------------------------
# Helpers: label & disease / symptom lookup
# ---------------------------------------------------------------------

def _label_dict_for_type(
    rdf_type: URIRef,
    g: Optional[Graph] = None,
) -> Dict[str, URIRef]:
    """
    Build a dict: {label_lower: uri} for all individuals of a given rdf:type.
    """
    g = g or get_graph()
    label_to_uri: Dict[str, URIRef] = {}

    for s in g.subjects(RDF.type, rdf_type):
        for lbl in g.objects(s, RDFS.label):
            label_str = str(lbl).strip()
            if label_str:
                label_to_uri[label_str.lower()] = s

    return label_to_uri


def get_all_symptom_labels() -> List[str]:
    """
    Return all distinct symptom labels (strings), sorted alphabetically.
    """
    g = get_graph()
    labels = set()

    #print(f"[LLM] Getting all symptoms....:")
    # Individuals of type med:Symptom
    for s in g.subjects(RDF.type, MED.Symptom):
        for lbl in g.objects(s, RDFS.label):
            labels.add(str(lbl).strip())

    return sorted(labels)


def get_all_disease_labels() -> List[str]:
    """
    Return all distinct disease labels (strings), sorted alphabetically.
    Uses med:Disease type, which after reasoning covers the 395 diseases.
    """
    g = get_graph()
    labels = set()

    #print(f"[LLM] Getting all diseases....:")
    for s in g.subjects(RDF.type, MED.Disease):
        for lbl in g.objects(s, RDFS.label):
            labels.add(str(lbl).strip())

    return sorted(labels)


def find_disease_uri_by_label(
    disease_label: str,
    g: Optional[Graph] = None,
) -> Optional[URIRef]:
    """
    Map a human-readable disease label (case-insensitive) to its URI.
    """
    g = g or get_graph()
    label_clean = (disease_label or "").strip().lower()
    if not label_clean:
        return None

    label_to_uri = _label_dict_for_type(MED.Disease, g=g)
    return label_to_uri.get(label_clean)


# ---------------------------------------------------------------------
# Symptoms & treatments for a given disease
# ---------------------------------------------------------------------

def get_symptoms_for_disease(
    disease_label: str,
) -> List[str]:
    """
    Given a disease label, return the list of symptom labels from the KG.
    If the disease is not found, return [].
    """
    g = get_graph()
    d_uri = find_disease_uri_by_label(disease_label, g=g)
    if d_uri is None:
        return []

    symptoms = []
    for sym in g.objects(d_uri, MED.hasSymptom):
        for lbl in g.objects(sym, RDFS.label):
            symptoms.append(str(lbl).strip())

    # Remove duplicates, keep stable-ish order
    return sorted(set(symptoms))


def get_treatments_for_disease(
    disease_label: str,
) -> List[str]:
    """
    Given a disease label, return the list of treatment labels from the KG.
    If the disease is not found or treatments are not present, return [].
    """
    g = get_graph()
    d_uri = find_disease_uri_by_label(disease_label, g=g)
    if d_uri is None:
        return []

    treatments = []
    for tr in g.objects(d_uri, MED.treatedWith):
        for lbl in g.objects(tr, RDFS.label):
            treatments.append(str(lbl).strip())

    return sorted(set(treatments))


def get_body_system_for_disease(
    disease_label: str,
) -> Optional[str]:
    """
    Return the label of the body system affected by this disease, if any.
    Uses med:affectsSystem edges.
    """
    g = get_graph()
    d_uri = find_disease_uri_by_label(disease_label, g=g)
    if d_uri is None:
        return None

    for sys in g.objects(d_uri, MED.affectsSystem):
        for lbl in g.objects(sys, RDFS.label):
            return str(lbl).strip()

    return None


# ---------------------------------------------------------------------
# Differential diagnosis: diseases for symptoms
# ---------------------------------------------------------------------

def diseases_for_symptoms(
    symptom_labels: List[str],
    min_matches: int = 1,
    top_k: int = 15,
) -> List[DiseaseMatch]:
    """
    Given a list of symptom labels (strings), return candidate diseases
    that have at least `min_matches` of these symptoms.

    For each disease, we compute:
      - matched_symptoms
      - total_symptoms in KG
      - body_system (if available)

    Results are sorted by (#matched_symptoms DESC, total_symptoms ASC, label).
    """
    g = get_graph()
    # Normalize input symptoms -> lowercase for matching with labels
    symptom_labels_norm = [s.strip().lower() for s in symptom_labels if s.strip()]
    if not symptom_labels_norm:
        return []

    # Build a mapping from symptom_label_lower -> list of symptom URIs
    label_to_sym_uri: Dict[str, List[URIRef]] = {}
    for s in g.subjects(RDF.type, MED.Symptom):
        labels = [str(lbl).strip() for lbl in g.objects(s, RDFS.label)]
        for lbl in labels:
            if not lbl:
                continue
            key = lbl.lower()
            label_to_sym_uri.setdefault(key, []).append(s)

    # Which symptom URIs does the user mention?
    mentioned_sym_uris = set()
    for name in symptom_labels_norm:
        # Because large KG may have multiple symptom URIs with same label,
        # we unify them into a set
        for uri in label_to_sym_uri.get(name, []):
            mentioned_sym_uris.add(uri)

    # Now, for each disease, compute matches
    disease_matches: List[DiseaseMatch] = []

    for d_uri in g.subjects(RDF.type, MED.Disease):
        # label
        labels = [str(lbl).strip() for lbl in g.objects(d_uri, RDFS.label)]
        if not labels:
            continue
        d_label = labels[0]

        # all symptoms for this disease
        sym_uris = list(g.objects(d_uri, MED.hasSymptom))
        if not sym_uris:
            continue

        # matched symptom URIs
        matched_uris = [s for s in sym_uris if s in mentioned_sym_uris]
        if len(matched_uris) < min_matches:
            continue

        # convert matched symptom URIs to labels
        matched_labels: List[str] = []
        for s in matched_uris:
            for lbl in g.objects(s, RDFS.label):
                matched_labels.append(str(lbl).strip())

        # total symptom count
        total_symptoms = len(sym_uris)

        # body system label (if any)
        body_system = None
        for sys in g.objects(d_uri, MED.affectsSystem):
            for lbl in g.objects(sys, RDFS.label):
                body_system = str(lbl).strip()
                break
            if body_system is not None:
                break

        disease_matches.append(
            DiseaseMatch(
                uri=d_uri,
                label=d_label,
                matched_symptoms=sorted(set(matched_labels)),
                total_symptoms=total_symptoms,
                body_system=body_system,
            )
        )

    # Sort: more matches first, then fewer total symptoms, then label
    disease_matches.sort(
        key=lambda d: (-len(d.matched_symptoms), d.total_symptoms, d.label.lower())
    )

    if top_k is not None and top_k > 0:
        disease_matches = disease_matches[:top_k]

    return disease_matches

# ---------------------------------------------------------------------
# Wrapper functions for assistant logic
# ---------------------------------------------------------------------

def find_diseases_for_symptoms(symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    Thin wrapper so the assistant logic can call a consistent name,
    mirroring the mini-project API.

    It simply delegates to diseases_for_symptoms(symptoms).
    """
    return diseases_for_symptoms(symptoms)


# ---------------------------------------------------------------------
# Convenience summary function (for UI / LLM explanations)
# ---------------------------------------------------------------------

def get_disease_summary(
    disease_label: str,
) -> Dict[str, Optional[str] | List[str]]:
    """
    High-level summary for one disease, useful for UI and LLM explanations.
    Returns a dict with label, body_system, symptoms, treatments.
    If disease not found, returns an empty dict.
    """
    g = get_graph()
    d_uri = find_disease_uri_by_label(disease_label, g=g)
    if d_uri is None:
        return {}

    labels = [str(lbl).strip() for lbl in g.objects(d_uri, RDFS.label)]
    label = labels[0] if labels else disease_label

    symptoms = get_symptoms_for_disease(label)
    treatments = get_treatments_for_disease(label)
    body_system = get_body_system_for_disease(label)

    return {
        "label": label,
        "body_system": body_system,
        "symptoms": symptoms,
        "treatments": treatments,
    }
