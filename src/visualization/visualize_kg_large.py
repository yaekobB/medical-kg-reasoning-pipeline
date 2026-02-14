# src/visualization/visualize_kg_large.py

from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph
from rdflib import Graph, Namespace, RDF, RDFS

# ---------------------------------------------------------------------
# Paths and namespaces
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ONTOLOGY_DIR = DATA_DIR / "ontology"
FIG_DIR = DATA_DIR / "visualizations"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INF_KG_PATH = PROCESSED_DIR / "data_medical_large_inferred.ttl"

MED = Namespace("http://example.org/medkg#")


# ---------------------------------------------------------------------
# Util helpers
# ---------------------------------------------------------------------


def load_inferred_graph() -> Graph:
    """
    Load the inferred medical KG (large version) from Turtle.

    Returns
    -------
    rdflib.Graph
        Graph containing schema + instance triples after reasoning.
    """
    g = Graph()
    print(f"[INFO] Loading inferred KG from: {INF_KG_PATH}")
    g.parse(INF_KG_PATH)
    print(f"[INFO] Loaded graph with {len(g)} triples.")
    return g


def label_for(g: Graph, node) -> str:
    """
    Return a human-readable label for an RDF node.

    Priority:
        1. rdfs:label
        2. local part of the URI
        3. str(node)
    """
    lab = g.value(node, RDFS.label)
    if lab:
        return str(lab)

    try:
        return g.namespace_manager.normalizeUri(node).split(":")[-1]
    except Exception:
        return str(node)


# ---------------------------------------------------------------------
# 1. Body system distribution (total / chronic / infectious)
# ---------------------------------------------------------------------


def collect_body_system_stats(
    g: Graph,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Compute, for each body system:
      - total number of diseases
      - number of chronic diseases (rdf:type med:ChronicDisease)
      - number of infectious diseases (rdf:type med:InfectiousDisease)

    We work on the inferred graph so all types are present.
    """
    total_counts: Dict[str, int] = defaultdict(int)
    chronic_counts: Dict[str, int] = defaultdict(int)
    infectious_counts: Dict[str, int] = defaultdict(int)

    for d in g.subjects(RDF.type, MED.Disease):
        systems = list(g.objects(d, MED.affectsSystem))
        if systems:
            labels = {label_for(g, s) for s in systems}
            system_label = ", ".join(sorted(labels))
        else:
            system_label = "Unknown / not classified"

        total_counts[system_label] += 1

        if (d, RDF.type, MED.ChronicDisease) in g:
            chronic_counts[system_label] += 1

        if (d, RDF.type, MED.InfectiousDisease) in g:
            infectious_counts[system_label] += 1

    return total_counts, chronic_counts, infectious_counts


def plot_body_system_distribution(g: Graph, dpi: int = 150) -> None:
    """
    Plot, for each body system:
      - total diseases
      - chronic diseases
      - infectious diseases
    """
    total_counts, chronic_counts, infectious_counts = collect_body_system_stats(g)

    if not total_counts:
        print("[WARN] No body-system stats found.")
        return

    systems_sorted = sorted(total_counts.keys(), key=lambda s: -total_counts[s])
    totals = [total_counts[s] for s in systems_sorted]
    chronics = [chronic_counts.get(s, 0) for s in systems_sorted]
    infectious = [infectious_counts.get(s, 0) for s in systems_sorted]

    x = range(len(systems_sorted))
    width = 0.28

    plt.figure(figsize=(14, 6))
    plt.bar([i - width for i in x], totals, width=width, label="Total diseases")
    plt.bar(x, chronics, width=width, label="Chronic diseases")
    plt.bar([i + width for i in x], infectious, width=width, label="Infectious diseases")

    plt.xticks(x, systems_sorted, rotation=60, ha="right")
    plt.ylabel("Number of diseases")
    plt.title("Body system distribution: total vs chronic vs infectious")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "body_system_distribution.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved body system distribution to: {out_path}")


# ---------------------------------------------------------------------
# 2. Degree distributions and frequency plots
# ---------------------------------------------------------------------


def compute_disease_symptom_mappings(
    g: Graph,
) -> Tuple[Dict, Dict]:
    """
    Build dictionaries:

      disease_to_symptoms[disease] -> set(symptoms)
      symptom_to_diseases[symptom] -> set(diseases)
    """
    disease_to_symptoms: Dict = defaultdict(set)
    symptom_to_diseases: Dict = defaultdict(set)

    for d in g.subjects(RDF.type, MED.Disease):
        for s in g.objects(d, MED.hasSymptom):
            disease_to_symptoms[d].add(s)
            symptom_to_diseases[s].add(d)

    return disease_to_symptoms, symptom_to_diseases


def plot_disease_symptom_degree_hist(g: Graph, dpi: int = 150) -> None:
    """
    Histogram: number of symptoms per disease.
    """
    disease_to_symptoms, _ = compute_disease_symptom_mappings(g)
    degrees = [len(syms) for syms in disease_to_symptoms.values()]

    if not degrees:
        print("[WARN] No disease-symptom data for degree histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=range(1, max(degrees) + 2), edgecolor="black")
    plt.xlabel("Number of symptoms")
    plt.ylabel("Number of diseases")
    plt.title("Distribution of number of symptoms per disease")
    plt.tight_layout()

    out_path = FIG_DIR / "disease_symptom_degree_hist.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved disease-symptom degree histogram to: {out_path}")


def plot_symptom_disease_degree_hist(g: Graph, dpi: int = 150) -> None:
    """
    Histogram: number of diseases per symptom.
    """
    _, symptom_to_diseases = compute_disease_symptom_mappings(g)
    degrees = [len(ds) for ds in symptom_to_diseases.values()]

    if not degrees:
        print("[WARN] No symptom-disease data for degree histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=range(1, max(degrees) + 2), edgecolor="black")
    plt.xlabel("Number of diseases")
    plt.ylabel("Number of symptoms")
    plt.title("Distribution of number of diseases per symptom")
    plt.tight_layout()

    out_path = FIG_DIR / "symptom_disease_degree_hist.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved symptom-disease degree histogram to: {out_path}")


def plot_top_symptoms_frequency(g: Graph, top_n: int = 40, dpi: int = 150) -> None:
    """
    Bar chart of the top-N most frequent symptoms across diseases.
    """
    _, symptom_to_diseases = compute_disease_symptom_mappings(g)
    if not symptom_to_diseases:
        print("[WARN] No symptom data for frequency plot.")
        return

    counts = [
        (s, len(ds)) for s, ds in symptom_to_diseases.items()
    ]
    counts_sorted = sorted(counts, key=lambda x: -x[1])[:top_n]

    symptoms = [label_for(g, s) for s, _ in counts_sorted]
    n_diseases = [c for _, c in counts_sorted]

    plt.figure(figsize=(8, 10))
    plt.barh(range(len(symptoms)), n_diseases)
    plt.yticks(range(len(symptoms)), symptoms)
    plt.xlabel("Number of diseases")
    plt.title(f"Top {top_n} most frequent symptoms")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out_path = FIG_DIR / "symptom_frequency_top40.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved top-symptom frequency plot to: {out_path}")


# ---------------------------------------------------------------------
# 3. Disease–symptom bipartite sample
# ---------------------------------------------------------------------


import numpy as np
from rdflib.namespace import RDFS

def _get_label(g: Graph, node) -> str:
    """Human-readable label: prefer rdfs:label, else URI fragment."""
    for _, _, lbl in g.triples((node, RDFS.label, None)):
        return str(lbl)

    s = str(node)
    if "#" in s:
        return s.split("#")[-1]
    if "/" in s:
        return s.rstrip("/").split("/")[-1]
    return s

def _short_label(text: str, max_len: int = 40) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def plot_disease_symptom_bipartite_sample(
    g: Graph,
    max_diseases: int = 40,
    dpi: int = 150,
) -> None:
    """
    Plot a bipartite network for a sample of diseases and their symptoms.

    Left side: diseases (red circles) – stretched to same height as symptoms.
    Right side: symptoms (blue squares).
    """
    disease_to_symptoms, _ = compute_disease_symptom_mappings(g)

    if not disease_to_symptoms:
        print("[WARN] No disease-symptom data for bipartite plot.")
        return

    # Sample first N diseases (or all if fewer)
    diseases = list(disease_to_symptoms.keys())[:max_diseases]

    B = nx.Graph()
    for d in diseases:
        B.add_node(d, bipartite=0)
        for s in disease_to_symptoms[d]:
            B.add_node(s, bipartite=1)
            B.add_edge(d, s)

    # ------------------------------------------------------------------
    # Layout: normalize y-coordinates so both sides span [0, 1]
    # ------------------------------------------------------------------
    left_nodes = [n for n in B.nodes if B.nodes[n].get("bipartite") == 0]
    right_nodes = list(set(B.nodes) - set(left_nodes))

    # Avoid division by zero if we ever have 1 node
    if len(left_nodes) > 1:
        y_left = np.linspace(0, 1, len(left_nodes))
    else:
        y_left = np.array([0.5])

    if len(right_nodes) > 1:
        y_right = np.linspace(0, 1, len(right_nodes))
    else:
        y_right = np.array([0.5])

    pos = {}
    for i, n in enumerate(left_nodes):
        pos[n] = (0.0, float(y_left[i]))      # x=0 for diseases
    for i, n in enumerate(right_nodes):
        pos[n] = (1.0, float(y_right[i]))     # x=1 for symptoms
    # ------------------------------------------------------------------

    plt.figure(figsize=(18, 8))

    # Edges
    nx.draw_networkx_edges(
        B,
        pos,
        edge_color="lightgray",
        alpha=0.4,
    )

    # Nodes: diseases (left), symptoms (right)
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=left_nodes,
        node_color="red",
        node_size=60,
        label="Diseases",
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=right_nodes,
        node_color="tab:blue",
        node_shape="s",
        node_size=40,
        label="Symptoms",
    )

    # Labels
    for d in left_nodes:
        lbl = _short_label(_get_label(g, d), max_len=40)
        x, y = pos[d]
        plt.text(
            x - 0.02,
            y,
            lbl,
            ha="right",
            va="center",
            fontsize=6,
        )

    for s in right_nodes:
        lbl = _short_label(_get_label(g, s), max_len=40)
        x, y = pos[s]
        plt.text(
            x + 0.02,
            y,
            lbl,
            ha="left",
            va="center",
            fontsize=5,
        )

    plt.legend(loc="upper right")
    plt.title(f"Disease–symptom bipartite network (sample of {len(diseases)} diseases)")
    plt.axis("off")
    plt.tight_layout()

    out_path = FIG_DIR / "disease_symptom_network_sample.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved disease–symptom bipartite sample to: {out_path}")



# ---------------------------------------------------------------------
# 4. Disease similarity network (shared symptoms ≥ threshold)
# ---------------------------------------------------------------------


def plot_disease_similarity_network(
    g: Graph,
    min_shared_symptoms: int = 3,
    dpi: int = 150,
) -> None:
    """
    Build a disease–disease similarity network:

      - Nodes: diseases
      - Edge between d1 and d2 if they share >= min_shared_symptoms symptoms
      - Edge width proportional to #shared symptoms

    Only diseases that participate in at least one such edge are shown.
    """
    disease_to_symptoms, _ = compute_disease_symptom_mappings(g)
    diseases = list(disease_to_symptoms.keys())

    if len(diseases) < 2:
        print("[WARN] Not enough diseases for similarity network.")
        return

    # precompute symptom sets
    d_sym = {d: disease_to_symptoms[d] for d in diseases}

    G = nx.Graph()
    for i, d1 in enumerate(diseases):
        for j in range(i + 1, len(diseases)):
            d2 = diseases[j]
            shared = d_sym[d1] & d_sym[d2]
            w = len(shared)
            if w >= min_shared_symptoms:
                G.add_edge(d1, d2, weight=w)

    if G.number_of_edges() == 0:
        print(f"[WARN] No disease pairs share ≥{min_shared_symptoms} symptoms.")
        return

    # keep only nodes that are incident to at least one edge
    G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.spring_layout(G, k=0.7, iterations=100, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges]
    max_w = max(weights)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.4,
        width=[1 + 2 * (w / max_w) for w in weights],
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=120,
        node_color="tab:blue",
    )

    labels = {n: label_for(g, n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title(
        f"Disease similarity network (only diseases with shared symptoms ≥ {min_shared_symptoms})"
    )
    plt.axis("off")
    plt.tight_layout()

    out_path = FIG_DIR / "disease_similarity_network_sample.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved disease similarity network to: {out_path}")


# ---------------------------------------------------------------------
# 5. Instance-level KG sample (top diseases by #symptoms)
# ---------------------------------------------------------------------


def plot_instance_kg_sample(
    g: Graph,
    max_diseases: int = 12,
    dpi: int = 150,
) -> None:
    """
    Visualize an instance-level KG sample:

      - pick up to `max_diseases` diseases with most symptoms
      - for each, show:
            Disease (red), Symptom (blue), Treatment (green)
            edges hasSymptom / treatedWith
      - labels shown only for diseases, to keep it readable
    """
    disease_to_symptoms, _ = compute_disease_symptom_mappings(g)

    if not disease_to_symptoms:
        print("[WARN] No disease-symptom data for instance KG.")
        return

    # rank diseases by number of symptoms
    ranked = sorted(
        disease_to_symptoms.items(),
        key=lambda kv: -len(kv[1]),
    )
    selected_diseases = [d for d, _ in ranked[:max_diseases]]

    # build NetworkX graph
    G = nx.Graph()
    for d in selected_diseases:
        G.add_node(d)
        for s in disease_to_symptoms[d]:
            G.add_node(s)
            G.add_edge(d, s)
        for t in g.objects(d, MED.treatedWith):
            G.add_node(t)
            G.add_edge(d, t)

    if G.number_of_nodes() == 0:
        print("[WARN] Empty instance KG sample.")
        return

    pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)

    disease_nodes = []
    symptom_nodes = []
    treatment_nodes = []

    for n in G.nodes():
        if (n, RDF.type, MED.Disease) in g:
            disease_nodes.append(n)
        elif (n, RDF.type, MED.Symptom) in g:
            symptom_nodes.append(n)
        elif (n, RDF.type, MED.Treatment) in g:
            treatment_nodes.append(n)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.4)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=disease_nodes,
        node_color="red",
        node_shape="o",
        label="Disease",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=symptom_nodes,
        node_color="blue",
        node_shape="d",
        label="Symptom",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=treatment_nodes,
        node_color="green",
        node_shape="s",
        label="Treatment",
    )

    labels = {d: label_for(g, d) for d in disease_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("Instance-level KG (sample): top diseases by number of symptoms")
    plt.legend(scatterpoints=1, fontsize=8)
    plt.axis("off")
    plt.tight_layout()

    out_path = FIG_DIR / "instance_kg_large_sample.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved instance-level KG sample to: {out_path}")


# ---------------------------------------------------------------------
# 6. Ontology / schema Graphviz figure
# ---------------------------------------------------------------------


def plot_ontology_graphviz_large() -> None:
    """
    Draw the core ontology (schema) using Graphviz.

    Classes:
        Disease, ChronicDisease, InfectiousDisease,
        Symptom, Treatment, BodySystem

    Properties:
        hasSymptom, treatedWith, affectsSystem
        subClassOf edges for ChronicDisease/InfectiousDisease -> Disease
    """
    dot = Digraph(comment="Medical KG ontology (large)", format="png")
    dot.attr(rankdir="LR")

    # class nodes
    dot.node("Disease", "Disease", shape="box")
    dot.node("ChronicDisease", "Chronic disease", shape="box")
    dot.node("InfectiousDisease", "Infectious disease", shape="box")
    dot.node("Symptom", "Symptom", shape="box")
    dot.node("Treatment", "Treatment", shape="box")
    dot.node("BodySystem", "Body system", shape="box")

    # subclass relations
    dot.edge("ChronicDisease", "Disease", label="subClassOf")
    dot.edge("InfectiousDisease", "Disease", label="subClassOf")

    # object properties
    dot.edge("Disease", "Symptom", label="hasSymptom")
    dot.edge("Disease", "Treatment", label="treatedWith")
    dot.edge("Disease", "BodySystem", label="affectsSystem")

    out_stem = (FIG_DIR / "ontology_graphviz_large").as_posix()
    dot.render(out_stem, cleanup=True)
    print(f"[INFO] Saved ontology Graphviz figure to: {out_stem}.png")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> None:
    g_inf = load_inferred_graph()

    # 1. Body system distribution (3 bars)
    plot_body_system_distribution(g_inf)

    # 2. Degree distributions & symptom frequencies
    plot_disease_symptom_degree_hist(g_inf)
    plot_symptom_disease_degree_hist(g_inf)
    plot_top_symptoms_frequency(g_inf, top_n=40)

    # 3. Bipartite sample and similarity network
    plot_disease_symptom_bipartite_sample(g_inf, max_diseases=40)
    plot_disease_similarity_network(g_inf, min_shared_symptoms=3)

    # 4. Instance-level KG sample
    plot_instance_kg_sample(g_inf, max_diseases=12)

    # 5. Ontology / schema figure
    plot_ontology_graphviz_large()


if __name__ == "__main__":
    main()
