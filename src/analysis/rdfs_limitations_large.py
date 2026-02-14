from __future__ import annotations

from pathlib import Path
import pandas as pd
from rdflib import Graph

from src import config

MED = "http://example.org/medkg#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

# ============================================================
# CONTROL query (facts-based) for Q10
# ============================================================
# Uses only asserted facts: ChronicDisease + affectsSystem med:RespiratorySystem
# Should return SAME results for RDFS and OWL-RL.
CONTROL_QUERY = f"""
PREFIX med:  <{MED}>
PREFIX rdfs: <{RDFS}>

SELECT ?d ?label
WHERE {{
  ?d a med:ChronicDisease ;
     med:affectsSystem med:RespiratorySystem ;
     rdfs:label ?label .
}}
ORDER BY ?label
""".strip()

# ============================================================
# Q10 limitation query (OWL-defined class)
# ============================================================
# Requires OWL semantics (equivalentClass / Restriction) to materialize membership.
LIMITATION_QUERY = f"""
PREFIX med:  <{MED}>
PREFIX rdfs: <{RDFS}>

SELECT ?d ?label
WHERE {{
  ?d a med:RespiratoryChronicDisease ;
     rdfs:label ?label .
}}
ORDER BY ?label
""".strip()

# ============================================================
# Q11-A CONTROL (facts-based) for inverse demo
# ============================================================
# Uses existing direction (Disease -> hasSymptom -> Symptom).
# Should work in BOTH RDFS and OWL-RL.
#
# Robust label matching:
#   - doesn't depend on @en
#   - matches "fever" case-insensitively
Q11_CONTROL_HAS_SYMPTOM = f"""
PREFIX med:  <{MED}>
PREFIX rdfs: <{RDFS}>

SELECT ?d ?label
WHERE {{
  ?d a med:Disease ;
     med:hasSymptom ?s ;
     rdfs:label ?label .
  ?s rdfs:label ?sl .
  FILTER(LCASE(STR(?sl)) = "fever")
}}
ORDER BY ?label
""".strip()

# ============================================================
# Q11-B INVERSE DEMO query (requires owl:inverseOf materialization)
# ============================================================
# Uses inverse direction (Symptom -> symptomOf -> Disease).
# Expected:
#   - RDFS inferred KG: 0 rows (RDFS ignores owl:inverseOf)
#   - OWL-RL inferred KG: >0 rows (OWL-RL materializes inverse triples)
Q11_INVERSE_SYMPTOM_OF = f"""
PREFIX med:  <{MED}>
PREFIX rdfs: <{RDFS}>

SELECT ?d ?label
WHERE {{
  ?s rdfs:label ?sl ;
     med:symptomOf ?d .
  FILTER(LCASE(STR(?sl)) = "fever")
  ?d rdfs:label ?label .
}}
ORDER BY ?label
""".strip()


def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def run_select_labels(g: Graph, q: str) -> pd.DataFrame:
    rows = []
    for r in g.query(q):
        uri = str(r[0])
        label = str(r[1])
        rows.append({"uri": uri, "label": label})
    return pd.DataFrame(rows)


def _print_result_block(
    title: str,
    q: str,
    df_rdfs: pd.DataFrame,
    df_owl: pd.DataFrame,
    head_n: int = 25,
) -> None:
    print(f"\n=== {title} ===")
    print("Query:")
    print(q)
    print("")
    print(f"RDFS results count   : {len(df_rdfs)}")
    if df_rdfs.empty:
        print("RDFS results: (no rows)")
    else:
        print(df_rdfs.head(head_n).to_string(index=False))

    print(f"\nOWL-RL results count : {len(df_owl)}")
    if df_owl.empty:
        print("OWL-RL results: (no rows)")
    else:
        print(df_owl.head(head_n).to_string(index=False))


def _append_sample_lines(report_lines: list[str], title: str, df: pd.DataFrame, max_items: int = 15) -> None:
    report_lines.append(title)
    if df.empty:
        report_lines.append("  (no rows)")
        report_lines.append("")
        return
    for x in df["label"].head(max_items).tolist():
        report_lines.append(f"  - {x}")
    report_lines.append("")


def main() -> int:
    print("=======================================================")
    print("RDFS Limitation Demonstration (RDFS vs OWL-RL)")
    print("=======================================================\n")

    rdfs_path = Path(config.KG_INFERRED_TTL)
    owlrl_path = Path(config.PROCESSED_DIR) / "data_medical_large_owlrl_inferred.ttl"

    if not rdfs_path.exists():
        raise FileNotFoundError(f"RDFS inferred KG not found: {rdfs_path}")
    if not owlrl_path.exists():
        raise FileNotFoundError(
            f"OWL-RL inferred KG not found: {owlrl_path}\n"
            "Run the OWL-RL reasoning step first (Step 7)."
        )

    print(f"[INFO] Loading RDFS KG : {rdfs_path}")
    print(f"[INFO] Loading OWL-RL KG: {owlrl_path}")

    g_rdfs = load_graph(rdfs_path)
    g_owl = load_graph(owlrl_path)

    # -------------------------
    # Q10: logical class limitation
    # -------------------------
    df_ctrl_rdfs = run_select_labels(g_rdfs, CONTROL_QUERY)
    df_ctrl_owl = run_select_labels(g_owl, CONTROL_QUERY)

    df_lim_rdfs = run_select_labels(g_rdfs, LIMITATION_QUERY)
    df_lim_owl = run_select_labels(g_owl, LIMITATION_QUERY)

    # -------------------------
    # Q11: inverse property limitation
    # -------------------------
    df_q11_ctrl_rdfs = run_select_labels(g_rdfs, Q11_CONTROL_HAS_SYMPTOM)
    df_q11_ctrl_owl = run_select_labels(g_owl, Q11_CONTROL_HAS_SYMPTOM)

    df_q11_inv_rdfs = run_select_labels(g_rdfs, Q11_INVERSE_SYMPTOM_OF)
    df_q11_inv_owl = run_select_labels(g_owl, Q11_INVERSE_SYMPTOM_OF)

    # -------------------------
    # Build report text (rich + includes samples)
    # -------------------------
    report_lines: list[str] = []
    report_lines.append("=======================================================")
    report_lines.append("RDFS Limitation Demonstration (RDFS vs OWL-RL)")
    report_lines.append("=======================================================")
    report_lines.append("")

    # ---- Limitation #1 (Q10)
    report_lines.append("Limitation #1: OWL-defined class membership (restrictions / equivalentClass)")
    report_lines.append("Goal:")
    report_lines.append("  med:RespiratoryChronicDisease ≡ ChronicDisease AND (affectsSystem hasValue RespiratorySystem)")
    report_lines.append("")
    report_lines.append("Why RDFS fails:")
    report_lines.append("  RDFS does NOT interpret owl:equivalentClass / owl:Restriction.")
    report_lines.append("  So it cannot materialize rdf:type med:RespiratoryChronicDisease.")
    report_lines.append("")
    report_lines.append("CONTROL query (facts-based) used:")
    report_lines.append(CONTROL_QUERY)
    report_lines.append("")
    report_lines.append(f"CONTROL RDFS count   : {len(df_ctrl_rdfs)}")
    report_lines.append(f"CONTROL OWL-RL count : {len(df_ctrl_owl)}")
    report_lines.append("")
    _append_sample_lines(report_lines, "CONTROL sample labels:", df_ctrl_owl, max_items=12)

    report_lines.append("Q10 limitation query (OWL-defined class) used:")
    report_lines.append(LIMITATION_QUERY)
    report_lines.append("")
    report_lines.append(f"Q10 RDFS count   : {len(df_lim_rdfs)}")
    report_lines.append(f"Q10 OWL-RL count : {len(df_lim_owl)}")
    report_lines.append("")
    _append_sample_lines(report_lines, "Q10 OWL-RL sample labels:", df_lim_owl, max_items=20)

    report_lines.append("-------------------------------------------------------")
    report_lines.append("")

    # ---- Limitation #2 (Q11)
    report_lines.append("Limitation #2: Inverse properties (owl:inverseOf)")
    report_lines.append("OWL axiom used (in owlrl_extensions.ttl):")
    report_lines.append("  med:symptomOf owl:inverseOf med:hasSymptom")
    report_lines.append("")
    report_lines.append("Idea:")
    report_lines.append("  If Disease hasSymptom Symptom, OWL-RL can infer Symptom symptomOf Disease.")
    report_lines.append("  RDFS cannot infer inverse edges from owl:inverseOf.")
    report_lines.append("")

    report_lines.append("Q11-A CONTROL query (hasSymptom fever) used:")
    report_lines.append(Q11_CONTROL_HAS_SYMPTOM)
    report_lines.append("")
    report_lines.append(f"Q11-A RDFS count   : {len(df_q11_ctrl_rdfs)}")
    report_lines.append(f"Q11-A OWL-RL count : {len(df_q11_ctrl_owl)}")
    report_lines.append("")
    _append_sample_lines(report_lines, "Q11-A sample labels:", df_q11_ctrl_owl, max_items=15)

    report_lines.append("Q11-B INVERSE DEMO query (symptomOf fever) used:")
    report_lines.append(Q11_INVERSE_SYMPTOM_OF)
    report_lines.append("")
    report_lines.append(f"Q11-B RDFS count   : {len(df_q11_inv_rdfs)}")
    report_lines.append(f"Q11-B OWL-RL count : {len(df_q11_inv_owl)}")
    report_lines.append("")
    _append_sample_lines(report_lines, "Q11-B OWL-RL sample labels:", df_q11_inv_owl, max_items=25)

    # -------------------------
    # Save files
    # -------------------------
    out_txt = Path(config.PROCESSED_DIR) / "rdfs_limitations_report.txt"
    out_txt.write_text("\n".join(report_lines), encoding="utf-8")

    # Q10 CSVs
    df_ctrl_rdfs.to_csv(Path(config.PROCESSED_DIR) / "rdfs_control_rdfs_results.csv", index=False)
    df_ctrl_owl.to_csv(Path(config.PROCESSED_DIR) / "rdfs_control_owlrl_results.csv", index=False)
    df_lim_rdfs.to_csv(Path(config.PROCESSED_DIR) / "rdfs_limitation_rdfs_results.csv", index=False)
    df_lim_owl.to_csv(Path(config.PROCESSED_DIR) / "rdfs_limitation_owlrl_results.csv", index=False)

    # Q11 CSVs
    df_q11_ctrl_rdfs.to_csv(Path(config.PROCESSED_DIR) / "inverse_control_rdfs.csv", index=False)
    df_q11_ctrl_owl.to_csv(Path(config.PROCESSED_DIR) / "inverse_control_owlrl.csv", index=False)
    df_q11_inv_rdfs.to_csv(Path(config.PROCESSED_DIR) / "inverse_demo_rdfs.csv", index=False)
    df_q11_inv_owl.to_csv(Path(config.PROCESSED_DIR) / "inverse_demo_owlrl.csv", index=False)

    print(f"[OK] Report saved to: {out_txt}")

    # -------------------------
    # Print to console (PowerShell) like Q1–Q9
    # -------------------------
    _print_result_block(
        "CONTROL (facts-based): ChronicDisease + affectsSystem med:RespiratorySystem",
        CONTROL_QUERY,
        df_ctrl_rdfs,
        df_ctrl_owl,
        head_n=25,
    )

    _print_result_block(
        "Q10 / Limitation Demo: OWL-defined class membership (med:RespiratoryChronicDisease)",
        LIMITATION_QUERY,
        df_lim_rdfs,
        df_lim_owl,
        head_n=50,
    )

    _print_result_block(
        "Q11-A CONTROL (facts-based): Diseases that have symptom 'fever' via med:hasSymptom",
        Q11_CONTROL_HAS_SYMPTOM,
        df_q11_ctrl_rdfs,
        df_q11_ctrl_owl,
        head_n=30,
    )

    _print_result_block(
        "Q11-B LIMITATION DEMO: Diseases retrieved via inverse link med:symptomOf (should fail in RDFS)",
        Q11_INVERSE_SYMPTOM_OF,
        df_q11_inv_rdfs,
        df_q11_inv_owl,
        head_n=30,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
