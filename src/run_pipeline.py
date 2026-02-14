from __future__ import annotations

"""
End-to-end pipeline runner for the medical-kg-reasoning project.

Run from project root:

    python -m src.run_pipeline

Steps:

  1. Preprocess large dataset -> diseases_large.csv
  2. Build instance KG -> data_medical_large.ttl
  2.5 SHACL validation on RAW KG  (NEW)
  3. Run RDFS reasoning -> data_medical_large_inferred.ttl
  3.5 SHACL validation on INFERRED KG (optional) (NEW)
  4. Run SPARQL demo queries on the reasoned KG
  5. Generate visualizations on the reasoned KG
  6. RAW vs REASONED comparison

We invoke each step as a Python module:

    python -m <module_name> [args...]

This avoids assumptions about script structure (main(), etc).
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from src import config


def _to_str_path(p) -> str:
    """Convert config paths that might be str or Path into a string."""
    if isinstance(p, Path):
        return str(p)
    return str(p)


def run_module(module_name: str, args: Optional[List[str]] = None) -> None:
    """
    Run a Python module as:

        python -m <module_name> [args...]

    using the current interpreter (sys.executable).
    """
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    print(f"[INFO] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_step(name: str, module_name: str, args: Optional[List[str]] = None) -> bool:
    """Run a pipeline step (module) with pretty logging and error handling."""
    print("\n" + "=" * 70)
    print(f"[STEP] {name}")
    print("=" * 70)

    try:
        run_module(module_name, args=args)
        print(f"[OK] {name} completed successfully.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {name} failed with exception:")
        print(exc)
        traceback.print_exc()
        return False


def _shapes_path() -> str:
    """
    Compute the SHACL shapes file path from config.
    Expected: data/ontology/shapes_medical_large.ttl
    """
    ontology_dir = getattr(config, "ONTOLOGY_DIR", None)
    if ontology_dir is None:
        # fallback: assume relative to project root if config missing
        return str(Path("data") / "ontology" / "shapes_medical_large.ttl")

    return str(Path(_to_str_path(ontology_dir)) / "shapes_medical_large.ttl")


def main() -> None:
    print("=======================================================")
    print("Medical Knowledge-Graph Reasoning – End-to-End Pipeline")
    print("=======================================================\n")

    # Optional: print config summary once
    if hasattr(config, "print_config_summary"):
        config.print_config_summary()

    # 1. Preprocessing
    if not _run_step(
        "1. Preprocessing large dataset",
        "src.preprocessing.preprocess_large",
    ):
        return

    # 2. Build instance KG
    if not _run_step(
        "2. Building instance KG (large)",
        "src.rdf_build.build_instances_large",
    ):
        return

    # 2.5 SHACL validation on RAW KG (NEW)
    # Enable/disable with RUN_SHACL (default: enabled)
    if os.environ.get("RUN_SHACL", "1") == "1":
        kg_raw_ttl = getattr(config, "KG_RAW_TTL", None)
        if kg_raw_ttl is None:
            print("[WARN] config.KG_RAW_TTL not found; skipping SHACL RAW validation.")
        else:
            if not _run_step(
                "2.5 SHACL validation on RAW KG",
                "src.validation.shacl_validate_large",
                args=[
                    "--graph",
                    _to_str_path(kg_raw_ttl),
                    "--shapes",
                    _shapes_path(),
                    "--inference",
                    "none",
                ],
            ):
                return

    # 3. Reasoning
    if not _run_step(
        "3. RDFS reasoning on large KG",
        "src.reasoning.reasoning_large",
    ):
        return

    # 3.5 SHACL validation on INFERRED KG (optional) (NEW)
    # Enable with RUN_SHACL_INFERRED=1
    if os.environ.get("RUN_SHACL", "1") == "1" and os.environ.get("RUN_SHACL_INFERRED", "0") == "1":
        kg_inf_ttl = getattr(config, "KG_INFERRED_TTL", None)
        if kg_inf_ttl is None:
            print("[WARN] config.KG_INFERRED_TTL not found; skipping SHACL INFERRED validation.")
        else:
            if not _run_step(
                "3.5 SHACL validation on INFERRED KG",
                "src.validation.shacl_validate_large",
                args=[
                    "--graph",
                    _to_str_path(kg_inf_ttl),
                    "--shapes",
                    _shapes_path(),
                    "--inference",
                    "rdfs",
                ],
            ):
                return

    # 4. SPARQL demo queries
    if not _run_step(
        "4. SPARQL demo queries on reasoned KG",
        "src.sparql.sparql_queries_large",
    ):
        return

    # 5. Visualizations
    if not _run_step(
        "5. Visualizations on reasoned KG",
        "src.visualization.visualize_kg_large",
    ):
        return

    # 6. RAW vs REASONED comparison (large)
    _run_step(
        "6. RAW vs REASONED comparison (large)",
        "src.visualization.compare_reasoning_large",
    )
    
    # 7. RDFS limitation demo (OWL-RL reasoning + comparison report)
    if os.environ.get("RUN_RDFS_LIMIT", "0") == "1":
        if not _run_step(
            "7. OWL-RL reasoning (for RDFS limitation demo)",
            "src.reasoning.reasoning_owlrl_large",
        ):
            return

        _run_step(
            "7.1 RDFS limitation report (RDFS vs OWL-RL)",
            "src.analysis.rdfs_limitations_large",
        )


    print(
        "\nAll pipeline steps finished.\n"
        "Check data/processed/ for TTL files and "
        "data/visualizations/ for plots."
    )

    # --------------------------------------------------------------
    # OPTIONAL: Launch Streamlit UI at the end of the pipeline
    # --------------------------------------------------------------
    # Enable by running:
    #   RUN_STREAMLIT=1 bash run_all.sh        (Linux/macOS, Git Bash)
    # or on Windows PowerShell:
    #   $env:RUN_STREAMLIT="1"; bash run_all.sh
    if os.environ.get("RUN_STREAMLIT", "0") == "1":
        print("\n[INFO] Launching Streamlit UI (app_medium.py)...")
        print("[INFO] Close the Streamlit app (Ctrl+C) to return to the shell.")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/ui_prototypes/app_medium.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
