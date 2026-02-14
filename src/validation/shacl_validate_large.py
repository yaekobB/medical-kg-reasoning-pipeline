from __future__ import annotations

import argparse
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDF
from pyshacl import validate

SH = Namespace("http://www.w3.org/ns/shacl#")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHACL validation for the medical KG (large).")
    p.add_argument(
        "--graph",
        required=True,
        help="Path to TTL graph to validate (RAW or INFERRED).",
    )
    p.add_argument(
        "--shapes",
        required=True,
        help="Path to SHACL shapes TTL file.",
    )
    p.add_argument(
        "--report-out",
        default=None,
        help="Optional output path for validation report text (default: data/processed/shacl_report_<name>.txt).",
    )
    p.add_argument(
        "--inference",
        default="rdfs",
        choices=["none", "rdfs", "owlrl"],
        help="Inference during validation (default: rdfs).",
    )
    return p.parse_args()


def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def _default_processed_dir(graph_path: Path) -> Path:
    """
    Prefer: <project>/data/processed when possible.
    Otherwise: graph_path.parent
    """
    parts = list(graph_path.parts)
    if "data" in parts:
        data_idx = parts.index("data")
        project_root = Path(*parts[:data_idx])
        return (project_root / "data" / "processed").resolve()
    return graph_path.parent.resolve()


def _shapes_summary(shapes_graph: Graph) -> tuple[int, int]:
    """
    Count:
      - NodeShapes: subjects that are sh:NodeShape OR have any sh:target*
      - PropertyConstraints: number of sh:property blank nodes + explicit sh:PropertyShape
    Note: in your TTL, property constraints are blank nodes inside sh:property [...],
    and they are not typed as sh:PropertyShape (valid SHACL), so we count sh:property arcs.
    """
    # NodeShapes by explicit typing
    node_shapes = set(shapes_graph.subjects(RDF.type, SH.NodeShape))

    # Also treat anything with target declarations as a node shape (common SHACL style)
    for pred in (SH.targetClass, SH.targetNode, SH.targetSubjectsOf, SH.targetObjectsOf):
        node_shapes.update(shapes_graph.subjects(pred, None))

    # Property constraints: count sh:property arcs (blank nodes or IRIs)
    prop_bn = list(shapes_graph.objects(None, SH.property))
    property_constraints = len(prop_bn)

    # Plus any explicitly typed PropertyShape that exists standalone
    property_constraints += len(set(shapes_graph.subjects(RDF.type, SH.PropertyShape)))

    return len(node_shapes), property_constraints


def main() -> int:
    args = parse_args()

    graph_path = Path(args.graph).resolve()
    shapes_path = Path(args.shapes).resolve()

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not shapes_path.exists():
        raise FileNotFoundError(f"Shapes file not found: {shapes_path}")

    data_graph = load_graph(graph_path)
    shapes_graph = load_graph(shapes_path)
    
    #print("Inference mode:", args.inference)

    # pyshacl expects None or "rdfs"/"owlrl" (keeping "none" user-friendly)
    inference = None if args.inference == "none" else args.inference

    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference=inference,
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        debug=False,
    )

    node_shapes_count, prop_constraints_count = _shapes_summary(shapes_graph)

    print("\n=======================================================")
    print("SHACL Validation")
    print("=======================================================")
    print(f"Data graph      : {graph_path}")
    print(f"Shapes graph    : {shapes_path}")
    print(f"Inference       : {args.inference}")
    print(f"Shapes summary  : NodeShapes={node_shapes_count}, PropertyConstraints={prop_constraints_count}")
    print(f"Conforms        : {conforms}")
    print("-------------------------------------------------------")
    print(report_text)

    # Default report paths
    processed_dir = _default_processed_dir(graph_path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.report_out:
        report_text_out = Path(args.report_out).resolve()
    else:
        report_text_out = processed_dir / f"shacl_report_{graph_path.stem}.txt"

    report_ttl_out = processed_dir / f"shacl_report_{graph_path.stem}.ttl"

    report_text_out.write_text(report_text, encoding="utf-8")
    print(f"[INFO] Report text saved to: {report_text_out}")

    # Save report graph (machine-readable)
    report_graph.serialize(destination=str(report_ttl_out), format="turtle")
    print(f"[INFO] Report graph saved to: {report_ttl_out}")

    return 0 if conforms else 1


if __name__ == "__main__":
    raise SystemExit(main())
