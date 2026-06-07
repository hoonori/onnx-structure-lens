from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .impact import analyze_costs
from .loader import load_graph
from .shapes import extract_shape_params
from .subgroups import detect_subgroups
from .topology import analyze_topology


@dataclass(slots=True)
class AnalysisReport:
    graph: dict[str, Any]
    topology: dict[str, Any]
    shape_params: list[dict[str, Any]]
    node_costs: list[dict[str, Any]]
    param_impacts: list[dict[str, Any]]
    subgroups: list[dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_model(path: str | Path) -> AnalysisReport:
    graph = load_graph(path)
    topo = analyze_topology(graph)
    params = extract_shape_params(graph)
    costs, impacts = analyze_costs(graph, params)
    subgroups = detect_subgroups(graph, topo)
    known_flops = sum(c.flops or 0 for c in costs)
    known_bytes = sum(c.bytes_moved or 0 for c in costs)
    recommendations = _recommendations(graph, topo, params, costs, impacts, subgroups)
    return AnalysisReport(
        graph={
            "name": graph.name,
            "node_count": len(graph.nodes),
            "tensor_count": len(graph.tensors),
            "input_count": len(graph.inputs),
            "output_count": len(graph.outputs),
            "known_flops": known_flops,
            "known_bytes": known_bytes,
        },
        topology=asdict(topo),
        shape_params=[_shape_param_dict(p) for p in params],
        node_costs=[_node_cost_dict(c) for c in costs],
        param_impacts=[asdict(i) for i in impacts],
        subgroups=[asdict(s) for s in subgroups],
        recommendations=recommendations,
    )


def _shape_param_dict(p: Any) -> dict[str, Any]:
    return {
        "key": p.key,
        "label": p.label,
        "dim_value": p.dim_value,
        "tensor_axes": p.tensor_axes,
        "producer_nodes": sorted(p.producer_nodes),
        "consumer_nodes": sorted(p.consumer_nodes),
        "node_count": p.node_count,
    }


def _node_cost_dict(c: Any) -> dict[str, Any]:
    return {
        "node": c.node,
        "op_type": c.op_type,
        "flops": c.flops,
        "bytes_moved": c.bytes_moved,
        "formula": c.formula,
        "affected_params": sorted(c.affected_params),
    }


def _recommendations(graph: Any, topo: Any, params: list[Any], costs: list[Any], impacts: list[Any], subgroups: list[Any]) -> list[str]:
    recs: list[str] = []
    if topo.fanout_nodes:
        node, degree = topo.fanout_nodes[0]
        recs.append(f"Inspect fanout node `{node}` first: it branches into {degree} consumers and may define a major structural split.")
    if topo.fanin_nodes:
        node, degree = topo.fanin_nodes[0]
        recs.append(f"Inspect fanin node `{node}` first: it merges {degree} producers and may correspond to residual/concat behavior.")
    if impacts:
        top = impacts[0]
        recs.append(f"Most impactful visible shape parameter is `{top.param}` with {len(top.affected_nodes)} affected nodes and {top.known_flops:,} known FLOPs.")
    if subgroups:
        kinds = sorted({s.kind for s in subgroups})
        recs.append(f"Detected {len(subgroups)} collapsible subgroups ({', '.join(kinds)}); use these as a first graph simplification layer.")
    unknown_costs = [c.node for c in costs if c.flops is None]
    if unknown_costs:
        recs.append(f"{len(unknown_costs)} nodes have unknown FLOPs formulas; add op-specific formulas for: {', '.join(unknown_costs[:8])}.")
    if not recs:
        recs.append("Graph is small/simple; topology report is likely sufficient for first-pass understanding.")
    return recs
