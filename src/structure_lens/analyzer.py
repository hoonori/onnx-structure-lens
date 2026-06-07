from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .impact import analyze_costs
from .ir import Graph
from .loader import load_graph
from .shapes import canonical_axis_name, extract_shape_params
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
    what_if: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_model(path: str | Path, what_if: dict[str, int] | None = None) -> AnalysisReport:
    graph = load_graph(path)
    report = analyze_graph(graph)
    if what_if:
        report.what_if = [_what_if_summary(graph, report, what_if)]
        report.recommendations.insert(0, _what_if_recommendation(report.what_if[0]))
    return report


def analyze_graph(graph: Graph) -> AnalysisReport:
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
        what_if=[],
    )


def apply_axis_overrides(graph: Graph, overrides: dict[str, int]) -> Graph:
    """Return a graph copy with dimensions replaced by axis role.

    Example: {"S": 256} changes every tensor axis whose canonical role is S to
    256. This is intentionally structural, not semantic: it answers "what graph
    costs move if I resize every sequence-like axis?" and is useful as a first
    pass before adding model-family-specific rules.
    """

    updated = deepcopy(graph)
    for tensor in updated.tensors.values():
        for axis in range(len(tensor.shape)):
            label = canonical_axis_name(tensor, axis)
            if label in overrides:
                tensor.shape[axis] = overrides[label]
    return updated


def _what_if_summary(graph: Graph, baseline: AnalysisReport, overrides: dict[str, int]) -> dict[str, Any]:
    changed_graph = apply_axis_overrides(graph, overrides)
    changed = analyze_graph(changed_graph)
    before_cost = {c["node"]: c for c in baseline.node_costs}
    after_cost = {c["node"]: c for c in changed.node_costs}
    changed_nodes: list[dict[str, Any]] = []
    for node, after in after_cost.items():
        before = before_cost.get(node, {})
        bf = before.get("flops")
        af = after.get("flops")
        bb = before.get("bytes_moved")
        ab = after.get("bytes_moved")
        if bf != af or bb != ab:
            changed_nodes.append(
                {
                    "node": node,
                    "op_type": after.get("op_type"),
                    "flops_before": bf,
                    "flops_after": af,
                    "flops_delta": None if bf is None or af is None else af - bf,
                    "flops_ratio": None if not bf or af is None else af / bf,
                    "bytes_before": bb,
                    "bytes_after": ab,
                    "bytes_delta": None if bb is None or ab is None else ab - bb,
                }
            )
    changed_nodes.sort(key=lambda x: abs(x["flops_delta"] or 0), reverse=True)
    bf_total = baseline.graph["known_flops"]
    af_total = changed.graph["known_flops"]
    bb_total = baseline.graph["known_bytes"]
    ab_total = changed.graph["known_bytes"]
    return {
        "overrides": overrides,
        "known_flops_before": bf_total,
        "known_flops_after": af_total,
        "known_flops_delta": af_total - bf_total,
        "known_flops_ratio": None if not bf_total else af_total / bf_total,
        "known_bytes_before": bb_total,
        "known_bytes_after": ab_total,
        "known_bytes_delta": ab_total - bb_total,
        "changed_node_count": len(changed_nodes),
        "top_changed_nodes": changed_nodes[:30],
    }


def _what_if_recommendation(summary: dict[str, Any]) -> str:
    ratio = summary.get("known_flops_ratio")
    ratio_text = "unknown" if ratio is None else f"{ratio:.2f}x"
    overrides = ", ".join(f"{k}={v}" for k, v in summary["overrides"].items())
    return (
        f"What-if `{overrides}` changes known FLOPs by {summary['known_flops_delta']:,} "
        f"({ratio_text}) across {summary['changed_node_count']} nodes."
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
