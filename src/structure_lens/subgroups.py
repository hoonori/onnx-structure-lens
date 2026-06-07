from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .ir import Graph, Node
from .topology import TopologyReport


@dataclass(slots=True)
class Subgroup:
    name: str
    kind: str
    nodes: list[str]
    reason: str


def _adjacency(topo: TopologyReport) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    out: dict[str, list[str]] = defaultdict(list)
    inc: dict[str, list[str]] = defaultdict(list)
    for src, dst, _ in topo.edges:
        out[src].append(dst)
        inc[dst].append(src)
    return out, inc


def detect_subgroups(graph: Graph, topo: TopologyReport) -> list[Subgroup]:
    by_name = graph.node_by_name()
    out, inc = _adjacency(topo)
    groups: list[Subgroup] = []
    seen_chains: set[tuple[str, ...]] = set()

    # Common local chains: Conv-BN-Activation, Conv-Activation, MatMul/Gemm-Add-Activation.
    patterns = [
        ("ConvBatchNormActivation", ["Conv", "BatchNormalization", {"Relu", "Sigmoid", "Clip"}]),
        ("ConvActivation", ["Conv", {"Relu", "Sigmoid", "Clip"}]),
        ("LinearBiasActivation", [{"MatMul", "Gemm"}, "Add", {"Relu", "Gelu", "Sigmoid", "Tanh"}]),
        ("LinearBias", [{"MatMul", "Gemm"}, "Add"]),
        ("NormThenLinear", [{"LayerNormalization", "BatchNormalization"}, {"MatMul", "Gemm"}]),
    ]

    def op_matches(node: Node, spec: object) -> bool:
        if isinstance(spec, set):
            return node.op_type in spec
        return node.op_type == spec

    for start in graph.nodes:
        for kind, specs in patterns:
            chain = [start.name]
            cur = start
            ok = op_matches(cur, specs[0])
            if not ok:
                continue
            for spec in specs[1:]:
                nexts = [n for n in out.get(cur.name, []) if n in by_name]
                if len(nexts) != 1:
                    ok = False
                    break
                nxt = by_name[nexts[0]]
                if not op_matches(nxt, spec):
                    ok = False
                    break
                chain.append(nxt.name)
                cur = nxt
            if ok:
                key = tuple(chain)
                if key not in seen_chains:
                    groups.append(Subgroup(kind, "pattern", chain, f"Matched op pattern {kind}"))
                    seen_chains.add(key)

    # Residual / fanin joins.
    for node_name, preds in inc.items():
        node = by_name[node_name]
        if node.op_type in {"Add", "Sum"} and len(set(preds)) >= 2:
            groups.append(Subgroup(f"ResidualJoin:{node_name}", "residual", sorted(set(preds)) + [node_name], "Add/Sum node merges multiple producer branches"))

    # Attention-like neighborhoods: MatMul -> Softmax -> MatMul with nearby Q/K/V projections.
    for n in graph.nodes:
        if n.op_type != "Softmax":
            continue
        before = [p for p in inc.get(n.name, []) if by_name[p].op_type in {"MatMul", "Gemm"}]
        after = [c for c in out.get(n.name, []) if by_name[c].op_type in {"MatMul", "Gemm"}]
        if before and after:
            nodes = sorted(set(before + [n.name] + after))
            groups.append(Subgroup(f"AttentionCore:{n.name}", "attention", nodes, "MatMul/Gemm -> Softmax -> MatMul/Gemm motif"))

    # Long single-input/single-output chains are good collapsible groups.
    visited: set[str] = set()
    for n in topo.topological_order:
        if n in visited:
            continue
        if len(inc.get(n, [])) > 1 or len(out.get(n, [])) != 1:
            continue
        chain = [n]
        cur = n
        while len(out.get(cur, [])) == 1:
            nxt = out[cur][0]
            if len(inc.get(nxt, [])) != 1 or nxt in chain:
                break
            chain.append(nxt)
            cur = nxt
            if len(out.get(cur, [])) != 1:
                break
        if len(chain) >= 4:
            visited.update(chain)
            groups.append(Subgroup(f"LinearChain:{chain[0]}..{chain[-1]}", "chain", chain, "Long single-producer/single-consumer chain"))

    # De-duplicate exact node sets within same kind.
    uniq: dict[tuple[str, tuple[str, ...]], Subgroup] = {}
    for g in groups:
        uniq.setdefault((g.kind, tuple(g.nodes)), g)
    return sorted(uniq.values(), key=lambda g: (g.kind, g.name))
