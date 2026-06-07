from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .ir import Graph


@dataclass(slots=True)
class TopologyReport:
    topological_order: list[str]
    depth_by_node: dict[str, int]
    critical_path: list[str]
    fanout_nodes: list[tuple[str, int]]
    fanin_nodes: list[tuple[str, int]]
    source_nodes: list[str]
    sink_nodes: list[str]
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # src, dst, tensor


def analyze_topology(graph: Graph) -> TopologyReport:
    nodes = graph.node_by_name()
    out_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    in_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    all_edges: list[tuple[str, str, str]] = []
    for tensor_name, tensor in graph.tensors.items():
        if not tensor.producer:
            continue
        for consumer in tensor.consumers:
            if consumer in nodes and tensor.producer in nodes:
                out_edges[tensor.producer].append((consumer, tensor_name))
                in_edges[consumer].append((tensor.producer, tensor_name))
                all_edges.append((tensor.producer, consumer, tensor_name))

    indegree = {n.name: len({src for src, _ in in_edges.get(n.name, [])}) for n in graph.nodes}
    q = deque([n.name for n in graph.nodes if indegree[n.name] == 0])
    order: list[str] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt, _ in out_edges.get(cur, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    if len(order) != len(graph.nodes):
        # ONNX should be acyclic, but retain input order if malformed.
        seen = set(order)
        order.extend(n.name for n in graph.nodes if n.name not in seen)

    depth: dict[str, int] = {name: 0 for name in nodes}
    prev_best: dict[str, str | None] = {name: None for name in nodes}
    for name in order:
        for nxt, _ in out_edges.get(name, []):
            cand = depth[name] + 1
            if cand > depth.get(nxt, 0):
                depth[nxt] = cand
                prev_best[nxt] = name

    end = max(depth, key=lambda n: depth[n], default=None)
    critical: list[str] = []
    while end:
        critical.append(end)
        end = prev_best.get(end)
    critical.reverse()

    source_nodes = [n for n in order if not in_edges.get(n)]
    sink_nodes = [n for n in order if not out_edges.get(n)]
    fanout_nodes = sorted(((n, len({d for d, _ in outs})) for n, outs in out_edges.items() if len({d for d, _ in outs}) > 1), key=lambda x: -x[1])
    fanin_nodes = sorted(((n, len({s for s, _ in ins})) for n, ins in in_edges.items() if len({s for s, _ in ins}) > 1), key=lambda x: -x[1])
    return TopologyReport(order, depth, critical, fanout_nodes, fanin_nodes, source_nodes, sink_nodes, all_edges)
