from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .ir import Graph, Node, Tensor
from .shapes import ShapeParam, canonical_axis_name

UNKNOWN = None


@dataclass(slots=True)
class NodeCost:
    node: str
    op_type: str
    flops: int | None
    bytes_moved: int | None
    formula: str
    affected_params: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ParamImpact:
    param: str
    affected_nodes: list[str]
    known_flops: int
    known_bytes: int
    explanation: str


def _num(x: int | str | None) -> int | None:
    return x if isinstance(x, int) and x > 0 else None


def _shape_nums(t: Tensor | None) -> list[int | None]:
    if not t:
        return []
    return [_num(x) for x in t.shape]


def _prod_known(values: list[int | None]) -> int | None:
    if any(v is None for v in values):
        return None
    return prod(v for v in values if v is not None)


def _tensor_elems(t: Tensor | None) -> int | None:
    return _prod_known(_shape_nums(t))


def _bytes_for_elems(elems: int | None) -> int | None:
    return None if elems is None else elems * 4


def cost_node(graph: Graph, node: Node) -> NodeCost:
    tins = [graph.tensors.get(x) for x in node.inputs]
    touts = [graph.tensors.get(x) for x in node.outputs]
    op = node.op_type
    flops: int | None = None
    formula = "shape-only / metadata op"

    if op == "Conv" and tins and touts:
        x, w = tins[0], tins[1] if len(tins) > 1 else None
        out = touts[0]
        out_elems = _tensor_elems(out)
        wshape = _shape_nums(w)
        k = _prod_known(wshape[2:]) if len(wshape) >= 3 else None
        cin_per_group = wshape[1] if len(wshape) >= 2 else None
        if out_elems is not None and k is not None and cin_per_group is not None:
            flops = 2 * out_elems * k * cin_per_group
        formula = "2 * B * Cout * Hout * Wout * Cin/group * Kh * Kw"
    elif op in {"MatMul", "Gemm"} and tins:
        a = _shape_nums(tins[0])
        b = _shape_nums(tins[1]) if len(tins) > 1 else []
        if len(a) >= 2 and len(b) >= 2:
            m, k, n = a[-2], a[-1], b[-1]
            batch = _prod_known(a[:-2]) or 1
            if None not in (m, k, n, batch):
                flops = 2 * batch * m * k * n  # type: ignore[operator]
        formula = "2 * batch * M * K * N"
    elif op in {"Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Sqrt", "Exp"}:
        elems = _tensor_elems(touts[0]) if touts else _tensor_elems(tins[0] if tins else None)
        flops = elems
        formula = "1 * output_elements"
    elif op in {"Gelu", "QuickGelu", "Swish", "Mish"}:
        elems = _tensor_elems(touts[0]) if touts else _tensor_elems(tins[0] if tins else None)
        flops = None if elems is None else 8 * elems
        formula = "~8 * output_elements activation approximation"
    elif op in {"LayerNormalization", "BatchNormalization", "InstanceNormalization"}:
        elems = _tensor_elems(touts[0]) if touts else _tensor_elems(tins[0] if tins else None)
        flops = None if elems is None else 5 * elems
        formula = "~5 * elements"
    elif op == "Softmax":
        elems = _tensor_elems(touts[0]) if touts else _tensor_elems(tins[0] if tins else None)
        flops = None if elems is None else 3 * elems
        formula = "~3 * elements along softmax axis"
    elif op in {"AveragePool", "MaxPool", "GlobalAveragePool"}:
        out_elems = _tensor_elems(touts[0]) if touts else None
        kernel = node.attrs.get("kernel_shape") or []
        k = prod(int(x) for x in kernel) if kernel else 1
        flops = None if out_elems is None else out_elems * k
        formula = "output_elements * kernel_area"

    read = sum((_bytes_for_elems(_tensor_elems(t)) or 0) for t in tins if t)
    write = sum((_bytes_for_elems(_tensor_elems(t)) or 0) for t in touts if t)
    bytes_moved = read + write if read or write else None
    return NodeCost(node=node.name, op_type=op, flops=flops, bytes_moved=bytes_moved, formula=formula)


def annotate_affected_params(graph: Graph, costs: list[NodeCost], params: list[ShapeParam]) -> None:
    by_node = {c.node: c for c in costs}
    for p in params:
        for node in p.producer_nodes | p.consumer_nodes:
            if node in by_node:
                by_node[node].affected_params.add(p.key)


def analyze_costs(graph: Graph, params: list[ShapeParam]) -> tuple[list[NodeCost], list[ParamImpact]]:
    costs = [cost_node(graph, n) for n in graph.nodes]
    annotate_affected_params(graph, costs, params)
    impacts: list[ParamImpact] = []
    for p in params:
        nodes = sorted(p.producer_nodes | p.consumer_nodes)
        selected = [c for c in costs if c.node in nodes]
        known_flops = sum(c.flops or 0 for c in selected)
        known_bytes = sum(c.bytes_moved or 0 for c in selected)
        if not selected:
            continue
        impacts.append(
            ParamImpact(
                param=p.key,
                affected_nodes=[c.node for c in selected],
                known_flops=known_flops,
                known_bytes=known_bytes,
                explanation=f"{p.key} appears in {len(p.tensor_axes)} tensor axes and touches {len(selected)} graph nodes.",
            )
        )
    impacts.sort(key=lambda x: (-(x.known_flops or 0), -len(x.affected_nodes), x.param))
    return costs, impacts
