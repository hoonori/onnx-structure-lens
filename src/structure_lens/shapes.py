from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ir import Graph, Tensor


@dataclass(slots=True)
class ShapeParam:
    key: str
    label: str
    dim_value: int | str | None
    tensor_axes: list[tuple[str, int]] = field(default_factory=list)
    producer_nodes: set[str] = field(default_factory=set)
    consumer_nodes: set[str] = field(default_factory=set)

    @property
    def node_count(self) -> int:
        return len(self.producer_nodes | self.consumer_nodes)


def canonical_axis_name(tensor: Tensor, axis: int) -> str:
    rank = tensor.rank
    name = tensor.name.lower()
    if rank == 4:
        return ["B", "C", "H", "W"][axis]
    if rank == 3:
        if axis == 0:
            return "B"
        if axis == 1:
            return "S"
        return "C"
    if rank == 2:
        return "M" if axis == 0 else "N"
    if rank == 1:
        if any(k in name for k in ["bias", "gamma", "beta", "scale"]):
            return "C"
        return "N"
    return f"D{axis}"


def normalize_dim_value(value: int | str | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, int):
        return str(value)
    s = str(value)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s or "?"


def extract_shape_params(graph: Graph) -> list[ShapeParam]:
    params: dict[tuple[str, str], ShapeParam] = {}
    node_by_name = graph.node_by_name()
    for tensor in graph.tensors.values():
        if not tensor.shape:
            continue
        for axis, dim in enumerate(tensor.shape):
            axis_name = canonical_axis_name(tensor, axis)
            dim_token = normalize_dim_value(dim)
            # Symbolic dimensions are grouped by symbol. Concrete dimensions are
            # grouped by axis role + value, which is readable and stable.
            key = (axis_name, dim_token if not isinstance(dim, str) else dim_token)
            param_key = f"{axis_name}={key[1]}"
            p = params.setdefault(key, ShapeParam(key=param_key, label=axis_name, dim_value=dim))
            p.tensor_axes.append((tensor.name, axis))
            if tensor.producer and tensor.producer in node_by_name:
                p.producer_nodes.add(tensor.producer)
            p.consumer_nodes.update(c for c in tensor.consumers if c in node_by_name)
    return sorted(params.values(), key=lambda p: (-p.node_count, p.label, str(p.dim_value)))
