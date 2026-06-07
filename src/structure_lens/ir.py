from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Dim = int | str | None


@dataclass(slots=True)
class Tensor:
    name: str
    shape: list[Dim] = field(default_factory=list)
    dtype: str | None = None
    producer: str | None = None
    consumers: list[str] = field(default_factory=list)
    is_initializer: bool = False

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(slots=True)
class Node:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Graph:
    name: str
    nodes: list[Node]
    tensors: dict[str, Tensor]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def rebuild_links(self) -> None:
        for t in self.tensors.values():
            t.producer = None if not t.is_initializer else t.producer
            t.consumers.clear()
        for n in self.nodes:
            for out in n.outputs:
                self.tensors.setdefault(out, Tensor(out)).producer = n.name
            for inp in n.inputs:
                self.tensors.setdefault(inp, Tensor(inp)).consumers.append(n.name)

    def node_by_name(self) -> dict[str, Node]:
        return {n.name: n for n in self.nodes}

    def tensor(self, name: str) -> Tensor:
        return self.tensors.setdefault(name, Tensor(name))
