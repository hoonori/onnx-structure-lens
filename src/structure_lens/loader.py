from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ir import Graph, Node, Tensor


def load_graph(path: str | Path) -> Graph:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json_graph(path)
    if path.suffix.lower() == ".onnx":
        return load_onnx_graph(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .onnx")


def load_json_graph(path: str | Path) -> Graph:
    data = json.loads(Path(path).read_text())
    tensors: dict[str, Tensor] = {}
    for name, raw in data.get("tensors", {}).items():
        if isinstance(raw, list):
            tensors[name] = Tensor(name=name, shape=raw)
        else:
            tensors[name] = Tensor(
                name=name,
                shape=list(raw.get("shape", [])),
                dtype=raw.get("dtype"),
                is_initializer=bool(raw.get("is_initializer", False)),
            )
    nodes = [
        Node(
            name=n.get("name") or f"{n['op_type']}_{i}",
            op_type=n["op_type"],
            inputs=list(n.get("inputs", [])),
            outputs=list(n.get("outputs", [])),
            attrs=dict(n.get("attrs", {})),
        )
        for i, n in enumerate(data.get("nodes", []))
    ]
    graph = Graph(
        name=data.get("name") or Path(path).stem,
        nodes=nodes,
        tensors=tensors,
        inputs=list(data.get("inputs", [])),
        outputs=list(data.get("outputs", [])),
    )
    graph.rebuild_links()
    return graph


def _dim_to_python(dim: Any) -> int | str | None:
    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)
    if getattr(dim, "dim_param", ""):
        return str(dim.dim_param)
    return None


def _value_info_shape(value_info: Any) -> tuple[list[int | str | None], str | None]:
    tt = value_info.type.tensor_type
    elem = getattr(tt, "elem_type", None)
    shape = [_dim_to_python(d) for d in tt.shape.dim]
    return shape, str(elem) if elem is not None else None


def _attr_to_python(attr: Any) -> Any:
    # Avoid importing onnx helper at module import time.
    import onnx

    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    return value


def load_onnx_graph(path: str | Path) -> Graph:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX loading requires `pip install onnx` or `pip install -e '.[onnx]'`") from exc

    model = onnx.load(str(path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        # Shape inference failure should not kill topology analysis.
        pass

    g = model.graph
    tensors: dict[str, Tensor] = {}

    def add_vi(vi: Any, *, is_initializer: bool = False) -> None:
        shape, dtype = _value_info_shape(vi)
        tensors[vi.name] = Tensor(name=vi.name, shape=shape, dtype=dtype, is_initializer=is_initializer)

    for vi in list(g.input) + list(g.value_info) + list(g.output):
        add_vi(vi)
    for init in g.initializer:
        tensors[init.name] = Tensor(
            name=init.name,
            shape=list(init.dims),
            dtype=str(init.data_type),
            is_initializer=True,
        )

    nodes: list[Node] = []
    used: dict[str, int] = {}
    for i, raw in enumerate(g.node):
        base = raw.name or f"{raw.op_type}_{i}"
        count = used.get(base, 0)
        used[base] = count + 1
        name = base if count == 0 else f"{base}_{count}"
        nodes.append(
            Node(
                name=name,
                op_type=raw.op_type,
                inputs=[x for x in raw.input if x],
                outputs=[x for x in raw.output if x],
                attrs={a.name: _attr_to_python(a) for a in raw.attribute},
            )
        )
    graph = Graph(
        name=g.name or Path(path).stem,
        nodes=nodes,
        tensors=tensors,
        inputs=[i.name for i in g.input if i.name not in {x.name for x in g.initializer}],
        outputs=[o.name for o in g.output],
    )
    graph.rebuild_links()
    return graph
