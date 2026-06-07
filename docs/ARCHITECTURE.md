# Architecture

`onnx-structure-lens` is intentionally small and report-oriented.

```text
ONNX / JSON graph
  -> loader.py       # ONNX or dependency-free JSON IR
  -> ir.py           # Graph, Node, Tensor
  -> topology.py     # DAG order, depth, critical path, fanin/fanout
  -> shapes.py       # shape/hyperparameter canonicalization
  -> impact.py       # rough FLOPs/bytes and param influence
  -> subgroups.py    # motifs and collapsible groups
  -> render.py       # JSON, Markdown, HTML
```

## Non-goals

- No hardware-specific performance prediction.
- No confidential backend rules.
- No exact compiler/runtime modeling.

## Extension points

- Add op formulas in `impact.py::cost_node`.
- Add grouping patterns in `subgroups.py::detect_subgroups`.
- Improve axis naming in `shapes.py::canonical_axis_name`.
- Build a UI on top of the stable JSON report.
