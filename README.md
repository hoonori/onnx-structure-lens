# ONNX Structure Lens

`onnx-structure-lens` is an experimental analysis tool for turning an ONNX graph
into a report that a human can reason about. It deliberately avoids
hardware-specific or proprietary optimization details. The focus is graph
structure, shape/hyperparameter linkage, and first-order operation impact.

## What it answers

- **Topology:** What are the sources/sinks, fanout points, layer depths, critical
  path, and high-connectivity nodes?
- **Hyperparameter linkage:** Which graph nodes and tensor axes are connected to
  model-level knobs like batch, sequence length, spatial resolution, channels,
  hidden size, kernel size, and head count hints?
- **Impact:** If a dimension-like hyperparameter changes, which operations are
  affected and how much do their rough FLOPs/bytes estimates move?
- **Subgroups:** What common human-readable structures appear, e.g. Conv-BN-Relu,
  residual joins, MLP blocks, attention-like neighborhoods, linear chains, and
  fanout/fanin islands?
- **Reports:** Emit JSON for machines, Markdown for review, and a self-contained
  HTML report for browsing.

## Install

```bash
pip install -e .
# Optional ONNX loading support:
pip install -e '.[onnx]'
```

## Quick demo

```bash
PYTHONPATH=src python -m structure_lens.cli examples/tiny_transformer_block.json \
  --what-if S=256 \
  --markdown reports/demo.md \
  --html reports/demo.html \
  --json reports/demo.json
```

The generated HTML report is designed for visual inspection: it starts with a
coarse, collapsible graph view and lets you click into grouped structures,
shape parameters, rough operation costs, and what-if deltas without reading raw
Markdown tables.

Run tests without extra dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

After installing the package, the console script is also available:

```bash
structure-lens examples/tiny_transformer_block.json --markdown reports/demo.md
```

The JSON input format is intentionally simple and mirrors the internal IR. ONNX
files are supported when the optional `onnx` package is installed:

```bash
structure-lens model.onnx --markdown reports/model.md --html reports/model.html
```

## Current scope

This is an **early prototype** for open-source model understanding workflows. It
is useful for model inspection and research, not a replacement for a proprietary
backend performance model.

Implemented:

- ONNX/JSON graph loaders
- Topological analysis and critical path scoring
- Shape symbol canonicalization (`B`, `S`, `H`, `W`, `C`, `K`, `M`, `N`, ...)
- Hyperparameter-to-node influence map
- Rough op impact model for Conv, MatMul/Gemm, Add/Mul, Norm, Softmax, Pool,
  Reshape/Transpose-like ops
- Subgroup detection for common chains and motifs
- JSON/Markdown/HTML report renderers

## Design principles

1. **Explainability first:** prefer imperfect but readable explanations over
   opaque exactness.
2. **No hardware assumptions:** estimates are graph-level arithmetic/memory
   proxies only.
3. **Dependency-light core:** JSON IR works without ONNX installed.
4. **Composable output:** every report has stable JSON sections for future UI
   work.
