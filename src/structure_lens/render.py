from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_json(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False))


def render_markdown(report: dict[str, Any]) -> str:
    g = report["graph"]
    lines: list[str] = []
    lines.append(f"# Structure Lens Report: {g['name']}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Nodes: **{g['node_count']}**")
    lines.append(f"- Tensors: **{g['tensor_count']}**")
    lines.append(f"- Known FLOPs proxy: **{g['known_flops']:,}**")
    lines.append(f"- Known bytes moved proxy: **{g['known_bytes']:,}**")
    lines.append("")
    lines.append("## Recommendations")
    for r in report["recommendations"]:
        lines.append(f"- {r}")
    lines.append("")
    topo = report["topology"]
    lines.append("## Topology")
    lines.append(f"- Source nodes: {', '.join(topo['source_nodes'][:12]) or '-'}")
    lines.append(f"- Sink nodes: {', '.join(topo['sink_nodes'][:12]) or '-'}")
    lines.append(f"- Critical path length: {len(topo['critical_path'])}")
    lines.append(f"- Critical path: {' → '.join(topo['critical_path'][:30])}")
    if topo["fanout_nodes"]:
        lines.append("- Top fanout nodes: " + ", ".join(f"{n}({d})" for n, d in topo["fanout_nodes"][:8]))
    if topo["fanin_nodes"]:
        lines.append("- Top fanin nodes: " + ", ".join(f"{n}({d})" for n, d in topo["fanin_nodes"][:8]))
    lines.append("")
    lines.append("## Shape / Hyperparameter Linkage")
    lines.append("| Param | Dim | Tensor axes | Nodes touched |")
    lines.append("|---|---:|---:|---:|")
    for p in report["shape_params"][:30]:
        lines.append(f"| `{p['key']}` | `{p['dim_value']}` | {len(p['tensor_axes'])} | {p['node_count']} |")
    lines.append("")
    lines.append("## Parameter Impact")
    lines.append("| Param | Affected nodes | Known FLOPs | Known bytes |")
    lines.append("|---|---:|---:|---:|")
    for p in report["param_impacts"][:30]:
        lines.append(f"| `{p['param']}` | {len(p['affected_nodes'])} | {p['known_flops']:,} | {p['known_bytes']:,} |")
    lines.append("")
    lines.append("## Detected Subgroups")
    for s in report["subgroups"][:50]:
        lines.append(f"- **{s['name']}** (`{s['kind']}`): {', '.join(s['nodes'])} — {s['reason']}")
    lines.append("")
    lines.append("## Top Node Costs")
    top_costs = sorted(report["node_costs"], key=lambda c: c["flops"] or 0, reverse=True)
    lines.append("| Node | Op | FLOPs | Bytes | Formula |")
    lines.append("|---|---|---:|---:|---|")
    for c in top_costs[:40]:
        flops = "?" if c["flops"] is None else f"{c['flops']:,}"
        bytes_moved = "?" if c["bytes_moved"] is None else f"{c['bytes_moved']:,}"
        lines.append(f"| `{c['node']}` | {c['op_type']} | {flops} | {bytes_moved} | {c['formula']} |")
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_markdown(report))


def render_html(report: dict[str, Any]) -> str:
    md = render_markdown(report)
    payload = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    body = html.escape(md)
    # Small dependency-free HTML: markdown remains preformatted for reliability.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Structure Lens: {html.escape(report['graph']['name'])}</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin:0; background:#0b1020; color:#dbeafe; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
header {{ position:sticky; top:0; background:#111827ee; backdrop-filter: blur(8px); padding:18px 28px; border-bottom:1px solid #334155; }}
h1 {{ margin:0; font-size:22px; }}
main {{ display:grid; grid-template-columns: minmax(0, 1.3fr) minmax(380px, .7fr); gap:20px; padding:24px; }}
.card {{ background:#111827; border:1px solid #334155; border-radius:16px; padding:18px; box-shadow: 0 8px 30px #0004; }}
pre {{ white-space:pre-wrap; word-break:break-word; margin:0; }}
.json {{ max-height:80vh; overflow:auto; color:#bfdbfe; }}
.badge {{ display:inline-block; margin-right:12px; color:#93c5fd; }}
</style>
</head>
<body>
<header>
<h1>Structure Lens: {html.escape(report['graph']['name'])}</h1>
<span class="badge">nodes {report['graph']['node_count']}</span>
<span class="badge">known FLOPs {report['graph']['known_flops']:,}</span>
<span class="badge">subgroups {len(report['subgroups'])}</span>
</header>
<main>
<section class="card"><pre>{body}</pre></section>
<aside class="card json"><pre>{payload}</pre></aside>
</main>
</body>
</html>"""


def write_html(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_html(report))
