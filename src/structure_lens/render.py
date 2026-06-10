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
    if report.get("what_if"):
        lines.append("## What-if Analysis")
        for w in report["what_if"]:
            overrides = ", ".join(f"{k}={v}" for k, v in w["overrides"].items())
            ratio = w.get("known_flops_ratio")
            ratio_text = "?" if ratio is None else f"{ratio:.2f}x"
            lines.append(f"### `{overrides}`")
            lines.append(f"- Known FLOPs: {w['known_flops_before']:,} → {w['known_flops_after']:,} ({w['known_flops_delta']:+,}, {ratio_text})")
            lines.append(f"- Known bytes: {w['known_bytes_before']:,} → {w['known_bytes_after']:,} ({w['known_bytes_delta']:+,})")
            lines.append(f"- Changed nodes: {w['changed_node_count']}")
            lines.append("| Node | Op | FLOPs before | FLOPs after | Delta | Ratio |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for n in w["top_changed_nodes"][:20]:
                bf = "?" if n["flops_before"] is None else f"{n['flops_before']:,}"
                af = "?" if n["flops_after"] is None else f"{n['flops_after']:,}"
                delta = "?" if n["flops_delta"] is None else f"{n['flops_delta']:+,}"
                nr = n.get("flops_ratio")
                nr_text = "?" if nr is None else f"{nr:.2f}x"
                lines.append(f"| `{n['node']}` | {n['op_type']} | {bf} | {af} | {delta} | {nr_text} |")
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


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_h(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{_h(_fmt(x))}</td>" for x in row) + "</tr>" for row in rows)
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"



def _node_op_map(report: dict[str, Any]) -> dict[str, str]:
    return {c["node"]: c["op_type"] for c in report.get("node_costs", [])}


def _group_node_lookup(groups: list[dict[str, Any]]) -> dict[str, str]:
    """Map a node to the first clickable subgroup that owns it.

    This is intentionally a simple UI ownership map. The raw report can contain
    overlapping motif groups; for graph navigation we prefer earlier/larger
    topology groups while preserving the exact groups in chips and detail panes.
    """

    owner: dict[str, str] = {}
    group_candidates = [g for g in groups if g["id"].startswith("subgroup:")]
    group_candidates.sort(key=lambda g: (-len(g.get("nodes", [])), g["kind"], g["title"]))
    for group in group_candidates:
        for node in group.get("nodes", []):
            owner.setdefault(node, group["id"])
    return owner


def _node_graph_data(report: dict[str, Any], nodes: list[str] | None = None) -> dict[str, Any]:
    topo = report["topology"]
    op_by_node = _node_op_map(report)
    selected = set(nodes or topo["topological_order"])
    order = [n for n in topo["topological_order"] if n in selected]
    depth = topo["depth_by_node"]
    edges = [
        {"src": src, "dst": dst, "tensor": tensor}
        for src, dst, tensor in topo["edges"]
        if src in selected and dst in selected
    ]
    return {
        "nodes": [{"name": n, "op_type": op_by_node.get(n, "?"), "depth": depth.get(n, 0)} for n in order],
        "edges": edges,
    }


def _attach_node_graphs(report: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    for group in groups:
        nodes = group.get("nodes")
        if group["id"] == "topology:overview":
            group["node_graph"] = _node_graph_data(report)
        elif nodes:
            group["node_graph"] = _node_graph_data(report, nodes)


def _node_graph_positions(graph: dict[str, Any]) -> dict[str, tuple[int, int]]:
    by_depth: dict[int, list[str]] = {}
    for node in graph.get("nodes", []):
        by_depth.setdefault(int(node.get("depth", 0)), []).append(node["name"])
    positions: dict[str, tuple[int, int]] = {}
    min_depth = min(by_depth, default=0)
    for depth, names in by_depth.items():
        for lane, name in enumerate(names):
            positions[name] = (90 + (depth - min_depth) * 150, 70 + lane * 82)
    return positions


def _render_node_graph_svg(
    graph: dict[str, Any],
    *,
    svg_id: str,
    aria_label: str,
    group_owner: dict[str, str] | None = None,
    node_class: str = "topo-node",
) -> str:
    if not graph.get("nodes"):
        return ""
    positions = _node_graph_positions(graph)
    max_x = max((x for x, _ in positions.values()), default=900) + 110
    max_y = max((y for _, y in positions.values()), default=260) + 70
    group_owner = group_owner or {}
    edge_svg = []
    for edge in graph.get("edges", []):
        src = edge["src"]
        dst = edge["dst"]
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        mid = (x1 + x2) // 2
        attr = f'{_h(src)}->{_h(dst)}'
        edge_svg.append(
            f'<path class="topo-edge" data-topology-edge="{attr}" data-node-edge="{attr}" data-overview-edge="{attr}" '
            f'd="M{x1+54},{y1} C{mid},{y1} {mid},{y2} {x2-54},{y2}" />'
            f'<text class="edge-label" x="{mid}" y="{(y1+y2)//2 - 5}">{_h(edge.get("tensor", ""))}</text>'
        )
    node_svg = []
    by_name = {n["name"]: n for n in graph.get("nodes", [])}
    for name, (x, y) in positions.items():
        node = by_name[name]
        target = group_owner.get(name, "topology:overview")
        node_svg.append(
            f'<g class="{node_class}" data-group-id="{_h(target)}" data-node-name="{_h(name)}" tabindex="0">'
            f'<rect x="{x-54}" y="{y-26}" width="108" height="52" rx="12" />'
            f'<text class="node-title" x="{x}" y="{y-4}">{_h(name[:16])}</text>'
            f'<text class="node-subtitle" x="{x}" y="{y+15}">{_h(node.get("op_type", "?"))}</text>'
            f'</g>'
        )
    return f'<svg id="{_h(svg_id)}" class="topology-svg" viewBox="0 0 {max_x} {max_y}" role="img" aria-label="{_h(aria_label)}">{"".join(edge_svg)}{"".join(node_svg)}</svg>'




def _overview_label(group: dict[str, Any]) -> str:
    title = group["title"]
    if ":" in title:
        return title.split(":", 1)[0]
    if ".." in title:
        return title.split("..", 1)[0]
    return title


def _overview_group_graph_data(report: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    owner = _group_node_lookup(groups)
    group_by_id = {g["id"]: g for g in groups}
    subgroup_ids = [g["id"] for g in groups if g["id"].startswith("subgroup:")]
    # Keep one coarse card per detected high-level group. Detailed node names stay
    # inside subgroup detail panes, not in this overview.
    nodes = [
        {
            "id": gid,
            "title": _overview_label(group_by_id[gid]),
            "kind": group_by_id[gid]["kind"],
            "depth": min((report["topology"]["depth_by_node"].get(n, 0) for n in group_by_id[gid].get("nodes", [])), default=0),
        }
        for gid in subgroup_ids
    ]
    nodes.insert(0, {"id": "graph:inputs", "title": "Inputs", "kind": "io", "depth": -1})
    nodes.append({"id": "graph:outputs", "title": "Outputs", "kind": "io", "depth": max(report["topology"]["depth_by_node"].values(), default=0) + 1})

    edges: dict[tuple[str, str], set[str]] = {}
    for src, dst, tensor in report["topology"]["edges"]:
        raw_src_group = owner.get(src)
        raw_dst_group = owner.get(dst)
        if raw_src_group is None and raw_dst_group is None:
            continue
        src_group = raw_src_group or "graph:inputs"
        dst_group = raw_dst_group or "graph:outputs"
        if src_group == dst_group:
            continue
        if src_group not in group_by_id and src_group not in {"graph:inputs", "graph:outputs"}:
            src_group = "graph:inputs"
        if dst_group not in group_by_id and dst_group not in {"graph:inputs", "graph:outputs"}:
            dst_group = "graph:outputs"
        if src_group == dst_group:
            continue
        edges.setdefault((src_group, dst_group), set()).add(tensor)
    return {
        "nodes": nodes,
        "edges": [{"src": s, "dst": d, "tensors": sorted(tensors)} for (s, d), tensors in edges.items()],
    }



def _mermaid_label(value: Any) -> str:
    return str(value).replace('"', "'").replace("\n", " ")


def _mermaid_id(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def _render_mermaid_graph_block(
    graph: dict[str, Any],
    *,
    block_id: str,
    css_class: str,
    node_title: str = "title",
    clickable: bool = False,
) -> str:
    lines = ["---", "config:", "  layout: dagre", "---", "flowchart LR"]
    node_ids: dict[str, str] = {}
    for i, node in enumerate(graph.get("nodes", [])):
        raw_id = str(node.get("id") or node.get("name"))
        mid = _mermaid_id("g" if clickable else "n", i)
        node_ids[raw_id] = mid
        title = _mermaid_label(node.get(node_title) or node.get("name") or raw_id)
        subtitle = _mermaid_label(node.get("kind") or node.get("op_type") or "")
        label = title if not subtitle else f"{title}<br/><small>{subtitle}</small>"
        lines.append(f'  {mid}["{label}"]')
        if clickable:
            lines.append(f'  click {mid} call selectGroup("{_mermaid_label(raw_id)}") "Open detail"')
    for edge in graph.get("edges", []):
        src = node_ids.get(str(edge["src"]))
        dst = node_ids.get(str(edge["dst"]))
        if not src or not dst:
            continue
        tensors = edge.get("tensors") or ([edge.get("tensor")] if edge.get("tensor") else [])
        label = _mermaid_label(", ".join(str(t) for t in tensors[:3]))
        lines.append(f"  %% data-overview-edge {edge['src']}->{edge['dst']}" if clickable else f"  %% data-node-edge {edge['src']}->{edge['dst']}")
        if label:
            lines.append(f'  {src} -->|"{label}"| {dst}')
        else:
            lines.append(f"  {src} --> {dst}")
    return f'<pre id="{_h(block_id)}" class="mermaid {css_class}">{_h(chr(10).join(lines))}</pre>'


def _overview_positions(graph: dict[str, Any]) -> dict[str, tuple[int, int]]:
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for node in graph.get("nodes", []):
        by_depth.setdefault(int(node.get("depth", 0)), []).append(node)
    min_depth = min(by_depth, default=0)
    positions: dict[str, tuple[int, int]] = {}
    for depth, nodes in by_depth.items():
        for lane, node in enumerate(sorted(nodes, key=lambda n: n["title"])):
            positions[node["id"]] = (120 + (depth - min_depth) * 180, 80 + lane * 100)
    return positions


def _render_overview_graph_svg(report: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    graph = _overview_group_graph_data(report, groups)
    return _render_mermaid_graph_block(
        graph,
        block_id="overview-graph",
        css_class="overview-mermaid",
        node_title="title",
        clickable=True,
    )

def _build_group_data(report: dict[str, Any]) -> list[dict[str, Any]]:
    g = report["graph"]
    topo = report["topology"]
    groups: list[dict[str, Any]] = [
        {
            "id": "model:summary",
            "title": "Model Summary",
            "kind": "model",
            "summary": f"{g['node_count']} nodes, {g['tensor_count']} tensors, {_fmt(g['known_flops'])} known FLOPs proxy.",
            "metrics": {
                "Nodes": g["node_count"],
                "Tensors": g["tensor_count"],
                "Known FLOPs": g["known_flops"],
                "Known bytes": g["known_bytes"],
            },
            "items": report["recommendations"],
        },
        {
            "id": "topology:overview",
            "title": "Topology",
            "kind": "topology",
            "summary": f"Critical path length {len(topo['critical_path'])}; {len(topo['fanout_nodes'])} fanout and {len(topo['fanin_nodes'])} fanin hotspots.",
            "metrics": {"Critical path": len(topo["critical_path"]), "Sources": len(topo["source_nodes"]), "Sinks": len(topo["sink_nodes"])},
            "items": [
                "Critical path: " + " → ".join(topo["critical_path"][:40]),
                "Sources: " + (", ".join(topo["source_nodes"][:12]) or "-"),
                "Sinks: " + (", ".join(topo["sink_nodes"][:12]) or "-"),
            ],
            "table": {
                "headers": ["Type", "Node", "Degree"],
                "rows": [["fanout", n, d] for n, d in topo["fanout_nodes"][:10]] + [["fanin", n, d] for n, d in topo["fanin_nodes"][:10]],
            },
        },
        {
            "id": "shape:params",
            "title": "Shape Params",
            "kind": "shape",
            "summary": "Canonical axis roles and the graph nodes they touch.",
            "metrics": {"Params": len(report["shape_params"]), "Top axes shown": min(12, len(report["shape_params"]))},
            "table": {
                "headers": ["Param", "Dim", "Tensor axes", "Nodes touched"],
                "rows": [[p["key"], p["dim_value"], len(p["tensor_axes"]), p["node_count"]] for p in report["shape_params"][:20]],
            },
        },
        {
            "id": "impact:params",
            "title": "Param Impact",
            "kind": "impact",
            "summary": "First-order FLOPs/bytes proxy grouped by shape parameter.",
            "metrics": {"Known FLOPs": g["known_flops"], "Known bytes": g["known_bytes"]},
            "table": {
                "headers": ["Param", "Affected nodes", "Known FLOPs", "Known bytes"],
                "rows": [[p["param"], len(p["affected_nodes"]), p["known_flops"], p["known_bytes"]] for p in report["param_impacts"][:20]],
            },
        },
        {
            "id": "cost:nodes",
            "title": "Node Costs",
            "kind": "cost",
            "summary": "Largest per-node rough operation costs.",
            "metrics": {"Costed nodes": len([c for c in report["node_costs"] if c["flops"] is not None])},
            "table": {
                "headers": ["Node", "Op", "FLOPs", "Bytes", "Formula"],
                "rows": [[c["node"], c["op_type"], c["flops"], c["bytes_moved"], c["formula"]] for c in sorted(report["node_costs"], key=lambda c: c["flops"] or 0, reverse=True)[:20]],
            },
        },
    ]
    if report.get("what_if"):
        for w in report["what_if"]:
            label = ", ".join(f"{k}={v}" for k, v in w["overrides"].items())
            groups.append(
                {
                    "id": f"whatif:{label}",
                    "title": f"What-if {label}",
                    "kind": "whatif",
                    "summary": f"Known FLOPs changes {_fmt(w['known_flops_before'])} → {_fmt(w['known_flops_after'])} ({_fmt(w['known_flops_ratio'])}x).",
                    "metrics": {"Δ FLOPs": w["known_flops_delta"], "Δ bytes": w["known_bytes_delta"], "Changed nodes": w["changed_node_count"]},
                    "table": {
                        "headers": ["Node", "Op", "Before", "After", "Δ", "Ratio"],
                        "rows": [[n["node"], n["op_type"], n["flops_before"], n["flops_after"], n["flops_delta"], n["flops_ratio"]] for n in w["top_changed_nodes"][:20]],
                    },
                }
            )
    used_ids: dict[str, int] = {}
    for s in report["subgroups"]:
        related_costs = [c for c in report["node_costs"] if c["node"] in set(s["nodes"])]
        base_id = f"subgroup:{s['name']}"
        used_ids[base_id] = used_ids.get(base_id, 0) + 1
        group_id = base_id if used_ids[base_id] == 1 else f"{base_id}#{used_ids[base_id]}"
        groups.append(
            {
                "id": group_id,
                "title": s["name"],
                "kind": s["kind"],
                "summary": s["reason"],
                "nodes": list(s["nodes"]),
                "metrics": {"Nodes": len(s["nodes"]), "Known FLOPs": sum(c["flops"] or 0 for c in related_costs)},
                "items": ["Nodes: " + ", ".join(s["nodes"])],
                "table": {
                    "headers": ["Node", "Op", "FLOPs", "Bytes"],
                    "rows": [[c["node"], c["op_type"], c["flops"], c["bytes_moved"]] for c in related_costs],
                },
            }
        )
    _attach_node_graphs(report, groups)
    return groups


def _render_static_detail(group: dict[str, Any]) -> str:
    metrics = "".join(f"<span class=\"metric\"><b>{_h(k)}</b><em>{_h(_fmt(v))}</em></span>" for k, v in group.get("metrics", {}).items())
    items = "".join(f"<li>{_h(item)}</li>" for item in group.get("items", []))
    table = ""
    if group.get("table"):
        table = _table(group["table"]["headers"], group["table"]["rows"])
    node_graph = ""
    if group.get("node_graph"):
        node_graph = '<h4>internal node graph</h4>' + _render_mermaid_graph_block(
            group["node_graph"],
            block_id=f"detail-node-graph-{group['id']}",
            css_class="detail-mermaid",
            node_title="name",
        )
    return f"<section class=\"detail-card\"><h3>{_h(group['title'])}</h3><p>{_h(group['summary'])}</p><div class=\"metrics\">{metrics}</div>{node_graph}{'<ul>'+items+'</ul>' if items else ''}{table}</section>"


def _graph_positions(groups: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {"model:summary": (520, 70)}
    core_ids = ["topology:overview", "shape:params", "impact:params", "cost:nodes"] + [g["id"] for g in groups if g["kind"] == "whatif"]
    core_ids = [gid for gid in core_ids if any(g["id"] == gid for g in groups)]
    if core_ids:
        step = 980 / max(1, len(core_ids))
        for i, gid in enumerate(core_ids):
            positions[gid] = (int(80 + step * i + step / 2), 210)
    subgroup_ids = [g["id"] for g in groups if g["id"].startswith("subgroup:")]
    cols = 3
    for i, gid in enumerate(subgroup_ids):
        positions[gid] = (220 + (i % cols) * 300, 360 + (i // cols) * 115)
    return positions


def _render_group_graph(groups: list[dict[str, Any]]) -> str:
    positions = _graph_positions(groups)
    by_id = {g["id"]: g for g in groups}
    core_ids = [gid for gid, (_, y) in positions.items() if y == 210]
    subgroup_ids = [gid for gid in positions if gid.startswith("subgroup:")]
    edges: list[tuple[str, str]] = [("model:summary", gid) for gid in core_ids]
    edges += [("impact:params", gid) for gid in subgroup_ids if "impact:params" in positions]
    height = max((y for _, y in positions.values()), default=500) + 95
    edge_svg = []
    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        edge_svg.append(f'<line class="edge" x1="{x1}" y1="{y1 + 34}" x2="{x2}" y2="{y2 - 34}" />')
    node_svg = []
    for gid, (x, y) in positions.items():
        group = by_id[gid]
        cls = f"node {group['kind']}"
        title = _h(group["title"][:26])
        subtitle = _h(group["kind"])
        node_svg.append(
            f'<g class="{cls}" data-group-id="{_h(gid)}" tabindex="0">'
            f'<rect x="{x-95}" y="{y-34}" width="190" height="68" rx="14" />'
            f'<text class="node-title" x="{x}" y="{y-5}">{title}</text>'
            f'<text class="node-subtitle" x="{x}" y="{y+17}">{subtitle}</text>'
            f'</g>'
        )
    return f'<svg id="overview-graph" viewBox="0 0 1120 {height}" role="img" aria-label="Group overview graph">{"".join(edge_svg)}{"".join(node_svg)}</svg>'






def _collapsed_edge_list(edges: list[dict[str, Any]], owner: dict[str, str]) -> list[dict[str, str]]:
    collapsed: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        src = owner.get(edge["src"], edge["src"])
        dst = owner.get(edge["dst"], edge["dst"])
        if src == dst:
            continue
        collapsed.setdefault((src, dst), set()).add(str(edge.get("tensor") or ""))
    return [
        {"src": src, "dst": dst, "tensor": ", ".join(t for t in sorted(tensors) if t)}
        for (src, dst), tensors in sorted(collapsed.items())
    ]


def _find_cycle_nodes(edges: list[dict[str, str]]) -> set[str]:
    outgoing: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        outgoing.setdefault(src, []).append(dst)
        nodes.update([src, dst])
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> set[str]:
        if node in visiting:
            return set(visiting[visiting.index(node):])
        if node in visited:
            return set()
        visiting.append(node)
        for nxt in outgoing.get(node, []):
            cycle = visit(nxt)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node)
        return set()

    for node in sorted(nodes):
        cycle = visit(node)
        if cycle:
            return cycle
    return set()


def _decompose_cycle_groups(
    display_groups: list[dict[str, Any]],
    owner: dict[str, str],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Split groups until the fully collapsed viewer graph is acyclic.

    Raw model graphs are expected to be DAGs, but collapsing arbitrary configured
    groups can introduce cycles (A1→B1→A2→B2 collapses to A↔B). When that happens
    we preserve the user's original group in the detail/chip config, but split
    the viewer's visible collapse unit into single-node parts so dagre always
    receives a DAG.
    """

    groups_by_id = {g["id"]: dict(g) for g in display_groups}
    decomposed: list[str] = []
    while True:
        collapsed = _collapsed_edge_list(edges, owner)
        cycle = _find_cycle_nodes(collapsed)
        splittable = [gid for gid in cycle if gid in groups_by_id and len(groups_by_id[gid].get("nodes", [])) > 1]
        if not splittable:
            return list(groups_by_id.values()), owner, decomposed
        for gid in splittable:
            group = groups_by_id.pop(gid)
            decomposed.append(gid)
            for node in group.get("nodes", []):
                part_id = f"{gid}::part::{node}"
                groups_by_id[part_id] = {
                    "id": part_id,
                    "title": node,
                    "full_title": f"{group.get('full_title', group['title'])} / {node}",
                    "kind": f"{group.get('kind', 'group')}-part",
                    "nodes": [node],
                    "source_group": group.get("source_group", gid),
                    "decomposed": True,
                }
                owner[node] = part_id


def _viewer_graph_data(report: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Build TensorBoard-style expandable, cycle-safe graph data.

    The viewer keeps raw model nodes and detected/configured groups separate. It
    first assigns each node to one canonical group, then checks the fully
    collapsed graph. If collapsing a configured group would introduce a cycle,
    that group is decomposed into smaller visible parts until the collapsed graph
    is acyclic. The original group remains available in the detail/chip data.
    """

    owner = _group_node_lookup(groups)
    node_graph = _node_graph_data(report)
    initial_groups: list[dict[str, Any]] = []
    for g in groups:
        if not g["id"].startswith("subgroup:"):
            continue
        owned_nodes = [n for n in g.get("nodes", []) if owner.get(n) == g["id"]]
        if not owned_nodes:
            continue
        initial_groups.append(
            {
                "id": g["id"],
                "title": _overview_label(g),
                "full_title": g["title"],
                "kind": g["kind"],
                "nodes": owned_nodes,
                "source_group": g["id"],
                "decomposed": False,
            }
        )
    display_groups, owner, decomposed = _decompose_cycle_groups(initial_groups, dict(owner), node_graph["edges"])
    source_group_ids = {g.get("source_group", g["id"]) for g in display_groups}
    tree = [
        {
            "id": source_id,
            "title": next((g["title"] for g in groups if g["id"] == source_id), source_id),
            "kind": next((g["kind"] for g in groups if g["id"] == source_id), "group"),
            "children": [
                {"id": vg["id"], "title": vg["title"], "kind": vg["kind"], "nodes": vg.get("nodes", [])}
                for vg in display_groups
                if vg.get("source_group", vg["id"]) == source_id
            ],
        }
        for source_id in sorted(source_group_ids)
    ]
    collapsed_edges = _collapsed_edge_list(node_graph["edges"], owner)
    return {
        "groups": sorted(display_groups, key=lambda g: min((report["topology"].get("depth_by_node", {}).get(n, 0) for n in g.get("nodes", [])), default=0)),
        "nodes": [
            {
                "id": n["name"],
                "label": n["name"],
                "op_type": n.get("op_type", "?"),
                "depth": n.get("depth", 0),
                "group": owner.get(n["name"]),
            }
            for n in node_graph["nodes"]
        ],
        "edges": node_graph["edges"],
        "collapsed_edges": collapsed_edges,
        "owner": owner,
        "tree": tree,
        "decomposed_groups": decomposed,
        "cycle_safe": not _find_cycle_nodes(collapsed_edges),
    }

def render_html(report: dict[str, Any]) -> str:
    groups = _build_group_data(report)
    viewer_graph = _viewer_graph_data(report, groups)
    payload = json.dumps({"report": report, "groups": groups, "viewerGraph": viewer_graph}, ensure_ascii=False).replace("</", "<\\/")
    initial_detail = _render_static_detail(groups[0])
    graph = report["graph"]
    chips = "".join(
        f'<button class="chip" data-group-id="{_h(g["id"])}">{_h(g["title"])}</button>'
        for g in groups[:18]
    )
    tree_html_parts: list[str] = []
    for root in viewer_graph.get("tree", []):
        focus_id = root.get("children", [{}])[0].get("id", root["id"]) if root.get("children") else root["id"]
        tree_html_parts.append(
            f'<button class="tree-row" data-tree-id="{_h(root["id"])}" onclick="focusGraphItem(\'{_h(focus_id)}\')"><span class="tree-toggle">▾</span><span>{_h(root["title"])}</span><span class="tree-kind">{_h(root["kind"])}</span></button>'
        )
        for child in root.get("children", []):
            tree_html_parts.append(
                f'<button class="tree-row" style="padding-left:24px" data-tree-id="{_h(child["id"])}" onclick="focusGraphItem(\'{_h(child["id"])}\')"><span class="tree-toggle">•</span><span>{_h(child["title"])}</span><span class="tree-kind">{_h(child["kind"])}</span></button>'
            )
    initial_tree = "".join(tree_html_parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Structure Lens: {_h(graph['name'])}</title>
<style>
:root {{ color-scheme: dark; --bg:#08111f; --panel:#101b2e; --panel2:#0d1627; --line:#334155; --text:#e5eefb; --muted:#93a4bb; --accent:#60a5fa; --good:#34d399; --warn:#fbbf24; --pink:#f472b6; --purple:#a78bfa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at 20% 0%, #13284a 0, var(--bg) 42%); color:var(--text); font:14px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
header {{ position:sticky; top:0; z-index:10; background:#08111fe8; backdrop-filter: blur(10px); padding:16px 22px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:22px; }}
.badge {{ display:inline-flex; gap:6px; align-items:center; margin:0 8px 6px 0; padding:5px 10px; border:1px solid #315071; border-radius:999px; color:#bfdbfe; background:#0b1d33; }}
main {{ padding:18px; max-width:1560px; margin:0 auto; }}
.grid {{ display:grid; grid-template-columns:280px minmax(0, 1.2fr) minmax(380px, .75fr); gap:18px; align-items:start; }}
.card, .detail-card {{ background:linear-gradient(180deg, #111d31, #0d1728); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 14px 40px #0007; }}
.card h2, .detail-card h3 {{ margin:0 0 10px; }}
.graph-card {{ grid-column:2 / 3; }}
.nav-card {{ position:sticky; top:92px; max-height:calc(100vh - 112px); overflow:auto; }}
.nav-card h2 {{ font-size:15px; margin:0 0 8px; color:#bfdbfe; }}
.cycle-safe {{ color:#86efac; font-size:12px; margin:4px 0 12px; }}
.file-tree {{ display:flex; flex-direction:column; gap:3px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
.tree-row {{ width:100%; display:flex; gap:6px; align-items:center; border:0; border-radius:8px; padding:6px 7px; background:transparent; color:#cbd5e1; text-align:left; cursor:pointer; }}
.tree-row:hover, .tree-row.active {{ background:#13223a; color:#fff; }}
.tree-toggle {{ color:#93c5fd; width:16px; display:inline-block; }}
.tree-kind {{ color:#94a3b8; font-size:10px; margin-left:auto; }}
.toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0 12px; }}
.toolbtn {{ border:1px solid #38516e; background:#0b1d33; color:#dbeafe; border-radius:10px; padding:7px 10px; cursor:pointer; }}
.toolbtn:hover {{ border-color:#7dd3fc; color:white; }}
#graph-viewer {{ width:100%; height:620px; min-height:440px; border:1px solid #263d59; border-radius:16px; background:#07101d; }}
.viewer-hint {{ color:#a9bdd4; margin:0 0 8px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}
.chip {{ border:1px solid #38516e; background:#0b1d33; color:#dbeafe; border-radius:999px; padding:7px 10px; cursor:pointer; }}
.chip:hover, .chip.active {{ border-color:#7dd3fc; color:white; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:12px 0; }}
.metric {{ display:block; padding:10px; border:1px solid #2e4661; background:#0b1829; border-radius:12px; }}
.metric b {{ display:block; color:#9fb4cc; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
.metric em {{ display:block; color:#f8fafc; font-style:normal; font-size:18px; font-weight:750; }}
.table-wrap {{ overflow:auto; margin-top:12px; border:1px solid #2d435c; border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; min-width:520px; }}
th, td {{ padding:8px 10px; border-bottom:1px solid #23374f; vertical-align:top; }}
th {{ position:sticky; top:0; background:#13223a; color:#bfdbfe; text-align:left; }}
td {{ color:#dbeafe; }}
ul {{ padding-left:20px; color:#d5e3f3; }}
.raw-json {{ margin-top:18px; }}
.raw-json summary {{ cursor:pointer; color:#93c5fd; }}
.raw-json pre {{ white-space:pre-wrap; max-height:360px; overflow:auto; background:#07101d; padding:14px; border-radius:12px; }}
@media (max-width: 1100px) {{ .grid {{ grid-template-columns:1fr; }} .graph-card {{ grid-column:auto; }} .nav-card {{ position:static; max-height:none; }} header {{ position:static; }} main {{ padding:12px; }} #graph-viewer {{ height:520px; }} }}
</style>
</head>
<body>
<header>
<h1>Structure Lens: {_h(graph['name'])}</h1>
<span class="badge">nodes <b>{_h(_fmt(graph['node_count']))}</b></span>
<span class="badge">known FLOPs <b>{_h(_fmt(graph['known_flops']))}</b></span>
<span class="badge">subgroups <b>{_h(_fmt(len(report['subgroups'])))}</b></span>
</header>
<main>
<section class="grid">
  <nav class="card nav-card" aria-label="Graph file structure">
    <h2>Graph Structure</h2>
    <p class="cycle-safe">cycle-safe collapse: groups are decomposed before a collapsed cycle can appear.</p>
    <div id="graph-tree" class="file-tree">{initial_tree}</div>
  </nav>
  <article class="card graph-card">
    <h2>TensorBoard-style expandable graph viewer</h2>
    <p class="viewer-hint">Click a block/node to inspect it. Double-click a block, or use the detail button, to expand/collapse internal nodes. Layout is recomputed with dagre after each change.</p>
    <div class="toolbar">
      <button class="toolbtn" onclick="expandAllGroups()">Expand all</button>
      <button class="toolbtn" onclick="collapseAllGroups()">Collapse all</button>
      <button class="toolbtn" onclick="fitGraph()">Fit</button>
    </div>
    <div id="graph-viewer" role="img" aria-label="Expandable model graph viewer"></div>
    <div class="chips">{chips}</div>
  </article>
  <aside id="detail-view" class="detail-card">{initial_detail}</aside>
</section>
<details class="raw-json card">
  <summary>Raw JSON data</summary>
  <pre>{_h(json.dumps(report, indent=2, ensure_ascii=False))}</pre>
</details>
<script id="lens-data" type="application/json">{payload}</script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
<script>
const lensPayload = JSON.parse(document.getElementById('lens-data').textContent);
const groups = new Map(lensPayload.groups.map(g => [g.id, g]));
const viewerGraph = lensPayload.viewerGraph;
const viewerGroups = new Map(viewerGraph.groups.map(g => [g.id, g]));
for (const vg of viewerGraph.groups) {{
  if (!groups.has(vg.id)) {{
    groups.set(vg.id, {{ id: vg.id, title: vg.full_title || vg.title, kind: vg.kind, summary: vg.decomposed ? `Decomposed cycle-safe part of ${{vg.source_group}}` : 'Viewer collapse group', nodes: vg.nodes || [], metrics: {{ Nodes: (vg.nodes || []).length }}, items: vg.decomposed ? ['This visible part was split from its configured group to prevent collapsed graph cycles.'] : [] }});
  }}
}}
const expandedGroups = new Set();
let selectedGroupId = null;
let cy = null;

function fmt(value) {{
  if (value === null || value === undefined) return '?';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  return String(value);
}}
function esc(value) {{
  return String(value).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function renderTable(table) {{
  if (!table) return '';
  const head = table.headers.map(h => `<th>${{esc(h)}}</th>`).join('');
  const rows = table.rows.map(row => `<tr>${{row.map(cell => `<td>${{esc(fmt(cell))}}</td>`).join('')}}</tr>`).join('');
  return `<div class="table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{rows}}</tbody></table></div>`;
}}
function renderMetrics(metrics) {{
  return Object.entries(metrics || {{}}).map(([k,v]) => `<span class="metric"><b>${{esc(k)}}</b><em>${{esc(fmt(v))}}</em></span>`).join('');
}}
function isViewerGroup(id) {{ return viewerGroups.has(id); }}
function treeRows() {{
  const roots = viewerGraph.tree || [];
  return roots.flatMap(root => [root, ...(root.children || [])]);
}}
function renderTree() {{
  const container = document.getElementById('graph-tree');
  if (!container) return;
  container.innerHTML = (viewerGraph.tree || []).map(root => {{
    const rootExpanded = true;
    const children = (root.children || []).map(child => `<button class="tree-row" style="padding-left:24px" data-tree-id="${{esc(child.id)}}" onclick="focusGraphItem('${{esc(child.id)}}')"><span class="tree-toggle">•</span><span>${{esc(child.title)}}</span><span class="tree-kind">${{esc(child.kind)}}</span></button>`).join('');
    return `<button class="tree-row" data-tree-id="${{esc(root.id)}}" onclick="focusGraphItem('${{esc((root.children && root.children[0] && root.children[0].id) || root.id)}}')"><span class="tree-toggle">${{rootExpanded ? '▾' : '▸'}}</span><span>${{esc(root.title)}}</span><span class="tree-kind">${{esc(root.kind)}}</span></button>${{children}}`;
  }}).join('');
}}
function markTree(id) {{
  document.querySelectorAll('[data-tree-id]').forEach(el => el.classList.toggle('active', el.getAttribute('data-tree-id') === id));
}}
function focusGraphItem(id) {{
  if (isViewerGroup(id) && !expandedGroups.has(id)) {{
    // Keep it collapsed but visible, matching file explorer focus semantics.
  }}
  selectGroup(id);
  if (cy) {{
    const ele = cy.getElementById(id);
    if (ele && ele.length) {{
      cy.elements().removeClass('selected');
      ele.addClass('selected');
      cy.animate({{ fit: {{ eles: ele, padding: 120 }}, duration: 240 }});
    }}
  }}
  markTree(id);
}}
function endpointFor(nodeId, role) {{
  const owner = viewerGraph.owner[nodeId];
  if (owner) return expandedGroups.has(owner) ? nodeId : owner;
  return role === 'src' ? 'graph:inputs' : 'graph:outputs';
}}
function visibleElements() {{
  const elements = [];
  elements.push({{ data: {{ id: 'graph:inputs', label: 'Inputs', kind: 'io', type: 'boundary' }} }});
  elements.push({{ data: {{ id: 'graph:outputs', label: 'Outputs', kind: 'io', type: 'boundary' }} }});
  for (const group of viewerGraph.groups) {{
    const expanded = expandedGroups.has(group.id);
    elements.push({{ data: {{ id: group.id, label: `${{group.title}} ${{expanded ? '[-]' : '[+]'}}`, kind: group.kind, type: 'group', groupId: group.id, expanded }} }});
    if (expanded) {{
      for (const nodeName of group.nodes) {{
        const n = viewerGraph.nodes.find(x => x.id === nodeName);
        if (!n) continue;
        elements.push({{ data: {{ id: n.id, label: n.label, op_type: n.op_type, kind: 'op', type: 'node', parent: group.id, groupId: group.id }} }});
      }}
    }}
  }}
  const edgeSeen = new Set();
  for (const e of viewerGraph.edges) {{
    const srcOwner = viewerGraph.owner[e.src];
    const dstOwner = viewerGraph.owner[e.dst];
    let src, dst;
    if (srcOwner && srcOwner === dstOwner) {{
      if (!expandedGroups.has(srcOwner)) continue;
      src = e.src; dst = e.dst;
    }} else {{
      src = endpointFor(e.src, 'src');
      dst = endpointFor(e.dst, 'dst');
    }}
    if (!src || !dst || src === dst) continue;
    const key = `${{src}}->${{dst}}:${{e.tensor || ''}}`;
    if (edgeSeen.has(key)) continue;
    edgeSeen.add(key);
    elements.push({{ data: {{ id: `edge:${{edgeSeen.size}}`, source: src, target: dst, label: e.tensor || '', srcRaw: e.src, dstRaw: e.dst }} }});
  }}
  return elements;
}}
function renderGraph() {{
  const elements = visibleElements();
  if (cy) cy.destroy();
  cy = cytoscape({{
    container: document.getElementById('graph-viewer'),
    elements,
    wheelSensitivity: 0.18,
    minZoom: 0.08,
    maxZoom: 3,
    style: [
      {{ selector: 'node', style: {{ 'background-color': '#173456', 'border-color': '#7dd3fc', 'border-width': 1.5, 'color': '#e5eefb', 'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center', 'font-size': 11, 'text-wrap': 'wrap', 'text-max-width': 110, 'width': 112, 'height': 46 }} }},
      {{ selector: 'node[type="group"]', style: {{ 'shape': 'round-rectangle', 'background-color': '#172554', 'border-color': '#60a5fa', 'font-weight': 700, 'width': 150, 'height': 58 }} }},
      {{ selector: 'node[type="boundary"]', style: {{ 'shape': 'round-rectangle', 'background-color': '#123524', 'border-color': '#34d399', 'width': 110 }} }},
      {{ selector: 'node[type="node"]', style: {{ 'background-color': '#102033', 'border-color': '#64748b', 'font-size': 10 }} }},
      {{ selector: ':parent', style: {{ 'background-opacity': 0.14, 'background-color': '#334155', 'border-style': 'dashed', 'padding': 18 }} }},
      {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#496580', 'target-arrow-color': '#7dd3fc', 'width': 1.5, 'label': 'data(label)', 'font-size': 8, 'color': '#a9bdd4', 'text-background-color': '#07101d', 'text-background-opacity': 0.9, 'text-background-padding': 2 }} }},
      {{ selector: '.selected', style: {{ 'border-width': 4, 'border-color': '#fbbf24' }} }}
    ],
    layout: {{ name: 'dagre', rankDir: 'LR', nodeSep: 50, rankSep: 90, edgeSep: 16, fit: true, padding: 40 }}
  }});
  cy.on('tap', 'node', event => {{
    const data = event.target.data();
    cy.elements().removeClass('selected');
    event.target.addClass('selected');
    if (data.groupId) selectGroup(data.groupId);
    else if (data.type === 'boundary') selectBoundary(data.id);
  }});
  cy.on('dbltap', 'node[type="group"]', event => toggleGroup(event.target.data('groupId')));
}}
function fitGraph() {{ if (cy) cy.fit(undefined, 40); }}
function toggleGroup(id) {{
  if (!isViewerGroup(id)) return;
  if (expandedGroups.has(id)) expandedGroups.delete(id); else expandedGroups.add(id);
  renderGraph();
  selectGroup(id);
}}
function expandAllGroups() {{ viewerGraph.groups.forEach(g => expandedGroups.add(g.id)); renderGraph(); }}
function collapseAllGroups() {{ expandedGroups.clear(); renderGraph(); }}
function selectBoundary(id) {{
  document.getElementById('detail-view').innerHTML = `<h3>${{id === 'graph:inputs' ? 'Inputs' : 'Outputs'}}</h3><p>Boundary placeholder for currently collapsed graph edges.</p>`;
}}
function selectGroup(id) {{
  const g = groups.get(id);
  if (!g) return;
  selectedGroupId = id;
  document.querySelectorAll('[data-group-id]').forEach(el => el.classList.toggle('active', el.getAttribute('data-group-id') === id));
  markTree(id);
  const items = (g.items || []).map(item => `<li>${{esc(item)}}</li>`).join('');
  const toggle = isViewerGroup(id) ? `<button class="toolbtn" onclick="toggleGroup('${{esc(id)}}')">${{expandedGroups.has(id) ? 'Collapse' : 'Expand'}} in graph</button>` : '';
  const nodes = g.nodes ? `<p><b>Nodes:</b> ${{esc(g.nodes.join(', '))}}</p>` : '';
  document.getElementById('detail-view').innerHTML = `
    <h3>${{esc(g.title)}}</h3>
    <p>${{esc(g.summary)}}</p>
    <div class="toolbar">${{toggle}}</div>
    <div class="metrics">${{renderMetrics(g.metrics)}}</div>
    ${{nodes}}
    ${{items ? `<ul>${{items}}</ul>` : ''}}
    ${{renderTable(g.table)}}
  `;
}}
document.querySelectorAll('[data-group-id]').forEach(el => {{
  el.addEventListener('click', () => selectGroup(el.getAttribute('data-group-id')));
  el.addEventListener('keydown', event => {{
    if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); selectGroup(el.getAttribute('data-group-id')); }}
  }});
}});
renderTree();
renderGraph();
selectGroup('model:summary');
</script>
</main>
</body>
</html>"""

def write_html(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_html(report))
