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
    if not graph["nodes"]:
        return ""
    positions = _overview_positions(graph)
    max_x = max((x for x, _ in positions.values()), default=900) + 140
    max_y = max((y for _, y in positions.values()), default=260) + 80
    edge_svg: list[str] = []
    for edge in graph["edges"]:
        src = edge["src"]
        dst = edge["dst"]
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        mid = (x1 + x2) // 2
        edge_id = f"{_h(src)}->{_h(dst)}"
        labels = ", ".join(edge.get("tensors", [])[:3])
        edge_svg.append(
            f'<path class="topo-edge" data-overview-edge="{edge_id}" d="M{x1+74},{y1} C{mid},{y1} {mid},{y2} {x2-74},{y2}" />'
            f'<text class="edge-label" x="{mid}" y="{(y1+y2)//2 - 6}">{_h(labels)}</text>'
        )
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    node_svg: list[str] = []
    for gid, (x, y) in positions.items():
        node = node_by_id[gid]
        node_svg.append(
            f'<g class="topo-node coarse-node {node["kind"]}" data-group-id="{_h(gid)}" tabindex="0">'
            f'<rect x="{x-74}" y="{y-30}" width="148" height="60" rx="14" />'
            f'<text class="node-title" x="{x}" y="{y-5}">{_h(node["title"][:20])}</text>'
            f'<text class="node-subtitle" x="{x}" y="{y+16}">{_h(node["kind"])}</text>'
            f'</g>'
        )
    return f'<svg id="overview-graph" class="topology-svg" viewBox="0 0 {max_x} {max_y}" role="img" aria-label="Coarse topology overview graph">{"".join(edge_svg)}{"".join(node_svg)}</svg>'

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
        node_graph = '<h4>internal node graph</h4>' + _render_node_graph_svg(
            group["node_graph"],
            svg_id=f"detail-node-graph-{group['id']}",
            aria_label=f"Internal graph for {group['title']}",
            node_class="mini-node",
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


def render_html(report: dict[str, Any]) -> str:
    groups = _build_group_data(report)
    overview_graph = _render_overview_graph_svg(report, groups)
    payload = json.dumps({"report": report, "groups": groups}, ensure_ascii=False).replace("</", "<\\/")
    initial_detail = _render_static_detail(groups[0])
    graph = report["graph"]
    chips = "".join(
        f'<button class="chip" data-group-id="{_h(g["id"])}">{_h(g["title"])}</button>'
        for g in groups[:18]
    )
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
main {{ padding:18px; max-width:1380px; margin:0 auto; }}
.grid {{ display:grid; grid-template-columns:minmax(0, 1.2fr) minmax(360px, .8fr); gap:18px; align-items:start; }}
.card, .detail-card {{ background:linear-gradient(180deg, #111d31, #0d1728); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 14px 40px #0007; }}
.card h2, .detail-card h3 {{ margin:0 0 10px; }}
.graph-card {{ overflow:auto; grid-column:1 / -1; }}
#overview-graph {{ min-width:900px; width:100%; height:auto; display:block; }}
.edge {{ stroke:#39536f; stroke-width:2; opacity:.9; }}
.topology-svg {{ min-width:900px; width:100%; height:auto; display:block; }}
.topo-edge {{ fill:none; stroke:#496580; stroke-width:2; opacity:.9; }}
.edge-label {{ fill:#88a2bd; font-size:10px; paint-order:stroke; stroke:#08111f; stroke-width:4px; stroke-linejoin:round; }}
.topo-node, .mini-node {{ cursor:pointer; outline:none; }}
.topo-node rect, .mini-node rect {{ fill:#13223a; stroke:#3b5d7d; stroke-width:1.3; filter: drop-shadow(0 7px 10px #0007); }}
.topo-node:hover rect, .topo-node.active rect, .mini-node:hover rect, .mini-node.active rect {{ fill:#173456; stroke:#7dd3fc; }}
.mini-node rect {{ fill:#102033; }}
.detail-card h4 {{ margin:16px 0 8px; color:#bfdbfe; }}
.node {{ cursor:pointer; outline:none; }}
.node rect {{ fill:#13223a; stroke:#3b5d7d; stroke-width:1.5; filter: drop-shadow(0 8px 12px #0008); transition:fill .15s, stroke .15s, transform .15s; }}
.node:hover rect, .node.active rect {{ fill:#173456; stroke:#7dd3fc; }}
.node.model rect {{ fill:#172554; stroke:#60a5fa; }}
.node.topology rect {{ fill:#123524; stroke:#34d399; }}
.node.shape rect {{ fill:#33245d; stroke:#a78bfa; }}
.node.impact rect, .node.cost rect {{ fill:#442817; stroke:#fbbf24; }}
.node.whatif rect {{ fill:#4a1635; stroke:#f472b6; }}
.node.attention rect {{ fill:#3b1f54; stroke:#c084fc; }}
.node.residual rect {{ fill:#482121; stroke:#fb7185; }}
.node text {{ text-anchor:middle; pointer-events:none; }}
.node-title {{ fill:#f8fafc; font-weight:700; font-size:13px; }}
.node-subtitle {{ fill:#a9bdd4; font-size:11px; text-transform:uppercase; letter-spacing:.07em; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}
.chip {{ border:1px solid #38516e; background:#0b1d33; color:#dbeafe; border-radius:999px; padding:7px 10px; cursor:pointer; }}
.chip:hover {{ border-color:#7dd3fc; color:white; }}
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
@media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} header {{ position:static; }} main {{ padding:12px; }} .card, .detail-card {{ border-radius:14px; }} }}
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
  <article class="card graph-card">
    <h2>Topology-aware overview graph</h2>
    <p>Edges follow producer → consumer tensor topology. Click a node to open its detected subgroup/detail pane; use chips for other report sections.</p>
    {overview_graph}
    <div class="chips">{chips}</div>
  </article>
  <aside id="detail-view" class="detail-card">{initial_detail}</aside>
</section>
<details class="raw-json card">
  <summary>Raw JSON data</summary>
  <pre>{_h(json.dumps(report, indent=2, ensure_ascii=False))}</pre>
</details>
<script id="lens-data" type="application/json">{payload}</script>
<script>
const lensPayload = JSON.parse(document.getElementById('lens-data').textContent);
const groups = new Map(lensPayload.groups.map(g => [g.id, g]));
function fmt(value) {{
  if (value === null || value === undefined) return '?';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  return String(value);
}}
function esc(value) {{
  return String(value).replace(/[&<>\"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[ch]));
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
function renderNodeGraph(graph) {{
  if (!graph || !graph.nodes || !graph.nodes.length) return '';
  const byDepth = new Map();
  graph.nodes.forEach(n => {{
    const depth = Number(n.depth || 0);
    if (!byDepth.has(depth)) byDepth.set(depth, []);
    byDepth.get(depth).push(n);
  }});
  const pos = new Map();
  const minDepth = Math.min(...[...byDepth.keys()], 0);
  [...byDepth.entries()].forEach(([depth, nodes]) => {{
    nodes.forEach((n, lane) => pos.set(n.name, {{x: 70 + (depth - minDepth) * 130, y: 54 + lane * 74}}));
  }});
  const maxX = Math.max(...[...pos.values()].map(p => p.x), 720) + 90;
  const maxY = Math.max(...[...pos.values()].map(p => p.y), 180) + 58;
  const paths = (graph.edges || []).map(e => {{
    const a = pos.get(e.src), b = pos.get(e.dst);
    if (!a || !b) return '';
    const mid = Math.round((a.x + b.x) / 2);
    const edgeId = `${{e.src}}->${{e.dst}}`;
    return `<path class="topo-edge" data-node-edge="${{esc(edgeId)}}" d="M${{a.x+46}},${{a.y}} C${{mid}},${{a.y}} ${{mid}},${{b.y}} ${{b.x-46}},${{b.y}}" />`;
  }}).join('');
  const nodes = graph.nodes.map(n => {{
    const p = pos.get(n.name);
    return `<g class="mini-node" data-node-name="${{esc(n.name)}}"><rect x="${{p.x-46}}" y="${{p.y-23}}" width="92" height="46" rx="10" /><text class="node-title" x="${{p.x}}" y="${{p.y-3}}">${{esc(n.name)}}</text><text class="node-subtitle" x="${{p.x}}" y="${{p.y+14}}">${{esc(n.op_type || '?')}}</text></g>`;
  }}).join('');
  return `<h4>internal node graph</h4><svg class="topology-svg" viewBox="0 0 ${{maxX}} ${{maxY}}" role="img" aria-label="internal node graph">${{paths}}${{nodes}}</svg>`;
}}
function selectGroup(id) {{
  const g = groups.get(id);
  if (!g) return;
  document.querySelectorAll('[data-group-id]').forEach(el => el.classList.toggle('active', el.getAttribute('data-group-id') === id));
  const items = (g.items || []).map(item => `<li>${{esc(item)}}</li>`).join('');
  document.getElementById('detail-view').innerHTML = `
    <h3>${{esc(g.title)}}</h3>
    <p>${{esc(g.summary)}}</p>
    <div class="metrics">${{renderMetrics(g.metrics)}}</div>
    ${{renderNodeGraph(g.node_graph)}}
    ${{items ? `<ul>${{items}}</ul>` : ''}}
    ${{renderTable(g.table)}}
  `;
}}
document.querySelectorAll('[data-group-id]').forEach(el => {{
  el.addEventListener('click', () => selectGroup(el.getAttribute('data-group-id')));
  el.addEventListener('keydown', event => {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      selectGroup(el.getAttribute('data-group-id'));
    }}
  }});
}});
selectGroup('model:summary');
</script>
</main>
</body>
</html>"""


def write_html(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_html(report))
