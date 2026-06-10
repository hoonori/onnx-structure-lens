import unittest

from structure_lens.analyzer import analyze_model
from structure_lens.render import render_markdown


class StructureLensAnalysisTests(unittest.TestCase):
    def test_tiny_transformer_report_has_core_sections(self):
        report = analyze_model("examples/tiny_transformer_block.json").to_dict()
        self.assertEqual(report["graph"]["node_count"], 15)
        self.assertGreater(report["graph"]["known_flops"], 0)
        self.assertTrue(report["topology"]["critical_path"])
        self.assertTrue(any(p["key"].startswith("S=") for p in report["shape_params"]))
        self.assertTrue(any(s["kind"] == "attention" for s in report["subgroups"]))
        self.assertTrue(any(s["kind"] == "residual" for s in report["subgroups"]))

    def test_param_impact_ranks_sequence_or_channel(self):
        report = analyze_model("examples/tiny_transformer_block.json").to_dict()
        params = [p["param"] for p in report["param_impacts"]]
        self.assertTrue(any(p.startswith("S=") for p in params))
        self.assertTrue(any(p.startswith("C=") or p.startswith("N=") for p in params))

    def test_markdown_renderer_mentions_recommendations(self):
        report = analyze_model("examples/tiny_transformer_block.json").to_dict()
        md = render_markdown(report)
        self.assertIn("Recommendations", md)
        self.assertIn("Parameter Impact", md)
        self.assertIn("Detected Subgroups", md)

    def test_what_if_sequence_override_reports_delta(self):
        report = analyze_model("examples/tiny_transformer_block.json", what_if={"S": 256}).to_dict()
        self.assertEqual(len(report["what_if"]), 1)
        summary = report["what_if"][0]
        self.assertEqual(summary["overrides"], {"S": 256})
        self.assertGreater(summary["known_flops_after"], summary["known_flops_before"])
        self.assertGreater(summary["changed_node_count"], 0)
        md = render_markdown(report)
        self.assertIn("What-if Analysis", md)
        self.assertIn("S=256", md)

    def test_html_renders_interactive_group_graph_not_raw_markdown(self):
        from structure_lens.render import render_html

        report = analyze_model("examples/tiny_transformer_block.json", what_if={"S": 256}).to_dict()
        doc = render_html(report)
        self.assertIn('id="graph-viewer"', doc)
        self.assertIn('id="detail-view"', doc)
        self.assertIn('id="lens-data"', doc)
        self.assertIn('TensorBoard-style expandable graph viewer', doc)
        self.assertIn('function selectGroup', doc)
        self.assertIn('function toggleGroup', doc)
        self.assertIn('data-group-id="subgroup:AttentionCore:softmax"', doc)
        self.assertIn('data-group-id="whatif:S=256"', doc)
        self.assertIn('<table', doc)
        self.assertNotIn('| Param | Dim | Tensor axes | Nodes touched |', doc)

    def test_html_overview_and_detail_graphs_follow_topology_edges(self):
        from structure_lens.render import render_html

        report = analyze_model("examples/tiny_transformer_block.json", what_if={"S": 256}).to_dict()
        doc = render_html(report)
        self.assertIn('id="graph-viewer"', doc)
        self.assertIn('https://cdn.jsdelivr.net/npm/cytoscape', doc)
        self.assertIn('https://cdn.jsdelivr.net/npm/dagre', doc)
        self.assertIn('cytoscape-dagre', doc)
        self.assertIn('function renderGraph', doc)
        self.assertIn('function toggleGroup', doc)
        self.assertIn('expandedGroups', doc)
        self.assertIn('viewerGraph', doc)
        self.assertIn('subgroup:AttentionCore:softmax', doc)
        self.assertIn('subgroup:LinearChain:ln2..fc2', doc)
        self.assertIn('qk_scores', doc)
        self.assertIn('softmax', doc)
        self.assertIn('attn_ctx', doc)
        self.assertIn('"src": "qk_scores", "dst": "softmax"', doc)
        self.assertIn('"src": "softmax", "dst": "attn_ctx"', doc)

    def test_viewer_graph_decomposes_groups_that_would_create_collapsed_cycles(self):
        from structure_lens.render import _viewer_graph_data

        report = {
            "topology": {
                "topological_order": ["a1", "b1", "a2", "b2"],
                "depth_by_node": {"a1": 0, "b1": 1, "a2": 2, "b2": 3},
                "edges": [("a1", "b1", "t1"), ("b1", "a2", "t2"), ("a2", "b2", "t3")],
            },
            "node_costs": [
                {"node": "a1", "op_type": "Op"},
                {"node": "b1", "op_type": "Op"},
                {"node": "a2", "op_type": "Op"},
                {"node": "b2", "op_type": "Op"},
            ],
        }
        groups = [
            {"id": "subgroup:A", "title": "A", "kind": "pattern", "nodes": ["a1", "a2"]},
            {"id": "subgroup:B", "title": "B", "kind": "pattern", "nodes": ["b1", "b2"]},
        ]
        viewer = _viewer_graph_data(report, groups)
        self.assertTrue(viewer["decomposed_groups"])
        self.assertTrue(any(g["source_group"] == "subgroup:A" for g in viewer["groups"]))
        self.assertTrue(any(g["source_group"] == "subgroup:B" for g in viewer["groups"]))
        self.assertFalse(_has_cycle(viewer["collapsed_edges"]))

    def test_html_has_file_structure_navigator_and_focus_hooks(self):
        from structure_lens.render import render_html

        report = analyze_model("examples/tiny_transformer_block.json", what_if={"S": 256}).to_dict()
        doc = render_html(report)
        self.assertIn('id="graph-tree"', doc)
        self.assertIn('file-tree', doc)
        self.assertIn('function focusGraphItem', doc)
        self.assertIn('function renderTree', doc)
        self.assertIn('data-tree-id="subgroup:AttentionCore:softmax"', doc)
        self.assertIn('cycle-safe', doc)


def _has_cycle(edges):
    outgoing = {}
    nodes = set()
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        outgoing.setdefault(src, []).append(dst)
        nodes.update([src, dst])
    visiting = set()
    visited = set()

    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in outgoing.get(node, []):
            if visit(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in list(nodes))


if __name__ == "__main__":
    unittest.main()
