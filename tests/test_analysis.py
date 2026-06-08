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
        self.assertIn('id="overview-graph"', doc)
        self.assertIn('id="detail-view"', doc)
        self.assertIn('id="lens-data"', doc)
        self.assertIn('function selectGroup', doc)
        self.assertIn('data-group-id="subgroup:AttentionCore:softmax"', doc)
        self.assertIn('data-group-id="whatif:S=256"', doc)
        self.assertIn('<table', doc)
        self.assertNotIn('| Param | Dim | Tensor axes | Nodes touched |', doc)


if __name__ == "__main__":
    unittest.main()
