from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import analyze_model
from .render import render_markdown, write_html, write_json, write_markdown


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human-readable ONNX/graph structure analyzer")
    p.add_argument("model", help="Path to .onnx or Structure Lens .json graph")
    p.add_argument("--json", dest="json_path", help="Write full JSON report")
    p.add_argument("--markdown", "--md", dest="markdown_path", help="Write Markdown report")
    p.add_argument("--html", dest="html_path", help="Write self-contained HTML report")
    p.add_argument("--print-json", action="store_true", help="Print JSON report to stdout instead of Markdown summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_obj = analyze_model(args.model)
    report = report_obj.to_dict()
    if args.json_path:
        write_json(report, args.json_path)
    if args.markdown_path:
        write_markdown(report, args.markdown_path)
    if args.html_path:
        write_html(report, args.html_path)
    if args.print_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
