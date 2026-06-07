from __future__ import annotations

import argparse
import json

from .analyzer import analyze_model
from .render import render_markdown, write_html, write_json, write_markdown


def _parse_what_if(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected LABEL=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip().upper()
        if not key:
            raise argparse.ArgumentTypeError(f"Empty what-if label in {item!r}")
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"What-if value must be int in {item!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"What-if value must be positive in {item!r}")
        overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human-readable ONNX/graph structure analyzer")
    p.add_argument("model", help="Path to .onnx or Structure Lens .json graph")
    p.add_argument("--json", dest="json_path", help="Write full JSON report")
    p.add_argument("--markdown", "--md", dest="markdown_path", help="Write Markdown report")
    p.add_argument("--html", dest="html_path", help="Write self-contained HTML report")
    p.add_argument(
        "--what-if",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Override every axis with canonical role LABEL (e.g. S=256, B=4, C=128) and report deltas. Repeatable.",
    )
    p.add_argument("--print-json", action="store_true", help="Print JSON report to stdout instead of Markdown summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = _parse_what_if(args.what_if)
    report_obj = analyze_model(args.model, what_if=overrides or None)
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
