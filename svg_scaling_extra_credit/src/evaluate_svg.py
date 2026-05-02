from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lxml import etree


def xml_valid(svg: str):
    try:
        root = etree.fromstring(svg.encode("utf-8"), parser=etree.XMLParser(recover=False))
        return True, root
    except Exception:
        return False, None


def render_valid(svg: str, out_png: Path | None = None):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_png) if out_png else None)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=Path, required=True)
    p.add_argument("--render_dir", type=Path, default=None)
    args = p.parse_args()
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.sample_dir.glob("*.svg"))
    rows = []
    for f in files:
        svg = f.read_text(encoding="utf-8", errors="ignore")
        ok_xml, root = xml_valid(svg)
        root_ok = bool(ok_xml and root is not None and root.tag.lower().endswith("svg"))
        closed_svg = "</svg>" in svg.lower()
        has_viewbox = bool(re.search(r"viewBox\s*=", svg))
        png_path = (args.render_dir / (f.stem + ".png")) if args.render_dir else None
        ok_render = render_valid(svg, png_path) if ok_xml else False
        rows.append({"file": f.name, "xml_valid": ok_xml, "root_svg": root_ok, "closed_svg": closed_svg, "has_viewBox": has_viewbox, "render_valid": ok_render})

    n = max(1, len(rows))
    summary = {
        "n": len(rows),
        "xml_valid_rate": sum(r["xml_valid"] for r in rows) / n,
        "root_svg_rate": sum(r["root_svg"] for r in rows) / n,
        "closed_svg_rate": sum(r["closed_svg"] for r in rows) / n,
        "render_rate": sum(r["render_valid"] for r in rows) / n,
        "has_viewBox_rate": sum(r["has_viewBox"] for r in rows) / n,
        "rows": rows,
    }
    out = args.sample_dir / "evaluation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
