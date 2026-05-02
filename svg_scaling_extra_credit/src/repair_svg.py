import argparse
import re
from pathlib import Path
from lxml import etree

TAG_RE = re.compile(
    r"<(?:path|rect|circle|ellipse|line|polyline|polygon)\b[^<>]*/?>",
    flags=re.IGNORECASE,
)

VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')

def is_valid_child(tag_text: str) -> bool:
    try:
        wrapped = f'<svg xmlns="http://www.w3.org/2000/svg">{tag_text}</svg>'
        etree.fromstring(wrapped.encode("utf-8"))
        return True
    except Exception:
        return False

def repair_svg(text: str) -> str:
    viewbox_match = VIEWBOX_RE.search(text)
    viewbox = viewbox_match.group(1) if viewbox_match else "0 0 200 200"

    tags = TAG_RE.findall(text)
    valid_tags = []

    for tag in tags:
        if not tag.endswith("/>") and tag.startswith("<") and tag.endswith(">"):
            tag = tag[:-1] + "/>"

        if is_valid_child(tag):
            valid_tags.append(tag)

    if valid_tags:
        body = "\n  ".join(valid_tags)
    else:
        body = '<circle cx="100" cy="100" r="18" fill="gray" stroke="black" stroke-width="2"/>'

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="{viewbox}">
  <rect x="0" y="0" width="200" height="200" fill="white"/>
  {body}
</svg>
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(in_dir.glob("*.svg")):
        if path.name.startswith("~$"):
            continue
        text = path.read_text(errors="ignore")
        repaired = repair_svg(text)
        (out_dir / path.name).write_text(repaired)

    print(f"Wrote repaired SVGs to {out_dir}")

if __name__ == "__main__":
    main()
