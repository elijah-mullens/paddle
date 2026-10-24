import base64
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import tinyhtml5
from weasyprint import HTML

import argparse

parser = argparse.ArgumentParser(
    description="Print the combined Sphinx note with SVG equations."
)
parser.add_argument("--html", type=Path, required=True)
parser.add_argument(
    "--output", type=Path, default=Path(__file__).with_name("paddle_notes.pdf")
)
args = parser.parse_args()
source = args.html.resolve()
root = tinyhtml5.parse(source.read_text(), namespace_html_elements=False)
body = next(el for el in root.iter("div") if el.get("class") == "body")
for parent in list(body.iter()):
    for child in list(parent):
        if "headerlink" in child.get("class", "").split():
            parent.remove(child)
# Sphinx singlehtml can put a nested master below its shared asset folder.
for element in body.iter("img"):
    relative = element.get("src", "")
    asset = next(
        (
            parent / relative
            for parent in source.parents
            if (parent / relative).is_file()
        ),
        None,
    )
    if asset is None:
        raise FileNotFoundError(f"Missing figure: {relative}")
    element.set("src", asset.as_uri())
math_nodes = [el for el in body.iter() if "math" in el.get("class", "").split()]
equations = []
for el in math_nodes:
    text = "".join(el.itertext()).strip()
    display = el.tag == "div"
    equations.append(dict(tex=text[2:-2], display=display))
svgs = json.loads(
    subprocess.check_output(
        ["node", str(Path(__file__).with_name("render_math.js"))],
        input=json.dumps(equations).encode(),
    )
)
for el, svg in zip(math_nodes, svgs):
    width = re.search(r'width="([0-9.]+)ex"', svg).group(1)
    height = re.search(r'height="([0-9.]+)ex"', svg).group(1)
    align = re.search(r"vertical-align: ([^;]+)", svg).group(1)
    tail = el.tail
    el.clear()
    el.tail = tail
    if el.tag == "div":
        el.set("class", "equation")
    img = ET.SubElement(
        el,
        "img",
        {
            "src": "data:image/svg+xml;base64,"
            + base64.b64encode(svg.encode()).decode(),
            "style": f"width:{float(width)*.5}em;height:{float(height)*.5}em;vertical-align:{align};max-width:100%;",
        },
    )
css = """
@page { size: A4; margin: 18mm 17mm 19mm;
 @bottom-center {content: counter(page);font:9pt sans-serif;color:#666;}
}
body {font-family: "DejaVu Serif",serif; font-size:10pt;line-height:1.45;color:#222;}
h1,h2,h3,h4 {font-family:"DejaVu Sans",sans-serif;line-height:1.25;break-after:avoid;}
h1 {font-size:22pt;color:#173a54;margin-bottom:18pt;}
h2 {font-size:17pt;margin-top:20pt;color:#173a54;}
p {orphans:3;widows:3;}
a {color:#22577a;text-decoration:none;overflow-wrap:anywhere;}
code,pre {font-family:"DejaVu Sans Mono",monospace;font-size:8pt;}
pre {font-size:7.1pt;line-height:1.5;background:#f4f5f6;padding:9pt;white-space:pre-wrap;overflow-wrap:anywhere;}
.equation {text-align:center;margin:12pt 0;break-inside:avoid;font-size:10pt;}
table {border-collapse:collapse;width:100%;font-size:7.2pt;break-inside:avoid;margin:12pt 0;}
th,td {border-bottom:0.5pt solid #ccc;padding:5pt 2pt;text-align:left;}
th {background:#edf2f5;}
figure {break-inside:avoid;margin:0;text-align:center;}
figure img {width:auto!important;max-width:100%;height:auto;max-height:215mm;}
figcaption {font-size:9pt;text-align:left;margin-top:10pt;}
.caption-number {font-weight:bold;}
.citation {font-size:8.5pt;}
"""
html = (
    '<html><head><meta charset="utf-8"><title>Paddle technical notes</title><style>'
    + css
    + "</style></head><body>"
    + ET.tostring(body, encoding="unicode", method="html")
    + "</body></html>"
)
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
HTML(string=html, base_url=str(source.parent)).write_pdf(output)
print(f"Rendered {len(equations)} equations to {output}")
