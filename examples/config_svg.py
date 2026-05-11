import hashlib
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from IPython import get_ipython
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib_inline.backend_inline import set_matplotlib_formats


def deterministic_svg(fig):
    canvas = FigureCanvasSVG(fig)

    svg = canvas.print_svg(metadata={})

    root = ET.fromstring(svg)

    ns = {
        "svg": "http://www.w3.org/2000/svg",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    for elem in root.findall(".//dc:date", ns):
        parent = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
        if parent is not None:
            for child in list(parent):
                for sub in list(child):
                    if sub.tag.endswith("date"):
                        child.remove(sub)

    svg = ET.tostring(root, encoding="unicode")

    ids = sorted(set(re.findall(r'"(m[a-f0-9]+)"', svg)))

    mapping = {}

    for old in ids:
        stable = hashlib.sha1(old.encode()).hexdigest()[:12]
        mapping[old] = f"m{stable}"

    for old, new in mapping.items():
        svg = svg.replace(f'"{old}"', f'"{new}"')
        svg = svg.replace(f'#{old}"', f'#{new}"')

    return svg


svg_formatter = get_ipython().display_formatter.formatters["image/svg+xml"]
svg_formatter.for_type(plt.Figure, deterministic_svg)

set_matplotlib_formats("svg")

plt.rcParams["svg.hashsalt"] = ""
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["path.simplify"] = False
