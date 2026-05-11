"""Make notebook output deterministic."""

import matplotlib.pyplot as plt
from IPython import get_ipython
from matplotlib.figure import Figure
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg")

plt.rcParams["svg.hashsalt"] = ""
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["path.simplify"] = False

_ip = get_ipython()
_svg_formatter = _ip.display_formatter.formatters["image/svg+xml"]
_orig_svg = _svg_formatter.lookup_by_type(Figure)


def _filtered_svg(fig):
    svg = _orig_svg(fig)
    if svg:
        svg = "\n".join(
            line
            for line in svg.split("\n")
            if not line.lstrip().startswith("<dc:date>")
        )
    return svg


_svg_formatter.for_type(Figure, _filtered_svg)
