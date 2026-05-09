# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

examples_source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")
)
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
shutil.copytree("../../examples", "examples")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GPyTorch-QR"
copyright = "2026, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

autodoc_member_order = "bysource"

numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

plot_include_source = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "GPyTorch-QR",
    },
    "show_toc_level": 2,
}

plot_html_show_formats = False
plot_html_show_source_link = False
