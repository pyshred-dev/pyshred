# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import re
from pathlib import Path

project = 'PySHRED'
copyright = '2025, Kutz Research Group'
author = 'Kutz Research Group'

def get_version_from_setup():
    """Extract version from setup.py"""
    setup_py = Path(__file__).parent.parent.parent / "setup.py"
    content = setup_py.read_text()
    # Look for version="..." pattern
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    print('--------------------------------')
    print('match', match)
    print('--------------------------------')
    if match:
        return match.group(1)
    return "unknown"

version = release = get_version_from_setup()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_show_sourcelink = False

smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'

extensions = [
    "sphinx.ext.autodoc",      # pull in docstrings
    "sphinx.ext.napoleon",     # understand Google & NumPy styles
    "sphinx.ext.viewcode",     # link to your source on each docs page
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

html_theme_options = {
    "collapse_navigation": False,
}
# Tell Sphinx to recognize both .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_title = "PySHRED"