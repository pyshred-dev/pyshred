# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

project = 'PySHRED'
copyright = '2025, Kutz Research Group'
author = 'Kutz Research Group'

version = release = "v1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_show_sourcelink = False

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
    "navbar_end": ["version-switcher"],
    "switcher": {
        "json_url": "_static/versions.json",
        "version_match": version,
    },
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