# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PySHRED'
copyright = '2025, David Ye'
author = 'David Ye'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",      # pull in docstrings
    "sphinx.ext.napoleon",     # understand Google & NumPy styles
    "sphinx.ext.viewcode",     # link to your source on each docs page
    "myst_parser",
    "sphinx_rtd_theme",
]
html_theme = "sphinx_rtd_theme"

html_theme_options = {
  "navbar_links": [
    ("Getting Started", "getting_started/index"),
    ("User Guide", "user_guide/index"),
    ("Examples",   "examples/index"),
    ("API Reference", "pyshred/index"),
  ],
  "collapse_navigation": False,
}
# Tell Sphinx to recognize both .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
