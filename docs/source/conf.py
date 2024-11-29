# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Sphinx import all module with autodoc but don't need these modules to build API doc
autodoc_mock_imports = [
    "pympler",
    "apifish",
    "dask",
    "tqdm",
    "astropy",
    "tifffile",
    "scipy",
    "sklearn",
    "photutils",
    "stardist",
    "csbdeep",
    "numba",
    "pylab",
    "skimage",
]

sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("../../src/postProcessing"))
sys.path.insert(0, os.path.abspath("../../src/toolbox"))
sys.path.insert(0, os.path.abspath("../../src/toolbox/file_handling"))

# -- Project information -----------------------------------------------------

project = "pyHiM"
copyright = "2022, Marcelo Nollmann, Xavier Devos"
author = "Marcelo Nollmann, Xavier Devos"

# The full version, including alpha/beta/rc tags
release = "0.9.2a0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # include documentation from docstring
    "sphinx.ext.napoleon",  # allow google or numpy docstring
    "myst_parser",  # parse markdown files to be understood by sphinx
    "sphinxcontrib.mermaid",  # mermaid extension for MyST
    "sphinx_panels",  # for creating panels like pandas or numpy main doc page
    "nbsphinx",  # include jupyter notebook file, WARNING: uncompatible with mermaid on ReadTheDocs
    "IPython.sphinxext.ipython_console_highlighting",  # Resolve highlighting "literal_block" bug
]

mermaid_output_format = "png"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_context = {"default_mode": "light"}

html_theme_options = {
    "repository_url": "https://github.com/marcnol/pyHiM",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "path_to_docs": "docs",
    "logo_only": True,
}

html_logo = "_static/logo_pyHiM.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# You can use header bookmark links, locally;
# [](#header-anchor), or cross-file [](path/to/file.md#header-anchor).
# To achieve this, use the myst_heading_anchors = DEPTH configuration option,
# where DEPTH is the depth of header levels for which you wish to generate links.
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=anchor#auto-generated-header-anchors
myst_heading_anchors = 2
