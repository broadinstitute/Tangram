# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from sphinx.application import Sphinx
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('../tangram'))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # this way, we don't have to install squidpy
sys.path.insert(0, os.path.abspath("_ext"))


import tangram
# -- Project information -----------------------------------------------------

project = 'Tangram'
copyright = '2021, Department of AI/ML(Research Biology), Genentech' #no copyright
author = 'Department of AI/ML(Research Biology), Genentech' #GET FROM TANGRAM

# The full version, including alpha/beta/rc tags
release = '0.4.0'  #GET FROM TANGRAM


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 'sphinx.ext.autosummary','nbsphinx_link', 'sphinx_gallery.load_style']
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]
html_logo = "_static/images/tangram.png"
html_theme_options = {"navigation_depth":4, "logo_only":True}
html_show_sphinx = False


def setup(app):
	app.add_css_file("css/custom.css")
