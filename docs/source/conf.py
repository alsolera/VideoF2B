# -*- coding: utf-8 -*-
'''
Configuration file for the Sphinx documentation builder for VideoF2B.

For a full list of configuration options, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
'''

import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'VideoF2B'
copyright = f'2022-{datetime.datetime.today().year}, {project} Documentation Authors'
author = 'Alberto Solera'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx_inline_tabs',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    'sphinxcontrib.tikz',
]

# Number captions of figures, tables, and code-blocks.
numfig = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'style_external_links': True,
    # 'logo_only': True,

    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': False,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# TODO: another SVG with the text "VideoF2B" next to the logo would look better.
# In that case, `logo_only` in `html_theme_options` should be True.
# See the sidebar logo at
# https://sphinx-rtd-theme.readthedocs.io/en/stable/index.html for the concept.
html_logo = '../../resources/art/videof2b.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Autodoc extension options -----------------------------------------------

# Mock opencv import because it won't exist in the RTD build environment.
autodoc_mock_imports = ['cv2']


# -- Autodoc typehints extension options -------------------------------------

typehints_defaults = 'comma'

# -- MathJAX extension options -----------------------------------------------

# None at the moment. For the full list of available options, see
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax


# -- TikZ extension options --------------------------------------------------

tikz_proc_suite = 'pdf2svg'

tikz_latex_preamble = r'''
\usepackage{tikz-3dplot}
\pgfplotsset{compat=1.16}
'''
