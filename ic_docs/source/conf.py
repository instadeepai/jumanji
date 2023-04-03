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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'jumanji_routing'
copyright = ''
author = 'Randy Brown, Ole Jorgensen, Danila Kurganov, Ugop Okoroafor, ' \
         'Marta Wolinska'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

source_suffix = {
  ".rst": "restructuredtext",
  ".txt": "restructuredtext",
  ".md": "markdown"
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.viewcode",
  "sphinx.ext.githubpages",
  "sphinx.ext.napoleon",
  "sphinx.ext.todo",
  "sphinx.ext.autosummary",
  "sphinx.ext.extlinks",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "myst_parser",
  "sphinx.ext.autosectionlabel",
  "sphinx_autodoc_typehints",
  "sphinx_copybutton",
]

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
simplify_optional_unions = False

myst_enable_extensions = [
  "amsmath",
  "colon_fence",
  "deflist",
  "dollarmath",
  "html_admonition",
  "html_image",
  "replacements",
  "smartquotes",
  "substitution",
]

pygments_style = "sphinx"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*personal_*"]

autodoc_default_flags = ["members"]

autodoc_default_options = {
  "special-members": "__init__"
}
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Enable eval_rst in markdown
def setup(app):
  app.add_config_value(
    "recommonmark_config",
    {"enable_math": True, "enable_inline_math": True, "enable_eval_rst": True},
    True,
  )
  # app.add_transform(AutoStructify)
  app.add_object_type(
    "confval",
    "confval",
    objname="configuration value",
    indextemplate="pair: %s; configuration value",
  )
