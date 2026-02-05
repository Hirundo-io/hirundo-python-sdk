# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hirundo"
copyright = "2024, Hirundo"  # noqa: A001  Name is specified by Sphinx
author = "Hirundo"
release = "0.2.3.post2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_click",  # Used with Typer to document CLI commands
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_multiversion",
]

autodoc_pydantic_field_show_constraints = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_model_hide_reused_validator = True

smv_tag_whitelist = r"^v.*$"
smv_branch_whitelist = "None"

templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ],
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_js_files = [
    "open-sidebar.js",
]
