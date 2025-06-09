# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SCMEFitting"
copyright = "2025, Moritz Sallermann"
author = "Moritz Sallermann"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core autodoc engine
    "sphinx.ext.autosummary",  # Generate summary tables automatically
    "sphinx.ext.napoleon",  # Parse Google-/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    # "sphinx_autodoc_typehints",  # Show type hints in docs
]

# If you want autosummary to generate stubs for all modules automatically:
autosummary_generate = True  # <== key setting

# Napoleon settings (if you use Google or NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Auto-generate members by default
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "groupwise",  # or 'alphabetical'
    "separate" : True,
}
autodoc_mock_imports = ["pyscme"]

add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["src/_static"]
