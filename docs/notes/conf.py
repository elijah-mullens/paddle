"""Standalone Sphinx configuration for Paddle's technical-note collection."""

project = "Paddle technical notes"
master_doc = "index"
extensions = ["sphinx.ext.mathjax"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
