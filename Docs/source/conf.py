import os
import sys
import sphinx_gallery

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information --
project = 'Deep learning library'
copyright = '2024, Aatu Selkee'
author = 'Aatu Selkee'
release = '1.0.0'

# -- General configuration --
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]

modindex_common_prefix = ["DLL."]
add_module_names = True
templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = ["torch", "numpy", "scipy", "matplotlib", "cvxopt", "scienceplots"]

# -- Options for HTML output --
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 2,  # IMPORTANT
}

# -- Sphinx-Gallery Configuration --
sphinx_gallery_conf = {
    "examples_dirs": ["../../Tests"],  # Path to your example scripts
    "gallery_dirs": ["auto_examples"],  # Where generated output will be stored
    "filename_pattern": r".*\.py$",  # Include all scripts
    "ignore_pattern": r".*NO_DOCS_.*\.py$",
    "image_scrapers": ("matplotlib",),  # Capture Matplotlib plots automatically
    "plot_gallery": True,  # Enable automatic execution and embedding of plots
    # "default_thumb_file": "_static/Logo.png",  # Optional: default thumbnail
    "default_thumb_file": os.path.abspath("_static/Logo.png"),
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": "DLL",
    "reference_url": {"DLL": None},
}
