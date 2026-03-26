import os
import sys
import pydata_sphinx_theme

sys.path.insert(0, os.path.abspath("../.."))

project_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
doc_source_directory = os.path.join(project_directory, "docs", "source")
sys.path.insert(0, project_directory)
sys.path.insert(0, doc_source_directory)


# Project informations
def read_version():
    version_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "pysolvegn", "__version__.py"
    )
    with open(version_file, "r") as file:
        exec(file.read())
    return locals()["__version__"]


import datetime

project = "pysolvegn"
copyright = f"2026-{datetime.datetime.now().year}, Artezaru"
author = "Artezaru"
release = read_version()

# General configuration
html_theme = "pydata_sphinx_theme"
html_static_path = [
    os.path.join(doc_source_directory, "_static"),
]

extensions = [
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx.ext.autodoc",  # Automatically document the code
    "sphinx.ext.viewcode",  # Add links to the code
    "sphinx.ext.napoleon",  # Support for Google and Numpy docstring formats
    "sphinx.ext.githubpages",  # Publish the documentation on GitHub
    "sphinx.ext.autosummary",  # Generate summaries of the modules
    "sphinx.ext.mathjax",  # Render math in the documentation
    "sphinx_design",  # Add design elements to the documentation
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx_gallery.gen_gallery",  # Generate a gallery of examples
]

autosummary_generate = True

autodoc_typehints = "description"
add_module_names = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": [
        os.path.join(project_directory, "gallery"),
    ],
    "gallery_dirs": [
        os.path.join(doc_source_directory, "_gallery"),
    ],
    "backreferences_dir": os.path.join(doc_source_directory, "_gallery_backreferences"),
    "capture_repr": ["__repr__", "__str__", "_repr_html_"],
    "image_scrapers": ("matplotlib",),
    "filename_pattern": r"\.py$",
}
