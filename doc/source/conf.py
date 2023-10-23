import pathlib
import warnings

import packaging.version


needs_sphinx = '4.0'

project = 'qutip_jax'
author = 'QuTiP developers'
copyright = '2022 and later, ' + author


def _version():
    filename = pathlib.Path(__file__).absolute().parents[2] / 'VERSION'
    with open(filename, "r") as file:
        version = file.read().strip()
    # Canonicalise the version format, just in case.
    return str(packaging.version.parse(version))


release = _version()

extensions = [
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    # 'sphinx_gallery.gen_gallery',
    # 'sphinxcontrib.bibtex',
]

# Patterns to exclude when looking for sources in the build.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# HTML setup

html_theme = 'sphinx_rtd_theme'
# Directories for static files to be copied across after the build.
html_static_path = []


# Intersphinx setup

intersphinx_mapping = {
    'qutip': ('https://qutip.org/docs/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'cython': ('https://cython.readthedocs.io/en/latest/', None),
}

# -- Options for numpydoc ---------------------------------------

numpydoc_show_class_members = False
napoleon_numpy_docstring = True
napoleon_use_admonition_for_notes = True

# -- Options for api doc ---------------------------------------
# autosummary_generate can be turned on to automatically generate files
# in the apidoc folder. This is particularly useful for modules with
# lots of functions/classes like qutip_qip.operations. However, pay
# attention that some api docs files are adjusted manually for better illustration
# and should not be overwritten.
autosummary_generate = False
autosummary_imported_members = True

# -- Options for biblatex ---------------------------------------

# bibtex_bibfiles = ['references.bib']
# bibtex_default_style = 'unsrt'
