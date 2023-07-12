import pathlib
import warnings

import packaging.version


needs_sphinx = '4.0'

project = 'qutip-jax'
author = 'QuTiP developers'
copyright = '2022 and later, ' + author


def _check_imported_local_package():
    """
    Warn if the importable version of the package is not in the same repository
    as this documentation.  The imported version is expected to have been made
    available by ::

        pip install -e .

    or similar; if someone is trying to use a release version to build the docs
    from a local checkout, the versions likely will not match and they'll end
    up with a chimera.
    """
    import qutip_jax
    repo_dir = pathlib.Path(__file__).absolute().parents[1]
    expected_import_dir = repo_dir / 'src' / 'qutip_jax'
    imported_dir = pathlib.Path(qutip_jax.__file__).parent
    if expected_import_dir != imported_dir:
        warnings.warn(
            "The version of qutip_jax available on the path is not "
            "from the same location as the documentation.  This may result in "
            "the documentation containing text from different sources."
            f"\nDocumentation source   : {str(repo_dir)}"
            f"\nImported package source: {str(imported_dir)}"
        )


def _version():
    _check_imported_local_package()
    filename = pathlib.Path(__file__).absolute().parents[2] / 'VERSION'
    with open(filename, "r") as file:
        version = file.read().strip()
    # Canonicalise the version format, just in case.
    return str(packaging.version.parse(version))


release = _version()

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
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
