""" Tests for qutip_jax.family. """

import re

from qutip_jax import family


class TestVersion:
    def test_version(self):
        pkg, version = family.version()
        assert pkg == "qutip-jax"
        assert re.match(r"\d+\.\d+\.\d+.*", version)
