"""Setup script for nobrainer.

To install, run `python3 setup.py install`.
"""
import os
import sys

from setuptools import setup

# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
sys.path.append(os.path.dirname(__file__))

try:
    import versioneer

    setup_kw = {
        "version": versioneer.get_version(),
        "cmdclass": versioneer.get_cmdclass(),
    }
except ImportError:
    # see https://github.com/warner/python-versioneer/issues/192
    print("WARNING: failed to import versioneer, falling back to no version for now")
    setup_kw = {}

setup(name="nobrainer", **setup_kw)
