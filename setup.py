"""Setup script for nobrainer.

To install, run `python3 setup.py install`.
"""
from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

setup(version=version, cmdclass=cmdclass)
