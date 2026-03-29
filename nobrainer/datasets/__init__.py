"""Dataset fetching utilities for various neuroimaging sources.

Each submodule provides functions to install and fetch data from a
specific source.  All require the ``[versioning]`` optional extra
(``datalad``, ``git-annex``) unless noted otherwise.

Available sources
-----------------
- :mod:`nobrainer.datasets.openneuro` — OpenNeuro raw + derivatives
"""

from __future__ import annotations


def _check_datalad():
    """Import datalad.api, raising a clear error if not available."""
    try:
        import datalad.api as dl

        return dl
    except ImportError:
        raise ImportError(
            "DataLad is required for dataset fetching. "
            "Install with: pip install 'nobrainer[versioning]'\n"
            "Also install git-annex: uv tool install git-annex"
        ) from None


__all__ = ["openneuro"]
