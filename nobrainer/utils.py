"""Utilities."""


def _shapes_equal(x1, x2):
    """Return whether shapes of arrays or tensors `x1` and `x2` are equal."""
    return x1.shape == x2.shape


def _check_shapes_equal(x1, x2):
    """Raise `ValueError` if shapes of arrays or tensors `x1` and `x2` are
    unqeual.
    """
    if not _shapes_equal(x1, x2):
        raise ValueError("Shapes of both arrays or tensors must be equal.")
