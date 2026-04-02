"""Default FreeSurfer tissue class mappings for SynthSeg generation.

Groups FreeSurfer aparc+aseg label IDs by tissue type so that
anatomically similar structures share intensity distributions during
synthetic image generation.
"""

from __future__ import annotations

# Mapping: tissue class name → list of FreeSurfer label IDs
# Based on FreeSurfer's aparc+aseg parcellation (50-class and 115-class)
FREESURFER_TISSUE_CLASSES: dict[str, list[int]] = {
    "background": [0],
    "cerebral_wm": [2, 41, 77, 251, 252, 253, 254, 255],
    "cerebral_gm": [3, 42],
    "cerebellar_wm": [7, 46],
    "cerebellar_gm": [8, 47],
    "csf": [4, 5, 14, 15, 24, 43, 44],
    "thalamus": [10, 49],
    "caudate": [11, 50],
    "putamen": [12, 51],
    "pallidum": [13, 52],
    "hippocampus": [17, 53],
    "amygdala": [18, 54],
    "accumbens": [26, 58],
    "ventral_dc": [28, 60],
    "brainstem": [16],
    "optic_chiasm": [85],
    # Cortical parcellation labels (Desikan-Killiany atlas)
    # Left hemisphere: 1001-1035, Right hemisphere: 2001-2035
    "cortical_lh": list(range(1001, 1036)),
    "cortical_rh": list(range(2001, 2036)),
}

# Left-right label pairs for flipping remapping
# When flipping L/R, these labels must be swapped
FREESURFER_LR_PAIRS: list[tuple[int, int]] = [
    (2, 41),  # cerebral WM
    (3, 42),  # cerebral GM
    (4, 43),  # lateral ventricle
    (5, 44),  # inferior lateral ventricle
    (7, 46),  # cerebellar WM
    (8, 47),  # cerebellar GM
    (10, 49),  # thalamus
    (11, 50),  # caudate
    (12, 51),  # putamen
    (13, 52),  # pallidum
    (17, 53),  # hippocampus
    (18, 54),  # amygdala
    (26, 58),  # accumbens
    (28, 60),  # ventral DC
] + [
    (1000 + i, 2000 + i) for i in range(1, 36)
]  # cortical labels


def get_tissue_classes(
    name: str = "freesurfer",
) -> dict[str, list[int]]:
    """Return a tissue class mapping by name.

    Parameters
    ----------
    name : str
        ``"freesurfer"`` (default).

    Returns
    -------
    dict
        Mapping from tissue class names to label ID lists.
    """
    if name == "freesurfer":
        return FREESURFER_TISSUE_CLASSES
    raise ValueError(f"Unknown tissue class mapping '{name}'. Available: freesurfer")
