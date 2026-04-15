"""Dataset specification, validation, and inspection.

Self-contained module — imports only stdlib + nibabel + numpy.
Does NOT import from ``nobrainer.*`` so it can be used by external
CI scripts without installing torch/monai.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

VALID_AXIS_CODES = frozenset("RLAPIS")


class FileStatus(enum.Enum):
    """Result of a symlink-aware file presence check."""

    PRESENT = "present"
    ANNEX_MISSING = "annex_missing"
    NOT_FOUND = "not_found"


class Severity(enum.Enum):
    """Severity level for a validation finding."""

    ERROR = "error"
    WARNING = "warning"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ValidationError:
    """A single validation finding.

    Attributes:
        field: Dotted path to the offending field, e.g. ``"entries[3].image"``.
        subject_index: Entry index, or ``None`` for spec-level errors.
        message: Human-readable description of the problem.
        severity: ``Severity.ERROR`` or ``Severity.WARNING``.
    """

    field: str
    subject_index: int | None
    message: str
    severity: Severity

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "field": self.field,
            "subject_index": self.subject_index,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclasses.dataclass
class DataSpec:
    """Dataset manifest contract.

    Attributes:
        entries: List of ``{"image": path}`` or ``{"image": path, "label": path}``
            dicts. Mirrors the MONAI-style data dicts used by
            ``nobrainer.dataset.get_dataset()``.
        expected_classes: If set, the allowed integer label values.
        spacing_range: ``(min_mm, max_mm)`` — every voxel spacing axis must
            fall within this range.
        orientation: Expected 3-character orientation code (e.g. ``"RAS"``).
        zarr_chunk_shape: Expected Zarr inner-chunk dimensions.
        zarr_levels: Expected number of Zarr pyramid levels.
    """

    entries: list[dict[str, str]] = dataclasses.field(default_factory=list)
    expected_classes: set[int] | None = None
    spacing_range: tuple[float, float] | None = None
    orientation: str | None = None
    zarr_chunk_shape: tuple[int, ...] | None = None
    zarr_levels: int | None = None

    # -- Serialization -------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> DataSpec:
        """Load a manifest JSON file and return a ``DataSpec``.

        Relative entry paths are resolved against the manifest's parent
        directory.
        """
        path = Path(path)
        with open(path) as fh:
            raw = json.load(fh)

        base_dir = path.parent

        entries: list[dict[str, str]] = []
        for entry in raw.get("entries", []):
            resolved: dict[str, str] = {}
            for key in ("image", "label"):
                if key in entry:
                    p = Path(entry[key])
                    if not p.is_absolute():
                        p = base_dir / p
                    resolved[key] = str(p)
            entries.append(resolved)

        expected_classes = raw.get("expected_classes")
        if expected_classes is not None:
            expected_classes = set(expected_classes)

        spacing_range = raw.get("spacing_range")
        if spacing_range is not None:
            spacing_range = tuple(spacing_range)

        zarr_chunk_shape = raw.get("zarr_chunk_shape")
        if zarr_chunk_shape is not None:
            zarr_chunk_shape = tuple(zarr_chunk_shape)

        zarr_levels = raw.get("zarr_levels")

        return cls(
            entries=entries,
            expected_classes=expected_classes,
            spacing_range=spacing_range,
            orientation=raw.get("orientation"),
            zarr_chunk_shape=zarr_chunk_shape,
            zarr_levels=zarr_levels,
        )

    def to_json(self, path: str | Path) -> None:
        """Write this spec to a JSON manifest file."""
        data: dict = {"entries": self.entries}
        if self.expected_classes is not None:
            data["expected_classes"] = sorted(self.expected_classes)
        if self.spacing_range is not None:
            data["spacing_range"] = list(self.spacing_range)
        if self.orientation is not None:
            data["orientation"] = self.orientation
        if self.zarr_chunk_shape is not None:
            data["zarr_chunk_shape"] = list(self.zarr_chunk_shape)
        if self.zarr_levels is not None:
            data["zarr_levels"] = self.zarr_levels
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")


# ---------------------------------------------------------------------------
# File-presence helper
# ---------------------------------------------------------------------------


def check_file_presence(path: str | Path) -> FileStatus:
    """Symlink-aware file presence check.

    Returns:
        ``FileStatus.PRESENT`` if the file exists and is readable.
        ``FileStatus.ANNEX_MISSING`` if the path is a symlink whose target
        does not exist (typical of un-fetched git-annex / DataLad content).
        ``FileStatus.NOT_FOUND`` if the path does not exist at all.
    """
    p = Path(path)
    if p.is_symlink() and not p.exists():
        return FileStatus.ANNEX_MISSING
    if p.exists():
        return FileStatus.PRESENT
    return FileStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(spec: DataSpec) -> list[ValidationError]:
    """Check a ``DataSpec`` against its constraints.

    Returns a (possibly empty) list of ``ValidationError`` findings.
    Header-only reads for spacing/orientation; label-data read only when
    ``expected_classes`` is set.
    """
    errors: list[ValidationError] = []

    # -- Phase 1: spec-level checks (no I/O) --------------------------------

    if not spec.entries:
        errors.append(
            ValidationError(
                field="entries",
                subject_index=None,
                message="Entries list is empty.",
                severity=Severity.ERROR,
            )
        )
        return errors  # nothing else to check

    for i, entry in enumerate(spec.entries):
        if "image" not in entry:
            errors.append(
                ValidationError(
                    field=f"entries[{i}]",
                    subject_index=i,
                    message="Entry is missing required 'image' key.",
                    severity=Severity.ERROR,
                )
            )

    if spec.spacing_range is not None:
        lo, hi = spec.spacing_range
        if lo <= 0 or hi <= 0:
            errors.append(
                ValidationError(
                    field="spacing_range",
                    subject_index=None,
                    message=f"Spacing range values must be positive, got ({lo}, {hi}).",
                    severity=Severity.ERROR,
                )
            )
        elif lo > hi:
            errors.append(
                ValidationError(
                    field="spacing_range",
                    subject_index=None,
                    message=f"Spacing range min ({lo}) > max ({hi}).",
                    severity=Severity.ERROR,
                )
            )

    if spec.orientation is not None:
        if len(spec.orientation) != 3 or not all(
            c in VALID_AXIS_CODES for c in spec.orientation.upper()
        ):
            errors.append(
                ValidationError(
                    field="orientation",
                    subject_index=None,
                    message=(
                        f"Invalid orientation code '{spec.orientation}'. "
                        f"Expected 3 characters from {{R,L,A,P,I,S}}."
                    ),
                    severity=Severity.ERROR,
                )
            )

    if spec.zarr_chunk_shape is not None:
        if any(d <= 0 for d in spec.zarr_chunk_shape):
            errors.append(
                ValidationError(
                    field="zarr_chunk_shape",
                    subject_index=None,
                    message="All chunk shape dimensions must be > 0.",
                    severity=Severity.ERROR,
                )
            )

    if spec.zarr_levels is not None and spec.zarr_levels < 1:
        errors.append(
            ValidationError(
                field="zarr_levels",
                subject_index=None,
                message=f"zarr_levels must be >= 1, got {spec.zarr_levels}.",
                severity=Severity.ERROR,
            )
        )

    # -- Phase 2 & 3: per-entry file presence + header validation ------------

    for i, entry in enumerate(spec.entries):
        if "image" not in entry:
            continue  # already reported in Phase 1

        # Track which files are present so Phase 3 can read them.
        image_present = _check_entry_file(
            errors, entry["image"], f"entries[{i}].image", i
        )

        label_path = entry.get("label")
        label_present = False
        if label_path is not None:
            label_present = _check_entry_file(
                errors, label_path, f"entries[{i}].label", i
            )

        # Phase 3: header validation for present files
        if image_present:
            _validate_nifti_header(
                errors, entry["image"], f"entries[{i}].image", i, spec
            )

        if label_present and spec.expected_classes is not None:
            _validate_label_classes(
                errors, label_path, f"entries[{i}].label", i, spec.expected_classes
            )

    return errors


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


def _check_entry_file(
    errors: list[ValidationError],
    path: str,
    field: str,
    index: int,
) -> bool:
    """Check file presence; append errors; return True if PRESENT."""
    status = check_file_presence(path)
    if status == FileStatus.NOT_FOUND:
        errors.append(
            ValidationError(
                field=field,
                subject_index=index,
                message=f"File does not exist: {path}",
                severity=Severity.ERROR,
            )
        )
        return False
    if status == FileStatus.ANNEX_MISSING:
        errors.append(
            ValidationError(
                field=field,
                subject_index=index,
                message=(
                    "File is a git-annex symlink but content is not present "
                    f"locally. Run: datalad get {path}"
                ),
                severity=Severity.ERROR,
            )
        )
        return False
    return True


def _validate_nifti_header(
    errors: list[ValidationError],
    path: str,
    field: str,
    index: int,
    spec: DataSpec,
) -> None:
    """Read NIfTI header (no full data load) and check spacing/orientation."""
    try:
        img = nib.load(path)
    except Exception as exc:
        errors.append(
            ValidationError(
                field=field,
                subject_index=index,
                message=f"Failed to read NIfTI header: {exc}",
                severity=Severity.ERROR,
            )
        )
        return

    # Spacing check
    if spec.spacing_range is not None:
        lo, hi = spec.spacing_range
        zooms = img.header.get_zooms()[:3]
        for axis_idx, z in enumerate(zooms):
            if not (lo <= z <= hi):
                errors.append(
                    ValidationError(
                        field=field,
                        subject_index=index,
                        message=(
                            f"Voxel spacing axis {axis_idx} = {z:.4f} mm "
                            f"is outside range [{lo}, {hi}]."
                        ),
                        severity=Severity.WARNING,
                    )
                )

    # Orientation check
    if spec.orientation is not None:
        axcodes = "".join(nib.aff2axcodes(img.affine))
        if axcodes != spec.orientation.upper():
            errors.append(
                ValidationError(
                    field=field,
                    subject_index=index,
                    message=(
                        f"Orientation is '{axcodes}', "
                        f"expected '{spec.orientation}'."
                    ),
                    severity=Severity.WARNING,
                )
            )


def _validate_label_classes(
    errors: list[ValidationError],
    path: str,
    field: str,
    index: int,
    expected_classes: set[int],
) -> None:
    """Read label volume data and check for unexpected class values."""
    try:
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        found = set(np.unique(data).astype(int).tolist())
        del data  # free memory immediately
    except Exception as exc:
        errors.append(
            ValidationError(
                field=field,
                subject_index=index,
                message=f"Failed to read label data: {exc}",
                severity=Severity.ERROR,
            )
        )
        return

    unexpected = found - expected_classes
    if unexpected:
        errors.append(
            ValidationError(
                field=field,
                subject_index=index,
                message=(
                    f"Label contains unexpected classes: {sorted(unexpected)}. "
                    f"Expected: {sorted(expected_classes)}."
                ),
                severity=Severity.WARNING,
            )
        )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_entry(path: str | Path) -> dict:
    """Return a summary dict for a single NIfTI or Zarr file.

    The dict always contains ``"path"`` and ``"file_status"`` keys.
    Shape, spacing, orientation, dtype, and file_size_bytes are populated
    only when the file is present and readable.
    """
    p = Path(path)
    status = check_file_presence(p)
    result: dict = {
        "path": str(p),
        "file_status": status.value,
        "shape": None,
        "spacing": None,
        "orientation": None,
        "dtype": None,
        "file_size_bytes": None,
    }

    if status != FileStatus.PRESENT:
        return result

    # File size (follows symlinks)
    try:
        result["file_size_bytes"] = p.stat().st_size
    except OSError:
        pass

    # Zarr store
    suffix = "".join(p.suffixes)
    if suffix == ".zarr" or p.is_dir():
        return _inspect_zarr(result, p)

    # NIfTI
    return _inspect_nifti(result, p)


def _inspect_nifti(result: dict, p: Path) -> dict:
    """Populate result dict from a NIfTI header."""
    try:
        img = nib.load(str(p))
        result["shape"] = list(img.shape[:3])
        result["spacing"] = [round(float(z), 4) for z in img.header.get_zooms()[:3]]
        result["orientation"] = "".join(nib.aff2axcodes(img.affine))
        result["dtype"] = str(img.header.get_data_dtype())
    except Exception:
        logger.debug("Could not read NIfTI header for %s", p, exc_info=True)
    return result


def _inspect_zarr(result: dict, p: Path) -> dict:
    """Populate result dict from Zarr store metadata."""
    try:
        import zarr

        store = zarr.open_group(str(p), mode="r")
        attrs = dict(store.attrs)
        result["shape"] = attrs.get("volume_shape")
        result["dtype"] = attrs.get("image_dtype")
        if "n_levels" in attrs:
            result["zarr_levels"] = attrs["n_levels"]
        if "chunk_shape" in attrs:
            result["zarr_chunk_shape"] = attrs["chunk_shape"]
        if "n_subjects" in attrs:
            result["n_subjects"] = attrs["n_subjects"]
    except Exception:
        logger.debug("Could not read Zarr metadata for %s", p, exc_info=True)
    return result
