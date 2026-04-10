"""Tests for nobrainer.data.spec — DataSpec validation and inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from click.testing import CliRunner
import nibabel as nib
import numpy as np
import pytest

from nobrainer.cli.main import cli
from nobrainer.data.spec import (
    DataSpec,
    FileStatus,
    Severity,
    ValidationError,
    check_file_presence,
    inspect_entry,
    validate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nifti(
    tmp_path: Path,
    name: str,
    shape: tuple[int, ...] = (16, 16, 16),
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    label_classes: list[int] | None = None,
) -> str:
    """Create a NIfTI file and return its path as a string."""
    affine = np.diag([*spacing, 1.0])
    if label_classes is not None:
        data = np.random.choice(label_classes, size=shape).astype(np.int32)
    else:
        data = np.random.rand(*shape).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return str(path)


def _make_manifest(tmp_path: Path, spec_dict: dict) -> str:
    """Write a manifest JSON and return its path."""
    path = tmp_path / "manifest.json"
    with open(path, "w") as fh:
        json.dump(spec_dict, fh)
    return str(path)


# ---------------------------------------------------------------------------
# check_file_presence
# ---------------------------------------------------------------------------


class TestCheckFilePresence:
    def test_present_file(self, tmp_path: Path) -> None:
        f = tmp_path / "real.nii.gz"
        f.write_bytes(b"data")
        assert check_file_presence(f) == FileStatus.PRESENT

    def test_not_found(self, tmp_path: Path) -> None:
        assert check_file_presence(tmp_path / "nope.nii.gz") == FileStatus.NOT_FOUND

    def test_annex_missing(self, tmp_path: Path) -> None:
        link = tmp_path / "annex_file.nii.gz"
        os.symlink("/nonexistent/.git/annex/objects/xx/yy/data", str(link))
        assert check_file_presence(link) == FileStatus.ANNEX_MISSING

    def test_valid_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "real.nii.gz"
        real.write_bytes(b"data")
        link = tmp_path / "link.nii.gz"
        os.symlink(str(real), str(link))
        assert check_file_presence(link) == FileStatus.PRESENT


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_frozen(self) -> None:
        err = ValidationError(
            field="entries[0].image",
            subject_index=0,
            message="bad",
            severity=Severity.ERROR,
        )
        with pytest.raises(AttributeError):
            err.field = "other"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        err = ValidationError("f", 1, "msg", Severity.WARNING)
        d = err.to_dict()
        assert d == {
            "field": "f",
            "subject_index": 1,
            "message": "msg",
            "severity": "warning",
        }


# ---------------------------------------------------------------------------
# DataSpec serialization
# ---------------------------------------------------------------------------


class TestDataSpecSerialization:
    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "img.nii.gz")
        lbl = _make_nifti(tmp_path, "lbl.nii.gz", label_classes=[0, 1, 2])
        spec = DataSpec(
            entries=[{"image": img, "label": lbl}],
            expected_classes={0, 1, 2},
            spacing_range=(0.5, 2.0),
            orientation="RAS",
            zarr_chunk_shape=(32, 32, 32),
            zarr_levels=3,
        )
        out = tmp_path / "spec.json"
        spec.to_json(out)
        loaded = DataSpec.from_json(out)
        assert loaded.expected_classes == spec.expected_classes
        assert loaded.spacing_range == spec.spacing_range
        assert loaded.orientation == spec.orientation
        assert loaded.zarr_chunk_shape == spec.zarr_chunk_shape
        assert loaded.zarr_levels == spec.zarr_levels
        assert len(loaded.entries) == 1

    def test_relative_paths_resolved(self, tmp_path: Path) -> None:
        _make_nifti(tmp_path, "sub01.nii.gz")
        manifest = {"entries": [{"image": "sub01.nii.gz"}]}
        mpath = _make_manifest(tmp_path, manifest)
        spec = DataSpec.from_json(mpath)
        assert Path(spec.entries[0]["image"]).is_absolute()


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_spec_passes(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "img.nii.gz")
        lbl = _make_nifti(tmp_path, "lbl.nii.gz", label_classes=[0, 1, 2])
        spec = DataSpec(
            entries=[{"image": img, "label": lbl}],
            expected_classes={0, 1, 2},
            spacing_range=(0.5, 1.5),
            orientation="RAS",
        )
        errors = validate(spec)
        assert errors == []

    def test_empty_entries_error(self) -> None:
        spec = DataSpec(entries=[])
        errors = validate(spec)
        assert len(errors) == 1
        assert errors[0].severity == Severity.ERROR
        assert errors[0].field == "entries"

    def test_missing_image_key(self) -> None:
        spec = DataSpec(entries=[{"label": "/some/path"}])
        errors = validate(spec)
        assert any(
            e.severity == Severity.ERROR and "'image'" in e.message for e in errors
        )

    def test_missing_file_detected(self, tmp_path: Path) -> None:
        spec = DataSpec(entries=[{"image": str(tmp_path / "nope.nii.gz")}])
        errors = validate(spec)
        assert any(
            e.severity == Severity.ERROR and "does not exist" in e.message
            for e in errors
        )

    def test_annex_missing_detected(self, tmp_path: Path) -> None:
        link = tmp_path / "annex.nii.gz"
        os.symlink("/nonexistent/.git/annex/objects/xx/yy/data", str(link))
        spec = DataSpec(entries=[{"image": str(link)}])
        errors = validate(spec)
        assert len(errors) == 1
        assert errors[0].severity == Severity.ERROR
        assert "datalad get" in errors[0].message
        # Must not crash trying to read the header

    def test_annex_present_passes(self, tmp_path: Path) -> None:
        real = _make_nifti(tmp_path, "real.nii.gz")
        link = tmp_path / "link.nii.gz"
        os.symlink(real, str(link))
        spec = DataSpec(entries=[{"image": str(link)}])
        errors = validate(spec)
        assert errors == []

    def test_spacing_mismatch_detected(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "wide.nii.gz", spacing=(3.0, 3.0, 3.0))
        spec = DataSpec(
            entries=[{"image": img}],
            spacing_range=(0.5, 2.0),
        )
        errors = validate(spec)
        warnings = [e for e in errors if e.severity == Severity.WARNING]
        assert len(warnings) >= 1
        assert any("outside range" in w.message for w in warnings)

    def test_spacing_in_range(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "ok.nii.gz", spacing=(1.0, 1.0, 1.0))
        spec = DataSpec(
            entries=[{"image": img}],
            spacing_range=(0.5, 2.0),
        )
        errors = validate(spec)
        assert not any(
            e.severity == Severity.WARNING and "spacing" in e.message.lower()
            for e in errors
        )

    def test_orientation_mismatch(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "ras.nii.gz")  # identity affine → RAS
        spec = DataSpec(
            entries=[{"image": img}],
            orientation="LPI",
        )
        errors = validate(spec)
        warnings = [e for e in errors if e.severity == Severity.WARNING]
        assert any("Orientation" in w.message for w in warnings)

    def test_orientation_match(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "ras.nii.gz")  # identity affine → RAS
        spec = DataSpec(
            entries=[{"image": img}],
            orientation="RAS",
        )
        errors = validate(spec)
        assert not any(
            e.severity == Severity.WARNING and "Orientation" in e.message
            for e in errors
        )

    def test_label_class_violation(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "img.nii.gz")
        lbl = _make_nifti(tmp_path, "lbl.nii.gz", label_classes=[0, 1, 2, 99])
        spec = DataSpec(
            entries=[{"image": img, "label": lbl}],
            expected_classes={0, 1, 2},
        )
        errors = validate(spec)
        warnings = [e for e in errors if e.severity == Severity.WARNING]
        assert any("unexpected classes" in w.message for w in warnings)

    def test_label_classes_match(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "img.nii.gz")
        lbl = _make_nifti(tmp_path, "lbl.nii.gz", label_classes=[0, 1, 2])
        spec = DataSpec(
            entries=[{"image": img, "label": lbl}],
            expected_classes={0, 1, 2},
        )
        errors = validate(spec)
        assert not any("unexpected" in e.message for e in errors)

    def test_invalid_spacing_range(self) -> None:
        spec = DataSpec(
            entries=[{"image": "/dummy"}],
            spacing_range=(2.0, 0.5),
        )
        errors = validate(spec)
        assert any(
            e.severity == Severity.ERROR and "spacing_range" in e.field for e in errors
        )

    def test_invalid_chunk_shape(self) -> None:
        spec = DataSpec(
            entries=[{"image": "/dummy"}],
            zarr_chunk_shape=(32, 0, 32),
        )
        errors = validate(spec)
        assert any(
            e.severity == Severity.ERROR and "zarr_chunk_shape" in e.field
            for e in errors
        )


# ---------------------------------------------------------------------------
# inspect_entry
# ---------------------------------------------------------------------------


class TestInspectEntry:
    def test_present_nifti(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "vol.nii.gz", shape=(32, 32, 32))
        result = inspect_entry(img)
        assert result["file_status"] == "present"
        assert result["shape"] == [32, 32, 32]
        assert result["spacing"] is not None
        assert result["orientation"] == "RAS"
        assert result["file_size_bytes"] > 0

    def test_not_found(self, tmp_path: Path) -> None:
        result = inspect_entry(tmp_path / "nope.nii.gz")
        assert result["file_status"] == "not_found"
        assert result["shape"] is None

    def test_annex_missing(self, tmp_path: Path) -> None:
        link = tmp_path / "annex.nii.gz"
        os.symlink("/nonexistent/.git/annex/objects/xx", str(link))
        result = inspect_entry(link)
        assert result["file_status"] == "annex_missing"
        assert result["shape"] is None


# ---------------------------------------------------------------------------
# CLI: validate
# ---------------------------------------------------------------------------


class TestCLIValidate:
    def test_valid_manifest_exit_zero(self, tmp_path: Path) -> None:
        _make_nifti(tmp_path, "img.nii.gz")
        manifest = _make_manifest(tmp_path, {"entries": [{"image": "img.nii.gz"}]})
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", manifest])
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    def test_invalid_manifest_exit_nonzero(self, tmp_path: Path) -> None:
        manifest = _make_manifest(
            tmp_path, {"entries": [{"image": "nonexistent.nii.gz"}]}
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", manifest])
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        _make_nifti(tmp_path, "img.nii.gz")
        manifest = _make_manifest(tmp_path, {"entries": [{"image": "img.nii.gz"}]})
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", manifest, "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)

    def test_annex_missing_shows_datalad_get(self, tmp_path: Path) -> None:
        link = tmp_path / "annex.nii.gz"
        os.symlink("/nonexistent/.git/annex/objects/xx", str(link))
        manifest = _make_manifest(tmp_path, {"entries": [{"image": "annex.nii.gz"}]})
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", manifest])
        assert result.exit_code == 1
        assert "datalad get" in result.output


# ---------------------------------------------------------------------------
# CLI: inspect
# ---------------------------------------------------------------------------


class TestCLIInspect:
    def test_single_nifti(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "vol.nii.gz")
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", img])
        assert result.exit_code == 0
        assert "PRESENT" in result.output

    def test_manifest_json(self, tmp_path: Path) -> None:
        _make_nifti(tmp_path, "img.nii.gz")
        manifest = _make_manifest(tmp_path, {"entries": [{"image": "img.nii.gz"}]})
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", manifest])
        assert result.exit_code == 0
        assert "1 entries" in result.output

    def test_json_output_has_file_status(self, tmp_path: Path) -> None:
        img = _make_nifti(tmp_path, "vol.nii.gz")
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", img, "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["file_status"] == "present"

    def test_directory_scan(self, tmp_path: Path) -> None:
        _make_nifti(tmp_path, "a.nii.gz")
        _make_nifti(tmp_path, "b.nii.gz")
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(tmp_path)])
        assert result.exit_code == 0
        assert "2 entries" in result.output
