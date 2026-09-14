"""Structural validation for downloaded HMC and sampled-MLE artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import h5netcdf
import h5py
import numpy as np

ArtifactKind = Literal["diagnosed_hmc", "raw_hmc", "sampled_mle"]
ValidationStatus = Literal["validated", "limited", "unusable", "missing"]

DIAGNOSED_GROUPS = (
    "posterior",
    "posterior_predictive",
    "sample_stats",
    "observed_data",
    "variable_diagnostic_stats",
    "sample_diagnostic_tests",
    "variable_diagnostic_tests",
)


@dataclass(frozen=True)
class FitExpectation:
    """Minimum array contract needed to use one fitted score."""

    score_variable: str
    variant_count: int
    posterior_shape: tuple[int, ...] | None = None
    predictive_variables: tuple[str, ...] = ()
    observed_variables: tuple[str, ...] = ()
    observed_arrays: tuple[tuple[str, np.ndarray], ...] = ()


@dataclass(frozen=True)
class ArtifactValidation:
    """Machine-readable validation result for one local artifact."""

    path: str
    kind: ArtifactKind
    status: ValidationStatus
    bytes: int | None
    issues: tuple[str, ...]
    errors: tuple[str, ...]
    groups: tuple[str, ...]
    score_variable: str
    score_shape: tuple[int, ...] | None
    predictive_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    observed_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    h5py_compatible: bool | None
    h5netcdf_compatible: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        payload = asdict(self)
        payload["predictive_shapes"] = dict(self.predictive_shapes)
        payload["observed_shapes"] = dict(self.observed_shapes)
        return payload


def _boundary_values(variable: h5py.Dataset) -> np.ndarray:
    if variable.ndim == 0:
        return np.asarray([variable[()]])
    first = tuple(0 for _ in variable.shape)
    last = tuple(size - 1 for size in variable.shape)
    return np.asarray([variable[first], variable[last]])


def _shape_issues(
    shape: tuple[int, ...],
    expectation: FitExpectation,
) -> list[str]:
    issues = []
    if len(shape) != 3:
        issues.append("score_not_chain_draw_variant")
    if not shape or shape[-1] != expectation.variant_count:
        issues.append("score_variant_count_mismatch")
    if expectation.posterior_shape is not None and shape != expectation.posterior_shape:
        issues.append("score_shape_mismatch")
    return issues


def _reader_compatibility(path: Path) -> tuple[bool, str | None]:
    try:
        with h5netcdf.File(path, "r"):
            pass
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


def _missing_validation(
    path: Path,
    *,
    kind: ArtifactKind,
    expectation: FitExpectation,
) -> ArtifactValidation:
    return ArtifactValidation(
        path=str(path),
        kind=kind,
        status="missing",
        bytes=None,
        issues=("file_missing",),
        errors=(),
        groups=(),
        score_variable=expectation.score_variable,
        score_shape=None,
        predictive_shapes=(),
        observed_shapes=(),
        h5py_compatible=False,
        h5netcdf_compatible=False,
    )


def validate_hmc_artifact(
    path: str | Path,
    expectation: FitExpectation,
    *,
    diagnosed: bool,
) -> ArtifactValidation:
    """Validate one diagnosed or raw HMC artifact without modifying it."""
    resolved = Path(path)
    kind: ArtifactKind = "diagnosed_hmc" if diagnosed else "raw_hmc"
    if not resolved.is_file():
        return _missing_validation(resolved, kind=kind, expectation=expectation)

    issues = []
    errors = []
    groups: tuple[str, ...] = ()
    score_shape = None
    predictive_shapes: dict[str, tuple[int, ...]] = {}
    observed_shapes: dict[str, tuple[int, ...]] = {}
    h5netcdf_compatible, reader_error = _reader_compatibility(resolved)
    if not h5netcdf_compatible:
        issues.append("h5netcdf_open_failure")
        assert reader_error is not None
        errors.append(reader_error)

    blocking = False
    try:
        with h5py.File(resolved, "r") as handle:
            groups = tuple(sorted(handle.keys()))
            posterior = handle.get("posterior")
            if posterior is None or expectation.score_variable not in posterior:
                issues.append("score_variable_missing")
                blocking = True
            else:
                score = posterior[expectation.score_variable]
                score_shape = tuple(int(size) for size in score.shape)
                shape_issues = _shape_issues(score_shape, expectation)
                issues.extend(shape_issues)
                blocking |= bool(shape_issues)
                try:
                    if not np.isfinite(_boundary_values(score)).all():
                        issues.append("score_boundary_nonfinite")
                        blocking = True
                except Exception as error:
                    issues.append("score_boundary_read_failure")
                    errors.append(f"{type(error).__name__}: {error}")
                    blocking = True

            predictive = handle.get("posterior_predictive")
            for variable_name in expectation.predictive_variables:
                if predictive is None or variable_name not in predictive:
                    issues.append(f"predictive_variable_missing:{variable_name}")
                    blocking = True
                    continue
                variable = predictive[variable_name]
                shape = tuple(int(size) for size in variable.shape)
                predictive_shapes[variable_name] = shape
                if not shape or shape[-1] != expectation.variant_count:
                    issues.append(f"predictive_variant_count_mismatch:{variable_name}")
                    blocking = True
                try:
                    _boundary_values(variable)
                except Exception as error:
                    issues.append(f"predictive_boundary_read_failure:{variable_name}")
                    errors.append(f"{type(error).__name__}: {error}")
                    blocking = True

            observed = handle.get("observed_data")
            expected_observed = dict(expectation.observed_arrays)
            required_observed = tuple(
                dict.fromkeys(
                    (*expectation.observed_variables, *expected_observed)
                )
            )
            for variable_name in required_observed:
                if observed is None or variable_name not in observed:
                    issues.append(f"observed_variable_missing:{variable_name}")
                    continue
                variable = observed[variable_name]
                shape = tuple(int(size) for size in variable.shape)
                observed_shapes[variable_name] = shape
                expected_values = expected_observed.get(variable_name)
                if expected_values is not None:
                    if shape != expected_values.shape:
                        issues.append(f"observed_shape_mismatch:{variable_name}")
                    else:
                        try:
                            if not np.array_equal(variable[:], expected_values):
                                issues.append(f"observed_value_mismatch:{variable_name}")
                        except Exception as error:
                            issues.append(f"observed_read_failure:{variable_name}")
                            errors.append(f"{type(error).__name__}: {error}")
                elif not shape or shape[-1] != expectation.variant_count:
                    issues.append(f"observed_variant_count_mismatch:{variable_name}")

            if diagnosed:
                for group_name in DIAGNOSED_GROUPS:
                    if group_name not in handle:
                        issues.append(f"diagnostic_group_missing:{group_name}")
    except Exception as error:
        issues.append("h5py_open_failure")
        errors.append(f"{type(error).__name__}: {error}")
        return ArtifactValidation(
            path=str(resolved),
            kind=kind,
            status="unusable",
            bytes=resolved.stat().st_size,
            issues=tuple(dict.fromkeys(issues)),
            errors=tuple(errors),
            groups=groups,
            score_variable=expectation.score_variable,
            score_shape=score_shape,
            predictive_shapes=tuple(sorted(predictive_shapes.items())),
            observed_shapes=tuple(sorted(observed_shapes.items())),
            h5py_compatible=False,
            h5netcdf_compatible=h5netcdf_compatible,
        )

    status: ValidationStatus
    if blocking:
        status = "unusable"
    elif issues:
        status = "limited"
    else:
        status = "validated"
    return ArtifactValidation(
        path=str(resolved),
        kind=kind,
        status=status,
        bytes=resolved.stat().st_size,
        issues=tuple(dict.fromkeys(issues)),
        errors=tuple(errors),
        groups=groups,
        score_variable=expectation.score_variable,
        score_shape=score_shape,
        predictive_shapes=tuple(sorted(predictive_shapes.items())),
        observed_shapes=tuple(sorted(observed_shapes.items())),
        h5py_compatible=True,
        h5netcdf_compatible=h5netcdf_compatible,
    )


def validate_mle_artifact(
    path: str | Path,
    expectation: FitExpectation,
) -> ArtifactValidation:
    """Validate one sampled-MLE artifact and its point estimate."""
    resolved = Path(path)
    if not resolved.is_file():
        return _missing_validation(
            resolved,
            kind="sampled_mle",
            expectation=expectation,
        )

    issues = []
    errors = []
    groups: tuple[str, ...] = ()
    score_shape = None
    h5netcdf_compatible, reader_error = _reader_compatibility(resolved)
    if not h5netcdf_compatible:
        issues.append("h5netcdf_open_failure")
        assert reader_error is not None
        errors.append(reader_error)

    blocking = False
    try:
        with h5py.File(resolved, "r") as handle:
            groups = tuple(sorted(handle.keys()))
            posterior = handle.get("posterior")
            mle = handle.get("mle")
            if posterior is None or expectation.score_variable not in posterior:
                issues.append("score_variable_missing")
                blocking = True
            else:
                score = posterior[expectation.score_variable]
                score_shape = tuple(int(size) for size in score.shape)
                shape_issues = _shape_issues(score_shape, expectation)
                issues.extend(shape_issues)
                blocking |= bool(shape_issues)
                try:
                    if not np.isfinite(_boundary_values(score)).all():
                        issues.append("score_boundary_nonfinite")
                        blocking = True
                except Exception as error:
                    issues.append("score_boundary_read_failure")
                    errors.append(f"{type(error).__name__}: {error}")
                    blocking = True
            if mle is None or expectation.score_variable not in mle:
                issues.append("mle_point_estimate_missing")
                blocking = True
            else:
                point_estimate = mle[expectation.score_variable]
                if point_estimate.ndim != 1 or point_estimate.shape[-1] != (
                    expectation.variant_count
                ):
                    issues.append("mle_point_estimate_shape_mismatch")
                    blocking = True
                try:
                    if not np.isfinite(_boundary_values(point_estimate)).all():
                        issues.append("mle_point_estimate_boundary_nonfinite")
                        blocking = True
                except Exception as error:
                    issues.append("mle_point_estimate_boundary_read_failure")
                    errors.append(f"{type(error).__name__}: {error}")
                    blocking = True
    except Exception as error:
        issues.append("h5py_open_failure")
        errors.append(f"{type(error).__name__}: {error}")
        return ArtifactValidation(
            path=str(resolved),
            kind="sampled_mle",
            status="unusable",
            bytes=resolved.stat().st_size,
            issues=tuple(dict.fromkeys(issues)),
            errors=tuple(errors),
            groups=groups,
            score_variable=expectation.score_variable,
            score_shape=score_shape,
            predictive_shapes=(),
            observed_shapes=(),
            h5py_compatible=False,
            h5netcdf_compatible=h5netcdf_compatible,
        )

    status: ValidationStatus
    if blocking:
        status = "unusable"
    elif issues:
        status = "limited"
    else:
        status = "validated"
    return ArtifactValidation(
        path=str(resolved),
        kind="sampled_mle",
        status=status,
        bytes=resolved.stat().st_size,
        issues=tuple(dict.fromkeys(issues)),
        errors=tuple(errors),
        groups=groups,
        score_variable=expectation.score_variable,
        score_shape=score_shape,
        predictive_shapes=(),
        observed_shapes=(),
        h5py_compatible=True,
        h5netcdf_compatible=h5netcdf_compatible,
    )
