"""Frozen dataclasses for every multi-value return in the package.

No bare tuples, no named tuples — every intermediate stage that returns
more than one value uses a typed record from this module. The public
`SplitResult` aggregates the user-visible subset.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    """Public return shape of `Splitter.split()`."""

    labels: np.ndarray
    side_of_label: dict[int, int]
    snapped_sources: np.ndarray
    snapped_sinks: np.ndarray
    aug_sources: Optional[np.ndarray] = None
    aug_sinks: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SnapResult:
    """Output of `snap_seeds_to_segment`."""

    snapped: np.ndarray
    moved_count: int
    elapsed_s: float


@dataclass(frozen=True)
class ArrivalGrids:
    """Per-side geodesic arrival times from the arrival-stage helper."""

    t_a: np.ndarray
    t_b: np.ndarray
    backend: str
    elapsed_s: float


@dataclass(frozen=True)
class EnforceCcReport:
    """Result of one connected-component-enforce pass for one side label."""

    label: int
    kept_components: int
    moved_to_stray: int
    elapsed_s: float


@dataclass(frozen=True)
class Resolve3Report:
    """Result of the stray-label (border-vote) resolver."""

    n_components: int
    moved_to_source: int
    moved_to_sink: int
    tie_break_count: int
    elapsed_s: float


@dataclass(frozen=True)
class SplitDiagnostics:
    """End-of-run aggregate for `SplitResult.diagnostics`."""

    backend: str
    downsample_zyx: tuple[int, int, int]
    full_shape_zyx: tuple[int, int, int]
    fg_bbox_shape_zyx: tuple[int, int, int]
    label_counts: dict[int, int]
    stages_elapsed_s: dict[str, float]
