"""Regression for the one 3.14 behavior change that touches our code:
PEP 649 / 749 deferred annotation evaluation as the default.
"""

import dataclasses
import typing

from supervoxel_splitter import SplitResult, Splitter


def test_dataclass_fields_resolve_under_deferred_annotations():
    """`dataclasses.fields(SplitResult)` returns the declared fields even though
    annotations are no longer evaluated at class-definition time in 3.14.
    """
    field_names = {f.name for f in dataclasses.fields(SplitResult)}
    assert field_names == {
        "labels",
        "side_of_label",
        "snapped_sources",
        "snapped_sinks",
        "aug_sources",
        "aug_sinks",
        "diagnostics",
    }


def test_get_type_hints_resolves_for_split_result():
    """`typing.get_type_hints(SplitResult)` must resolve without NameError under
    deferred annotations. Catches forward-ref / lazy-eval breakage early.
    """
    hints = typing.get_type_hints(SplitResult)
    # labels and side_of_label are required; presence is the assertion
    assert "labels" in hints
    assert "side_of_label" in hints


def test_get_type_hints_resolves_for_splitter_protocol_method():
    """`typing.get_type_hints(Splitter.split)` resolves all parameter and return
    annotations. Plugin authors and IDE tooling depend on this.
    """
    hints = typing.get_type_hints(Splitter.split)
    # SplitResult must resolve as the return type
    assert hints["return"] is SplitResult
