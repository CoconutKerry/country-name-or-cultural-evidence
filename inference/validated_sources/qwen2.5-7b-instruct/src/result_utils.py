"""Utilities for preserving directed experimental units in result analysis."""

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias


DirectedUnitKey: TypeAlias = tuple[str, str, str]
ResultGroupKey: TypeAlias = tuple[str, str, str, str]
ConditionRows: TypeAlias = dict[str, Mapping[str, Any]]


def _required_text(record: Mapping[str, Any], field: str) -> str:
    """Return a required, non-empty string field from a result record."""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Result record requires a non-empty {field!r}: {record!r}")
    return value


def directed_unit_key(record: Mapping[str, Any]) -> DirectedUnitKey:
    """Return the directed unit ``question, label country, evidence country``.

    ``country`` is the country label shown to the model, while
    ``conflict_country`` identifies the country supplying evidence in the
    conflict condition.  Reversing those countries therefore produces a
    different experimental unit.
    """
    return (
        _required_text(record, "question_id"),
        _required_text(record, "country"),
        _required_text(record, "conflict_country"),
    )


def result_group_key(record: Mapping[str, Any]) -> ResultGroupKey:
    """Return a model-specific key for one directed experimental unit."""
    question_id, country, conflict_country = directed_unit_key(record)
    return (
        _required_text(record, "model_name"),
        question_id,
        country,
        conflict_country,
    )


def group_results_by_directed_unit(
    rows: Iterable[Mapping[str, Any]],
) -> dict[ResultGroupKey, ConditionRows]:
    """Group full result rows without collapsing directed country pairs.

    Each model/unit/condition combination must occur exactly once.  Duplicate
    rows raise an error instead of silently replacing an earlier prediction.
    """
    groups: dict[ResultGroupKey, ConditionRows] = {}

    for row in rows:
        key = result_group_key(row)
        condition = _required_text(row, "condition")
        condition_rows = groups.setdefault(key, {})
        if condition in condition_rows:
            raise ValueError(
                "Duplicate result row for "
                f"model={key[0]!r}, directed_unit={key[1:]!r}, "
                f"condition={condition!r}"
            )
        condition_rows[condition] = row

    return groups
