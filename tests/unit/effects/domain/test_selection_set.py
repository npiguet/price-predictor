"""The frozen selection set: its JSON, its digest, its file and the seeded balance."""

from __future__ import annotations

import json
import math

import pytest

from effects.domain.selection_set import (
    BALANCE_RATIO,
    FrozenGrid,
    GridCell,
    LeftOut,
    ProbeRecord,
    SampleRecord,
    SelectionSet,
    SelectionSetError,
    TaggedRecord,
    balanced,
    error_of,
    error_value,
    item_key,
    load_selection_set,
    selection_set_path,
    write_selection_set,
)


def _selection(**changes) -> SelectionSet:
    fields = dict(
        corpus_digest="abc123",
        r=BALANCE_RATIO,
        halves={"game-disjoint": {"selection": ("g1", "g2"), "reporting": ("g3",)}},
        probes={"game-disjoint": {"selection": {
            "dies": (ProbeRecord("r1", "g1", "E5", 1.0), ProbeRecord("r2", "g2", "E6", 0.0)),
        }}},
        probe_kinds={"dies": "binary"},
        grids={"game-disjoint": {"selection": {"combat": FrozenGrid(
            (GridCell("1x1", 1.0), GridCell("1x2", 0.0)),
            (TaggedRecord("r3", "g1", ("E1", "E2")),),
        )}}},
        rule_tests={"game-disjoint": {"selection": {"704.5f": {
            "fires": (TaggedRecord("r4", "g1", ("E7",)),),
            "does_not_fire": (TaggedRecord("r5", "g2", ()),),
        }}}},
        deviance_samples={"game-disjoint": {"selection": (
            SampleRecord("r4", "g1", 0), SampleRecord("r4", "g1", 1),
        )}},
        standard_errors={
            item_key("game-disjoint", "selection", "rule_tests", "704.5f"): 1.5,
            item_key("game-disjoint", "selection", "deviance", "damage"): {
                "hard": 0.02, "soft": None},
        },
        left_out={"game-disjoint": {"selection": {"rule_tests": (
            LeftOut("702.2b", "untested"),),
        }}},
        settings={"min_games_per_half": 300},
    )
    fields.update(changes)
    return SelectionSet(**fields)


def test_json_round_trips_and_keeps_its_digest() -> None:
    sealed = _selection().sealed()
    back = SelectionSet.from_json(json.loads(json.dumps(sealed.to_json())))
    assert back == sealed
    assert back.computed_digest() == sealed.digest
    assert len(sealed.digest) == 64


def test_digest_moves_with_any_field_and_ignores_itself() -> None:
    one = _selection().sealed()
    assert _selection(r=2).sealed().digest != one.digest
    assert one.computed_digest() == _selection().computed_digest()


def test_record_ids_reads_every_distinct_record_of_a_half_once() -> None:
    assert _selection().record_ids("game-disjoint", "selection") == {
        "r1", "r2", "r3", "r4", "r5"}
    assert _selection().record_ids("card-disjoint", "selection") == frozenset()


def test_file_is_named_by_the_corpus_digest_and_loads_back(tmp_path) -> None:
    path = selection_set_path(tmp_path, "abc123")
    assert path.name == "selection-set-abc123.json"
    write_selection_set(path, _selection())
    loaded = load_selection_set(path, "abc123")
    assert loaded == _selection().sealed()


def test_load_refuses_missing_other_corpus_and_edited_files(tmp_path) -> None:
    path = selection_set_path(tmp_path, "abc123")
    with pytest.raises(SelectionSetError, match="no selection set"):
        load_selection_set(path, "abc123")
    write_selection_set(path, _selection())
    with pytest.raises(SelectionSetError, match="frozen against corpus digest"):
        load_selection_set(path, "other")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["r"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SelectionSetError, match="edited"):
        load_selection_set(path, "abc123")


def test_infinite_standard_errors_are_written_as_null() -> None:
    assert error_value(math.inf) is None
    assert error_of(None) == math.inf
    assert error_of(error_value(0.07)) == 0.07


def test_balance_keeps_r_times_the_smaller_side_by_a_seeded_draw() -> None:
    """Story 8 acceptance 9: 400 ``fires`` against 100 at r = 2 keep 200."""
    groups = {"fires": [f"f{i}" for i in range(400)],
              "does_not_fire": [f"n{i}" for i in range(100)]}
    out = balanced(groups, 2, seed=7, salt="704.5f", key=str)
    assert len(out["fires"]) == 200
    assert out["does_not_fire"] == groups["does_not_fire"]
    assert set(out["fires"]) <= set(groups["fires"])
    # Kept in their original order, and the same under the same seed.
    assert out["fires"] == sorted(out["fires"], key=groups["fires"].index)
    assert balanced(groups, 2, seed=7, salt="704.5f", key=str) == out
    assert balanced(groups, 2, seed=8, salt="704.5f", key=str)["fires"] != out["fires"]


def test_balance_ignores_an_empty_group() -> None:
    out = balanced({"a": [1, 2, 3], "b": []}, 1, seed=0, salt="", key=str)
    assert out == {"a": [1, 2, 3], "b": []}
