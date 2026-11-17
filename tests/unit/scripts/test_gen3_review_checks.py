"""``gen3_review_checks.py``: every check runs on the gen-3 fixture records and sidecars."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gen3_review_checks as grc
from gen3_review_checks_parts.convoke import is_convoke_row, mana_value, reduction
from gen3_review_checks_parts.key_checks import JoinedKeys
from gen3_review_checks_parts.model import Line
from gen3_review_checks_parts.sidecars import Sidecars
from gen3_review_checks_parts.tree_checks import tree_lines

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "effects"
SIDECARS = FIXTURES / "gen3-sidecars"
RECORDS = FIXTURES / "gen3-records.jsonl.gz"

CHECK_NAMES = (
    "keys-declared-by-two-lines",
    "acting-keys-line",
    "acting-keys-ambiguous",
    "board-lines-line",
    "board-lines-undescribed",
    "equip-halves-without-stat-change",
    "equip-halves-with-stat-change",
    "counter-halves-random-seat-countered[spell-target]",
    "counter-halves-random-seat-not-countered[stack-card-target]",
    "fizzled-halves-with-zero-targets",
    "forced-halves",
    "forced-halves-stamped-before-turn-action",
    "forced-deaths-without-unattached",
    "protection-forks-attachment-leaves-unnamed",
    "trigger-records-without-subject[DamageDealtOnce]",
    "trigger-records-without-subject[DamageAll]",
    "phase-skipped-with-unknown-phase",
    "marked-games-without-player-lost",
    "rescue-records-holding-more-than-the-outcome",
    "lethal-creatures-without-death[game-ending-combat]",
    "lethal-creatures-without-death[other-combat]",
    "library-cards-without-owns-and-controls",
    "non-token-copy-entities-with-granted-line",
    "mana-record-events-without-attribution[tapped]",
    "mana-record-events-without-attribution[mana_produced]",
    "convoke-candidates",
)


def _run() -> dict[str, tuple[int, int]]:
    lines = grc.run_checks([RECORDS], SIDECARS)
    return {line.name: (line.count, line.total) for line in lines}


def test_every_check_reports_a_count_and_a_total() -> None:
    results = _run()
    for name in CHECK_NAMES:
        assert name in results, name
        count, total = results[name]
        assert 0 <= count <= total


def test_phase_skipped_events_all_carry_the_unknown_phase() -> None:
    count, total = _run()["phase-skipped-with-unknown-phase"]
    assert (count, total) == PHASE_SKIPPED


def test_sidecar_tree_has_keys_declared_by_two_lines() -> None:
    """The fixture tree holds the current converter's sidecars wherever an
    older one declared a key twice, so it has none."""
    count, total = _run()["keys-declared-by-two-lines"]
    assert (count, total) == TREE_DUPLICATES


def test_a_key_declared_by_two_lines_is_flagged(tmp_path: Path) -> None:
    """Binding the Old Gods' sidecar with chapter I's key copied onto chapter
    II's line, as a sidecar written before the key change declared it."""
    import json

    from gen3_review_checks_parts.sidecars import keys_declared_by_two_lines

    relative = Path("cardsfolder/b/binding_the_old_gods.provenance.json")
    data = json.loads((SIDECARS / relative).read_text(encoding="utf-8"))
    data["lines"][1]["provenance"].append(data["lines"][0]["provenance"][0])
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_text(json.dumps(data), encoding="utf-8")
    line = keys_declared_by_two_lines(tmp_path)
    assert (line.count, line.total) == (1, 1)


def test_report_line_names_the_review_count() -> None:
    line = Line("phase-skipped-with-unknown-phase", 3, 3)
    assert grc.format_line(line).split()[0] == "phase-skipped-with-unknown-phase"
    assert "review=101 of 101" in grc.format_line(line)
    assert "review=n/a" in grc.format_line(Line("no-such-check", 0, 0))


PHASE_SKIPPED = (2, 2)
TREE_DUPLICATES = (0, 947)


STOKE_CONVOKE_ROW = {
    "line_index": 3,
    "line_kind": "cost reduction",
    "prose_offset": 16,
    "provenance": [{"face": 0, "trait_kind": "keyword", "index_within_kind": 0}],
    "sub_ability_links": [],
    "script_api_type": "Keyword",
    "script_param_keys": [],
    "script_text": "Convoke",
    "role_spans": [{"start": 0, "end": 7, "role": "effect"}],
}


def test_convoke_is_read_from_the_sidecar_row_whatever_its_line_kind() -> None:
    assert is_convoke_row(STOKE_CONVOKE_ROW)
    assert not is_convoke_row({**STOKE_CONVOKE_ROW, "script_text": "Flying"})


def test_convoke_cost_reduction_against_the_printed_cost() -> None:
    assert mana_value("{2}{R}{R}") == 4
    assert reduction("{R}{R}", "{2}{R}{R}") == "reduced"
    assert reduction("{2}{R}{R}", "{2}{R}{R}") == "same"
    assert reduction(None, "{2}{R}{R}") == "other"


# The tree checks, on real converted cards (tests/fixtures/gen3_review_checks_tree/README.md).

TREE = Path(__file__).resolve().parents[2] / "fixtures" / "gen3_review_checks_tree"


def _tree_copy(tmp_path: Path) -> Path:
    shutil.copytree(TREE, tmp_path / "tree")
    return tmp_path / "tree"


def _tree_counts(root: Path) -> dict[str, tuple[int, int]]:
    return {line.name: (line.count, line.total) for line in tree_lines(root)}


def _edit_sidecar(root: Path, relative: str, edit) -> None:
    path = root / "cardsfolder" / f"{relative}.provenance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _edit_text(root: Path, relative: str, old: str, new: str) -> None:
    path = root / "cardsfolder" / f"{relative}.txt"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_the_real_cards_pass_every_tree_check_but_the_spell_with_no_line(tmp_path: Path) -> None:
    counts = _tree_counts(_tree_copy(tmp_path))
    flagged = {name: n for name, (n, _) in counts.items() if n}
    assert flagged == {
        "spells-of-instants-and-sorceries-with-no-line": 1,
        "keyed-lines-with-placeholder[CARDNAME]": counts[
            "keyed-lines-with-placeholder[CARDNAME]"][0],
    }
    assert counts["spells-of-instants-and-sorceries-with-no-line"] == (1, 2)
    assert counts["meld-halves-without-face-1"] == (0, 1)
    assert counts["sidecars-raising-sidecar-error"] == (0, 6)


def test_a_keyed_line_with_empty_prose_is_flagged(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)
    bolt = "spell[1]: CARDNAME deals 3 damage to any target."
    _edit_text(root, "l/lightning_bolt", bolt, "spell[1]: ")
    _edit_sidecar(root, "l/lightning_bolt", lambda d: d["lines"][0].update(prose_offset=10))
    count, _ = _tree_counts(root)["keyed-lines-with-empty-prose"]
    assert count == 1


def test_unfilled_templates_are_flagged_each_under_its_own_name(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)
    _edit_text(root, "l/lightning_bolt", "CARDNAME deals", "ABILITY deals {C:x}")
    counts = _tree_counts(root)
    assert counts["keyed-lines-with-placeholder[ABILITY]"][0] == 1
    assert counts["keyed-lines-with-placeholder[{C:]"][0] == 1


def test_a_spell_with_a_line_is_not_flagged_and_one_dropped_is(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)
    _edit_sidecar(root, "c/crashing_wave", lambda d: d["dropped_keys"].clear())
    count, total = _tree_counts(root)["spells-of-instants-and-sorceries-with-no-line"]
    assert (count, total) == (0, 1)


def test_a_meld_half_with_no_face_1_key_is_flagged(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)

    def drop_face_1(data: dict) -> None:
        data["lines"] = [
            line for line in data["lines"] if all(k["face"] != 1 for k in line["provenance"])
        ]

    _edit_sidecar(root, "t/the_mightstone_and_weakstone", drop_face_1)
    assert _tree_counts(root)["meld-halves-without-face-1"] == (1, 1)


def test_a_class_level_line_declaring_a_kind_indexed_key_is_flagged(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)
    assert _tree_counts(root)["class-level-lines-with-kind-indexed-key"] == (0, 3)

    def build_level_2(data: dict) -> None:
        data["lines"][1]["provenance"][0]["trait_kind"] = "static"

    _edit_sidecar(root, "r/ranger_class", build_level_2)
    assert _tree_counts(root)["class-level-lines-with-kind-indexed-key"] == (1, 3)


def test_a_sidecar_declaring_a_key_twice_is_counted_not_raised(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)
    _edit_sidecar(
        root, "b/binding_the_old_gods",
        lambda d: d["lines"][1]["provenance"].append(d["lines"][0]["provenance"][0]),
    )
    assert _tree_counts(root)["sidecars-raising-sidecar-error"] == (1, 6)


def test_the_script_runs_the_tree_checks_alone_without_records(capsys) -> None:
    assert grc.main(["--converted-dir", str(TREE)]) == 0
    out = capsys.readouterr().out
    assert "meld-halves-without-face-1" in out
    assert "keys-declared-by-two-lines" in out
    assert "equip-halves" not in out


def _key(script_file: str, kind: str = "keyword", index: int = 0, **extra) -> dict:
    return {"script_file": script_file, "face": 0, "trait_kind": kind,
            "index_within_kind": index, **extra}


def _joined(acting: list[dict], board: list[dict]) -> dict[str, tuple[int, int]]:
    check = JoinedKeys(Sidecars(SIDECARS))
    entity = {"lines": [{"keys": [key]} for key in board]}
    check.feed({"ability": acting, "state": {"entities": [entity]}})
    return {line.name: (line.count, line.total) for line in check.lines() if line.total}


AIR_ELEMENTAL = "cardsfolder/a/air_elemental.txt"
SAGA = "cardsfolder/b/binding_the_old_gods.txt"


def test_a_built_key_of_a_one_line_keyword_resolves_to_its_line() -> None:
    built = _key(AIR_ELEMENTAL, built={"kind": "trigger", "index": 0})
    counts = _joined([built], [])
    assert counts["acting-keys-line"] == (1, 1)
    assert counts["acting-keys-undescribed"] == (0, 1)


def test_a_sagas_bare_key_is_ambiguous_acting_and_a_line_per_chapter_on_the_board() -> None:
    bare = _key(SAGA)
    counts = _joined([bare], [bare])
    assert counts["acting-keys-ambiguous"] == (1, 1)
    assert counts["board-lines-line"] == (3, 3)
    assert counts["board-lines-ambiguous"] == (0, 3)


def test_a_class_level_line_declaring_a_built_key_is_flagged(tmp_path: Path) -> None:
    root = _tree_copy(tmp_path)

    def build_level_2(data: dict) -> None:
        data["lines"][1]["provenance"][0]["built"] = {"kind": "static", "index": 0}

    _edit_sidecar(root, "r/ranger_class", build_level_2)
    assert _tree_counts(root)["class-level-lines-with-kind-indexed-key"] == (1, 3)
