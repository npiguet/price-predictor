"""Card names are folded once, where the converted tree is loaded (review
findings 16.2, 19.1): a converted name that keeps a capital (``dimension X``)
meets the lower-case holdout and the lower-case credits as the same card."""

from __future__ import annotations

from pathlib import Path

from effects.application.collect_coverage import build_coverage_units, count_shard
from effects.application.collector_common import read_name_list
from effects.infrastructure.card_sources import load_converted_cards
from tests.unit.effects.builders import make_entity, write_plain_shard
from tests.unit.effects.domain.test_coverage import _record

#: Converted names as the real tree spells them: they keep the capital X.
CAPITALISED = ("prototype X-8", "dimension X")


def _tree(tmp_path: Path, *names: str) -> Path:
    cards = tmp_path / "cardsfolder"
    cards.mkdir()
    for name in names:
        stem = name.lower().replace(" ", "_")
        (cards / f"{stem}.txt").write_text(
            f"name: {name}\ntypes: land\nreplacement: this land enters tapped.\n",
            encoding="utf-8",
        )
    return cards


def test_a_loaded_converted_name_is_folded(tmp_path):
    loaded = load_converted_cards(_tree(tmp_path, *CAPITALISED))

    assert sorted(loaded) == ["dimension x", "prototype x-8"]
    assert loaded["prototype x-8"].text.startswith("name: prototype X-8")


def test_a_held_out_card_that_keeps_a_capital_is_dropped_from_the_units(tmp_path):
    holdout_list = tmp_path / "holdout.txt"
    holdout_list.write_text("prototype x-8\n", encoding="utf-8")
    names = [*load_converted_cards(_tree(tmp_path, "prototype X-8", "shock"))]

    units = build_coverage_units(names, read_name_list(holdout_list))

    assert sorted(units) == ["shock"]


def test_a_record_of_a_capitalised_card_is_credited_to_its_unit(tmp_path):
    units = build_coverage_units(load_converted_cards(_tree(tmp_path, *CAPITALISED)), frozenset())
    record = _record([make_entity("E1", "dimension X")], source="E1")
    shard = write_plain_shard(tmp_path, "x", [record])

    credits = count_shard(shard).credits

    assert credits.keys() <= units.keys()
    assert credits.get("dimension x", 0) >= 1
