"""The knowledge probes' board abilities and a granted keyword on a gen-3 board."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from effect_knowledge_probes import (
    board,
    ladder,
)
from effect_knowledge_probes.board_edits import BoardEdit

from tests.unit.scripts.knowledge_probe_support import FIXTURES

# ── board abilities ─────────────────────────────────────────────────────


def _key(trait_kind: str, face: int = 0):
    from effects.domain.provenance import ProvenanceKey

    return ProvenanceKey("cardsfolder/x/x.txt", face, trait_kind, 0)


class _Sidecars:
    """A sidecar cache over made-up lines: ``get`` for a card, ``line_or_none`` for a key."""

    def __init__(self, lines, api_by_kind=None):
        self.lines = lines
        self.api_by_kind = api_by_kind or {}

    def get(self, script_file):
        return SimpleNamespace(lines=self.lines)

    def line_or_none(self, key):
        api = self.api_by_kind.get(key.trait_kind)
        return None if api is None else _line_of(api)


def _line_of(api, *keys):
    from effects.domain.provenance import SidecarLine

    # A keyword line's script is its keyword; any other line's, a script.
    text = "Flying" if api == "Keyword" else f"AB$ {api}"
    return SidecarLine(
        line_index=0, line_kind="static", provenance=keys,
        script_api_type=api, script_text=text, prose_offset=0,
    )


def _legality(*forbidden_kinds):
    from effects.domain.records import PlayabilitySubkind, RecordKind

    return SimpleNamespace(
        kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
        payload=SimpleNamespace(forbidden=[
            SimpleNamespace(responsible_static=(_key(kind),)) for kind in forbidden_kinds]),
    )


class TestBoard:
    def test_legality_is_split_by_the_restricting_static(self):
        sidecars = _Sidecars([], {"static": "CantBlock", "keyword": "Keyword"})
        assert board.subset_of(_legality("static", "keyword"), sidecars) == "legality-static"
        assert board.subset_of(_legality("keyword"), sidecars) == "legality-keyword"
        assert board.subset_of(_legality(), sidecars) == "legality-none"
        assert board.subset_of(_legality("trigger"), sidecars) is None

    def test_collect_stops_each_subset_at_its_quota(self):
        sidecars = _Sidecars([], {"static": "CantBlock"})
        chosen = board.collect([_legality()] * 5 + [_legality("static")] * 2,
                               sidecars, per_subset=3)
        assert len(chosen["legality-none"]) == 3
        assert len(chosen["legality-static"]) == 2
        assert set(chosen) == set(board.SUBSETS)

    def test_fixture_records_land_in_subsets_that_fit_them(self, records, sidecars):
        from effects.domain.records import Moment, RecordKind

        seen = set()
        for record in records:
            name = board.subset_of(record, sidecars)
            if name is None:
                continue
            seen.add(name)
            if name.startswith("legality"):
                assert record.kind is RecordKind.PLAYABILITY
            else:
                assert record.kind is RecordKind.RESOLUTION
                assert record.moment is Moment.RESOLUTION
            if name == "resolution-replacement":
                assert any(
                    "replacement" in board.unlisted_kinds(entity)
                    for entity in record.state.entities
                    if entity.zone == "battlefield" and entity.id != record.state.refs.source)
        assert {"legality-none", "resolution-none", "resolution-replacement"} <= seen

    def test_the_table_sets_subsets_side_by_side(self):
        body = {"base_loss": 0.5, "noise": {"cards": 0.01, "act": 0.2, "all": 0.2}}
        result = {
            "legality-static": {"n": {"card-disjoint": 3}, "ablation": {"blocker_legal": body}},
            "legality-none": {"n": {"card-disjoint": 4}, "ablation": {"blocker_legal": body}},
        }
        text = board.table(result)
        assert "| field | legality-static | legality-none |" in text
        assert "| blocker_legal | 0.500 | 0.500 |" in text
        assert "| blocker_legal | 0.010 | 0.010 |" in text
        assert "legality-static 3" in text

    def test_a_field_no_subset_measured_is_left_out(self):
        zero = {"base_loss": 0.0, "noise": {"cards": 0.0}}
        result = {"legality-none": {"n": {"card-disjoint": 1},
                                    "ablation": {"colors_lost": zero}}}
        assert "colors_lost" not in board.table(result)


class TestRung3ReadsTheRightOutput:
    """Rung 3 is the model's own prediction, read from the output that learned it."""

    def _outputs(self, gate_logit: float, legal_logit: float):
        import torch

        from effects.domain.effect_head_layout import FIELD_SLICES, GATE_INDEX, PER_ENTITY_WIDTH

        vector = torch.zeros(1, 1, PER_ENTITY_WIDTH)
        vector[0, 0, GATE_INDEX] = gate_logit
        vector[0, 0, FIELD_SLICES["target_legal"][0]] = legal_logit
        return vector

    @pytest.mark.parametrize("target", ["legal_target", "legal_target_protected"])
    def test_target_legality_reads_the_field_not_the_gate(self, target):
        """Gen-3 trains `target_legal` on every card and player of a targeting
        decision, 1 at a legal target and 0 elsewhere, and no gate on a
        decision record (FR-074): the field is what learned legality."""
        import torch

        outputs = self._outputs(gate_logit=-3.0, legal_logit=5.0)
        value = ladder._readout(target, outputs, None, ladder.ItemAt(0, 0))
        assert value == pytest.approx(float(torch.sigmoid(torch.tensor(5.0))))

    def test_damage_taken_reads_the_amount_alone(self):
        """Every damage_taken item took damage, so the gate's doubt about
        whether the creature was hit must not scale the head's amount."""
        import math

        from effects.domain.effect_head_layout import FIELD_SLICES

        outputs = self._outputs(gate_logit=-3.0, legal_logit=0.0)
        outputs[0, 0, FIELD_SLICES["damage_taken"][0]] = math.log(3.0)
        value = ladder._readout("damage_taken", outputs, None, ladder.ItemAt(0, 0))
        assert value == pytest.approx(3.0)


# ── a granted keyword on a gen-3 board (research M5) ────────────────────


def _is_battlefield_creature_with_lines(entity) -> bool:
    return bool(entity.lines is not None and entity.on_battlefield and entity.is_creature)


def _plain_battlefield_creature(entity) -> bool:
    """A battlefield creature with lines and neither flying nor fear, granted or lined."""
    if not _is_battlefield_creature_with_lines(entity):
        return False
    flying_or_fear = {"flying", "fear"}
    granted = flying_or_fear & set(entity.granted_temporary.keywords)
    lined = {line.keyword for line in entity.lines} & flying_or_fear
    return not granted and not lined


@pytest.fixture(scope="module")
def gen3_creature():
    """A gen-3 fixture record and one battlefield creature on it with neither
    flying nor fear nor any line of either."""
    from effects.infrastructure.shard_io import read_shard

    for record in read_shard(FIXTURES / "gen3-records.jsonl.gz"):
        for entity in record.state.entities:
            if _plain_battlefield_creature(entity):
                return record, entity.id
    raise AssertionError("no gen-3 fixture creature without flying or fear")


class TestGrantingAKeywordOnGen3:
    """A grant writes the kept flag and the granted keyword line; a keyword
    with no flag reaches the model through its line alone."""

    def _granted(self, gen3_creature, *keywords):
        record, entity_id = gen3_creature
        board = BoardEdit(record)
        for keyword in keywords:
            board = board.grant_keyword(entity_id, keyword)
        return record, board.record.state.entity(entity_id), record.state.entity(entity_id)

    @staticmethod
    def _flag(entity, record, keyword) -> float:
        from effects.domain.card_features import card_features, card_segments
        from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS

        flags = card_features(entity, record)[card_segments(entity, record)["keyword_flags"]]
        return flags[KEPT_KEYWORD_FLAGS.index(keyword)]

    def test_a_kept_keyword_gets_its_flag_and_a_granted_line(self, gen3_creature):
        from effects.domain.slot_lines import keyword_line_key, slot_lines

        record, granted, before = self._granted(gen3_creature, "flying")
        added = granted.lines[len(before.lines):]
        assert [(line.keys, line.kind, line.granted, line.fragment) for line in added] == [
            ((), "keyword", True, "Flying")]
        assert keyword_line_key("Flying") in {line.key for line in slot_lines(granted)}
        assert self._flag(granted, record, "flying") == 1.0
        assert self._flag(before, record, "flying") == 0.0

    def test_a_keyword_without_a_flag_is_read_through_its_line_alone(self, gen3_creature):
        from effects.domain.card_features import card_features
        from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
        from effects.domain.slot_lines import keyword_line_key, slot_lines

        assert "fear" not in KEPT_KEYWORD_FLAGS
        record, granted, before = self._granted(gen3_creature, "fear")
        assert card_features(granted, record) == card_features(before, record)
        assert keyword_line_key("Fear") in {line.key for line in slot_lines(granted)}

    @pytest.mark.parametrize("keyword, fragment", [
        ("first_strike", "First strike"),
        ("protection_from_green", "Protection from green"),
        ("unblockable", "CARDNAME can't be blocked."),
    ])
    def test_the_line_spells_the_keyword_as_forge_writes_it(
        self, gen3_creature, keyword, fragment,
    ):
        _record, granted, before = self._granted(gen3_creature, keyword)
        assert granted.lines[len(before.lines):][-1].fragment == fragment

    def test_granting_twice_adds_one_line(self, gen3_creature):
        _record, granted, before = self._granted(gen3_creature, "unblockable", "unblockable")
        assert len(granted.lines) == len(before.lines) + 1
        assert granted.granted_temporary.keywords.count("unblockable") == 1
