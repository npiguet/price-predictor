"""Target legality read off target_legal (T167)."""

from __future__ import annotations

from effect_knowledge_probes import (
    target_restrictions,
)

from effects.domain.records import (
    PlayabilityDecisionPayload,
)
from tests.unit.scripts.knowledge_probe_support import probabilities as _probabilities


class TestTargetLegality:
    def _decisions(self, records, sidecars):
        chosen = [(r, *found) for r in records
                  if (found := target_restrictions.decision_items(r, sidecars))]
        assert chosen
        return chosen

    def test_every_card_and_player_is_an_item(self, records, sidecars):
        for record, _, items in self._decisions(records, sidecars):
            refs = {ref for ref, _ in items}
            assert {p.id for p in record.state.players} <= refs
            assert {e.id for e in record.state.entities} <= refs
            legal = set(record.payload.candidates[0].legal_targets)
            assert {ref for ref, is_legal in items if is_legal} == legal & refs

    def test_the_field_and_the_gate_are_read_at_cards_and_players(
        self, records, sidecars, probe_model,
    ):
        chosen = self._decisions(records, sidecars)
        readings = target_restrictions.read_items(probe_model, [c[0] for c in chosen])
        assert set(readings) == {"target_legal", "affected"}
        for (record, _, _), reading in zip(chosen, readings["target_legal"]):
            assert {p.id for p in record.state.players} <= set(reading)
            _probabilities(list(reading.values()))
        tables = target_restrictions.breakdowns(chosen, readings)
        cell = next(iter(tables["family"].values()))
        assert len(cell["target_legal"][0]) == len(cell["affected"][0]) > 0

    def test_flips_touch_battlefield_targets_only(self, records, sidecars):
        for flip, entity, record, edited in target_restrictions.flip_items(
                self._decisions(records, sidecars)):
            assert record.state.entity(entity).zone == "battlefield"
            assert edited is not record


def test_decisions_in_the_fixture_name_players_as_legal_targets(records):
    """The fixture holds what the legality probe reads at a player's slot."""
    assert any(set(r.payload.candidates[0].legal_targets) & {p.id for p in r.state.players}
               for r in records if isinstance(r.payload, PlayabilityDecisionPayload)
               and r.payload.candidates and r.payload.candidates[0].legal_targets)
