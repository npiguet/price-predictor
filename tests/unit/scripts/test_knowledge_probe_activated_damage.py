"""The toughness sweep on activated any-target damage (T165)."""

from __future__ import annotations

from effect_knowledge_probes import (
    sweeps,
)

from effects.domain.records import (
    RecordKind,
)
from tests.unit.scripts.knowledge_probe_support import probabilities as _probabilities


class TestActivatedDamageSweep:
    def test_any_target_is_read_as_target_creature(self):
        text = ("AB$ DealDamage | Cost$ T | ValidTgts$ Any | NumDmg$ 1 | "
                "SpellDescription$ CARDNAME deals 1 damage to any target.")
        assert sweeps.target_creature_wording(text) == (
            "AB$ DealDamage | Cost$ T | ValidTgts$ Creature | NumDmg$ 1 | "
            "SpellDescription$ CARDNAME deals 1 damage to target creature.")
        assert sweeps.target_creature_wording("Any target.") == "Target creature."
        assert sweeps.target_creature_wording("ValidTgts$ Anything") == "ValidTgts$ Anything"

    def _ping(self, records, sidecars):
        from effects.domain.records import Moment

        for record in records:
            if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
                continue
            line = sidecars.line_or_none(record.ability[0]) if record.ability else None
            if (line is not None and line.line_kind == "activated"
                    and "ValidTgts$ Any" in (line.script_text or "")):
                return record
        raise AssertionError("no activated any-target ping in the fixture")

    def test_the_selector_takes_an_activated_ping_at_a_damaged_creature(
        self, records, sidecars, monkeypatch,
    ):
        record = self._ping(records, sidecars)
        creature = next(e for e in record.state.entities
                        if e.is_creature and e.on_battlefield)
        monkeypatch.setattr(sweeps, "_damaged_creature", lambda r: creature.id)
        selector = sweeps.SweepSelector(sidecars)
        selector.observe(record)
        rows = selector.result()["activated_damage"]
        assert rows == [{"record_id": record.record_id, "entity": creature.id}]

    def test_both_wordings_are_swept_off_a_gen3_model(self, records, sidecars, probe_model):
        record = self._ping(records, sidecars)
        creature = next(e for e in record.state.entities
                        if e.is_creature and e.on_battlefield)
        curve = sweeps.run_sweep("activated_damage", probe_model, record,
                                 {"record_id": record.record_id, "entity": creature.id})
        assert len(curve) == 2 * len(sweeps.TOUGHNESS_VALUES)
        assert {p["wording"] for p in curve} == set(sweeps.WORDINGS)
        _probabilities([p["prediction"] for p in curve] + [p["affected"] for p in curve])
        summary = sweeps.summarize([curve])
        assert [(row["wording"], row["value"]) for row in summary][:2] == [
            ("any target", 1), ("any target", 2)]
        assert all("affected" in row for row in summary)
        # The other sweeps summarize as before.
        assert sweeps.summarize([[{"value": 1, "prediction": 0.5}]]) == [
            {"value": 1, "n": 1, "prediction": 0.5}]
