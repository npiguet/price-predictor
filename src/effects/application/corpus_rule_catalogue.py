"""The rules catalogue as one ``build-corpus`` run reads it, once (FR-060a)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from effects.domain.rule_catalogue import MapVerdict, RuleMapEntry, ScenarioCatalogue

__all__ = ["RuleCatalogue"]


@dataclass(frozen=True, slots=True)
class RuleCatalogue:
    """``rules/scenarios/`` and ``rules/heads/`` as one build read them (FR-060a).

    Budgets, the manifest's map snapshot and every reader after it take a
    rule's verdict and outputs from here, never from the files again.
    """

    scenarios: ScenarioCatalogue
    #: ``rule -> RuleMapEntry`` for every heads line.
    heads: dict[str, RuleMapEntry]
    #: ``rules_catalogue_io.heads_digest``: the map's content hash.
    digest: str

    @classmethod
    def read(cls, rules_dir: Path) -> RuleCatalogue:
        from effects.infrastructure.rules_catalogue_io import (
            heads_digest,
            load_heads,
            load_scenarios,
        )

        return cls(load_scenarios(rules_dir), load_heads(rules_dir), heads_digest(rules_dir))

    @classmethod
    def empty(cls) -> RuleCatalogue:
        """No scenario and no map: nothing is tagged, nothing budgeted."""
        return cls(ScenarioCatalogue(()), {}, "")

    @property
    def budgeted(self) -> frozenset[str]:
        """The rules the map marks ``trained`` (FR-054)."""
        return frozenset(
            rule for rule, entry in self.heads.items() if entry.verdict is MapVerdict.TRAINED)

    def snapshot(self, rules: Iterable[str]) -> dict[str, RuleMapEntry]:
        """The manifest's ``rule_map``: every tagged rule's verdict and outputs,
        ``missing`` for one no heads line names."""
        return {
            rule: self.heads.get(rule) or RuleMapEntry(rule=rule, verdict=MapVerdict.MISSING)
            for rule in sorted(set(rules))
        }
