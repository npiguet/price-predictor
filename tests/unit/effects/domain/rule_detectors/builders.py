"""Catalogue scenarios reduced to what the detectors read."""

from __future__ import annotations

from effects.domain.rule_catalogue import (
    DetectorDeclaration,
    Polarity,
    Scenario,
    ScenarioCatalogue,
    ScenarioVariant,
)


def scenario(sid: str, rules: tuple[str, ...], kind: str, detector) -> Scenario:
    variants = (
        ScenarioVariant("base", Polarity.FIRES, "", "", is_base=True),
        ScenarioVariant("near", Polarity.DOES_NOT_FIRE, "one fact", ""),
    )
    return Scenario(sid, sid.split("-")[0], rules, kind, variants, detector)


def declaration(family: str, kind: str, needs=(), **params) -> DetectorDeclaration:
    return DetectorDeclaration(family, kind, "outcome", "near miss", params, tuple(needs))


def catalogue(*scenarios: Scenario) -> ScenarioCatalogue:
    return ScenarioCatalogue(tuple(scenarios))


def effect_halves(by_kind):
    """The resolution records that are the effect half (not the activation)."""
    return [r for r in by_kind("resolution") if r.moment.value == "resolution"]
