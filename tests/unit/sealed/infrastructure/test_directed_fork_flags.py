"""The directed-fork startup check (FR-014; story 3 acceptance 2).

One function, ``directed_fork_error``, called by ``match-outcomes`` and
``collect-variants`` and never by ``collect-coverage``, whose
``--directed-fork-rules`` is a label only. A listed rule needs a detector
declared (not ``none``), a contrasting variant classified into a fork
operation, and a board precondition the worker registers, which one run of
``DirectedRulesMain`` lists; the JVM query is stubbed here except in the
``integration`` test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import sealed.infrastructure.effect_collection_flags as flags
from sealed.infrastructure.effect_collection_flags import (
    UNTAGGABLE_SCENARIOS,
    directed_fork_error,
    rule_readiness,
)

_DETECTOR = {
    "family": "sba_zone_move", "record_kind": "resolution", "outcome": "o",
    "near_miss": "n", "params": {}, "needs": [],
}


def _variant(label: str, operation=None) -> dict:
    variant = {"label": label, "polarity": "fires", "delta": "", "expected": ""}
    if operation is not None:
        variant["operation"] = operation
    return variant


def _scenario(rules, *, detector=_DETECTOR, operation="pt") -> dict:
    scenario = {
        "id": f"{rules[0]}-1", "rules": list(rules), "record_kind": "resolution",
        "variants": [_variant("base"), _variant("other", operation)],
    }
    if detector is not None:
        scenario["detector"] = detector
    return scenario


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    """A catalogue of one rule per failure, and one that passes."""
    scenarios = [
        _scenario(["1.1"]),
        _scenario(["1.2"], detector={"none": "no field shows it"}),
        _scenario(["1.3"], detector=None),
        _scenario(["1.4"], operation={"unclassifiable": "a different target"}),
        # 1.5 is declared in one scenario and forkable in another.
        _scenario(["1.5"], operation={"unclassifiable": "a timing"}),
        _scenario(["1.5"], detector={"none": "unread"}, operation="tap"),
        _scenario(["1.6"], detector={"same_as": "1.1-1", "reason": "the same fact"}),
        # 104.4c-1 reads game_drawn, which no resolution record lists.
        {**_scenario(["1.7"]), "id": "104.4c-1"},
    ]
    root = tmp_path / "rules"
    (root / "scenarios").mkdir(parents=True)
    (root / "scenarios" / "1.json").write_text(
        json.dumps({"unit": "1", "scenarios": scenarios}), encoding="utf-8",
    )
    return root


def _never_asked() -> frozenset[str]:
    raise AssertionError("the JVM must not be asked")


class TestTheCatalogueChecks:

    def test_a_ready_registered_rule_passes(self, rules_dir):
        assert directed_fork_error(
            ("1.1",), 0.25, rules_dir=rules_dir, registered=lambda: frozenset({"1.1"}),
        ) is None

    @pytest.mark.parametrize(("rule", "complaint"), [
        ("1.2", "declares its detector none"),
        ("1.6", "same_as"),
        ("1.7", "an event its record kind never lists"),
        ("1.3", "has no detector declaration"),
        ("1.4", "no contrasting variant classified into a fork operation"),
        ("9.9", "has no detector declaration"),
    ])
    def test_a_rule_the_catalogue_cannot_fork_is_named(self, rules_dir, rule, complaint):
        error = directed_fork_error(
            ("1.1", rule), 0.25, rules_dir=rules_dir, registered=_never_asked,
        )
        assert rule in error and complaint in error

    def test_the_checks_read_every_scenario_of_the_rule(self, rules_dir):
        assert rule_readiness(rules_dir)["1.5"] == ("declared", True)
        assert directed_fork_error(
            ("1.5",), 0.25, rules_dir=rules_dir, registered=lambda: frozenset({"1.5"}),
        ) is None

    def test_the_base_variant_makes_nothing(self, rules_dir):
        """A classified operation on the first variant does not count."""
        doc = json.loads((rules_dir / "scenarios" / "1.json").read_text(encoding="utf-8"))
        doc["scenarios"].append({
            "id": "2.1-1", "rules": ["2.1"], "record_kind": "resolution",
            "detector": _DETECTOR,
            "variants": [_variant("base", "pt"), _variant("other", {"unclassifiable": "x"})],
        })
        (rules_dir / "scenarios" / "1.json").write_text(json.dumps(doc), encoding="utf-8")
        assert rule_readiness(rules_dir)["2.1"] == ("declared", False)


class TestTheRegistryQuery:

    def test_a_rule_the_worker_does_not_register_is_named(self, rules_dir):
        error = directed_fork_error(
            ("1.1", "1.5"), 0.25, rules_dir=rules_dir, registered=lambda: frozenset({"1.1"}),
        )
        assert "1.5 has no board precondition registered in forge-connector" in error

    def test_the_jvm_is_asked_once(self, rules_dir):
        asked = []

        def registered():
            asked.append(1)
            return frozenset({"1.1", "1.5"})

        assert directed_fork_error(
            ("1.1", "1.5"), 0.25, rules_dir=rules_dir, registered=registered,
        ) is None
        assert asked == [1]

    def test_no_rule_asks_nothing(self, rules_dir):
        assert directed_fork_error((), 0.25, rules_dir=rules_dir, registered=_never_asked) is None

    def test_a_failed_query_is_a_refusal(self, rules_dir):
        def failing():
            raise RuntimeError("DirectedRulesMain exited 1")

        error = directed_fork_error(("1.1",), 0.25, rules_dir=rules_dir, registered=failing)
        assert "cannot list the registered rules" in error

    def test_the_default_query_is_looked_up_when_called(self, rules_dir, monkeypatch):
        """So a test can stub it under any caller."""
        monkeypatch.setattr(flags, "registered_directed_rules", lambda: frozenset({"1.1"}))
        assert directed_fork_error(("1.1",), 0.25, rules_dir=rules_dir) is None


class TestTheProbability:

    @pytest.mark.parametrize("probability", [0.0, -0.1, 1.5])
    def test_outside_the_interval_is_refused(self, rules_dir, probability):
        error = directed_fork_error((), probability, rules_dir=rules_dir, registered=_never_asked)
        assert "--directed-fork-probability must be in (0, 1]" in error

    @pytest.mark.parametrize("probability", [1e-6, 0.25, 1.0])
    def test_inside_it_is_accepted(self, rules_dir, probability):
        assert directed_fork_error(
            (), probability, rules_dir=rules_dir, registered=_never_asked,
        ) is None


class TestTheCatalogueIsReadAsEffectsReadsIt:
    """``sealed`` may not import ``effects``, so the check reads the scenarios
    itself; over the real catalogue it must agree with ``rules_catalogue_io``."""

    def test_every_rule_agrees(self):
        from effects.domain.rule_catalogue import DetectorDeclaration, NoDetector, SameAs
        from effects.domain.rule_catalogue import ForkOperation as Operation
        from effects.domain.rule_detectors import DetectorIndex
        from effects.infrastructure.rules_catalogue_io import load_scenarios

        catalogue = load_scenarios()
        untaggable = DetectorIndex.of(catalogue).untaggable
        assert set(untaggable) == UNTAGGABLE_SCENARIOS
        expected: dict[str, tuple[str, bool]] = {}
        rank = {"missing": 0, "none": 1, "untaggable": 2, "same_as": 3, "declared": 4}
        for rule, scenarios in catalogue.by_rule().items():
            state = max(
                (
                    "untaggable" if s.id in untaggable
                    else "declared" if isinstance(s.detector, DetectorDeclaration)
                    else "none" if isinstance(s.detector, NoDetector)
                    else "same_as" if isinstance(s.detector, SameAs) else "missing"
                    for s in scenarios
                ),
                key=rank.__getitem__,
            )
            forkable = any(
                isinstance(v.operation, Operation)
                for s in scenarios for v in s.variants if not v.is_base
            )
            expected[rule] = (state, forkable)
        assert rule_readiness(Path("rules")) == expected


class TestTheCollectors:

    def test_collect_variants_runs_the_check_before_anything(self, tmp_path, caplog, monkeypatch):
        from effects.application.collect_variants import CollectVariantsConfig, run
        from effects.domain.collection_caps import CollectionCaps

        monkeypatch.setattr(flags, "registered_directed_rules", _never_asked)
        config = CollectVariantsConfig(
            effect_records=tmp_path / "records",
            forge_cards_path=tmp_path / "no-forge",
            variant_scripts=tmp_path / "variants",
            caps=CollectionCaps(aimed_at=("no-such-rule",), directed_fork_rules=("no-such-rule",)),
        )
        with caplog.at_level(logging.ERROR):
            assert run(config) == 1
        assert "collect-variants: --directed-fork-rules: no-such-rule" in caplog.text
        assert not (tmp_path / "variants").exists()

    def test_collect_variants_refuses_a_probability_outside_the_interval(self, tmp_path, caplog):
        from effects.application.collect_variants import CollectVariantsConfig, run
        from effects.domain.collection_caps import CollectionCaps

        config = CollectVariantsConfig(
            effect_records=tmp_path / "records", forge_cards_path=tmp_path / "no-forge",
            caps=CollectionCaps(aimed_at=(), directed_fork_probability=0.0),
        )
        with caplog.at_level(logging.ERROR):
            assert run(config) == 1
        assert "--directed-fork-probability" in caplog.text

@pytest.mark.integration
def test_the_worker_lists_its_registered_rules():
    """One real run of DirectedRulesMain: it exits 0 and prints rule ids only,
    704.5f's among them."""
    from price_predictor.infrastructure.forge_jvm import resolve_connector_jar

    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("the connector JAR is not built")
    registered = flags.registered_directed_rules()
    assert all(rule and " " not in rule for rule in registered)
    assert "704.5f" in registered


_DIRECTED = Path("forge-connector/src/main/java/com/pricepredictor/connector/effects/directed")

def _registered_classes(registry: str) -> list[str]:
    """The rule classes ``DirectedRules.REGISTERED`` lists in ``registry``, the
    registry's source: each ``X.rule()`` of the list, comments left out."""
    import re

    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", registry, flags=re.S)
    body = code[code.index("REGISTERED = of("):]
    body = body[:body.index(");")]
    return re.findall(r"(\w+)\.rule\(\)", body)


def _registered_in_source() -> frozenset[str]:
    """The rule ids ``DirectedRules.REGISTERED`` lists, read off the connector's
    source: each listed class's ``RULE`` constant. The integration test below
    asks the JVM itself."""
    import re

    registry = (_DIRECTED / "DirectedRules.java").read_text(encoding="utf-8")
    rules = set()
    for name in _registered_classes(registry):
        source = (_DIRECTED / "rules" / f"{name}.java").read_text(encoding="utf-8")
        rules.add(re.search(r'public static final String RULE = "([^"]+)";', source).group(1))
    return frozenset(rules)


def test_a_commented_out_line_is_no_registration():
    registry = """
    static final DirectedRules REGISTERED = of(
            StunCounter.rule(),
            // VanishingInstances.rule(),
            /* Dethrone.rule(), */
            ToughnessZero.rule()); // Aftermath.rule()
    """
    assert _registered_classes(registry) == ["StunCounter", "ToughnessZero"]


def test_the_startup_check_passes_over_the_whole_list():
    """Every rule of the real ``rules/directed-fork-rules.txt``, checked
    against the real catalogue and the connector's registrations, all at
    once: no listed rule is refused, and the failure names every rule the
    check would refuse, not the first."""
    listed = Path("rules/directed-fork-rules.txt").read_text(encoding="utf-8").split()
    registered = _registered_in_source()
    refused = {}
    for rule in listed:
        error = directed_fork_error((rule,), 0.25, registered=lambda: registered)
        if error is not None:
            refused[rule] = error
    assert refused == {}
    assert len(listed) == len(set(listed)) == 71
