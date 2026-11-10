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
        from effects.domain.rule_catalogue import DetectorDeclaration, NoDetector
        from effects.domain.rule_catalogue import ForkOperation as Operation
        from effects.infrastructure.rules_catalogue_io import load_scenarios

        expected: dict[str, tuple[str, bool]] = {}
        rank = {"missing": 0, "none": 1, "declared": 2}
        for rule, scenarios in load_scenarios().by_rule().items():
            state = max(
                (
                    "declared" if isinstance(s.detector, DetectorDeclaration)
                    else "none" if isinstance(s.detector, NoDetector) else "missing"
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

    def test_collect_coverage_runs_no_check(self, monkeypatch):
        """Its --directed-fork-rules is a label: a rule nothing knows still runs,
        no fork is enabled, and the JVM is never asked."""
        from unittest.mock import patch

        from effects.application import collect_coverage
        from effects.infrastructure.cli import build_parser, run_collect_coverage

        monkeypatch.setattr(flags, "registered_directed_rules", _never_asked)
        with (
            patch("effects.application.collect_coverage.run", return_value=0) as collect,
            patch("price_predictor.infrastructure.forge_jvm.check_worker_prerequisites"),
        ):
            assert run_collect_coverage(build_parser().parse_args(
                ["collect-coverage", "--directed-fork-rules", "no-such-rule",
                 "--exclude-cards", "h.txt"],
            )) == 0
        assert collect.call_args.args[0].caps.directed_fork_rules == ()
        assert "directed_fork_error" not in Path(collect_coverage.__file__).read_text(
            encoding="utf-8",
        )


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
