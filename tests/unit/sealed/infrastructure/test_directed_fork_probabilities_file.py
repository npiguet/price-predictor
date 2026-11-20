"""``match-outcomes --directed-fork-probabilities FILE`` (DF01, DR1): each listed
rule's one registered side and the records it has there, and the count a side
needs, from which the supervisor computes the rule's probability at start and
again at every log scan."""

from __future__ import annotations

import argparse
import json

import pytest

from sealed.domain.directed_fork_needs import DirectedForkNeeds, DirectedRuleNeed
from sealed.infrastructure.cli import _effect_collection_caps, build_parser
from sealed.infrastructure.directed_fork_probabilities_file import read_directed_fork_needs

RULES = ("704.5f", "118.12b")


def _write(tmp_path, text):
    path = tmp_path / "probabilities.json"
    path.write_text(text, encoding="utf-8")
    return path


def _document(rules, count=100):
    return json.dumps({"count_per_side": count, "rules": rules})


def test_a_file_of_listed_rules_is_read(tmp_path):
    path = _write(tmp_path, _document({"704.5f": {"does_not_fire": 6814},
                                       "118.12b": {"fires": 0}}))
    assert read_directed_fork_needs(path, RULES) == DirectedForkNeeds(100, {
        "704.5f": DirectedRuleNeed("does_not_fire", 6814),
        "118.12b": DirectedRuleNeed("fires", 0),
    })


def test_a_file_in_the_old_format_is_refused_with_a_message(tmp_path):
    """The flat map from rule to probability carried no counts to recompute from."""
    path = _write(tmp_path, json.dumps({"704.5f": 1.0, "118.12b": 0.25}))
    with pytest.raises(ValueError, match="old format.*--probabilities-output"):
        read_directed_fork_needs(path, RULES)


def test_a_rule_whose_registered_variants_span_two_sides_is_refused_naming_it(tmp_path):
    """The workers' tally counts a rule's kept forks, not each side's."""
    path = _write(tmp_path, _document({"704.5f": {"fires": 3, "does_not_fire": 5}}))
    with pytest.raises(ValueError, match="704.5f.*two sides"):
        read_directed_fork_needs(path, RULES)


@pytest.mark.parametrize("text", [
    "{", "[1, 2]",
    _document({"704.5f": {"fires": "many"}}), _document({"704.5f": {"fires": True}}),
    _document({"704.5f": {"fires": -1}}), _document({"704.5f": {"other": 3}}),
    _document({"704.5f": {}}), _document({"704.5f": 0.5}), _document([]),
    _document({"704.5f": {"fires": 0}}, count=0), _document({"704.5f": {"fires": 0}}, count=1.5),
])
def test_a_malformed_file_is_refused(tmp_path, text):
    with pytest.raises(ValueError, match="probabilities.json"):
        read_directed_fork_needs(_write(tmp_path, text), RULES)


def test_a_rule_not_in_directed_fork_rules_is_refused_naming_it(tmp_path):
    with pytest.raises(ValueError, match="999.9z"):
        read_directed_fork_needs(_write(tmp_path, _document({"999.9z": {"fires": 0}})), RULES)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nope.json"):
        read_directed_fork_needs(tmp_path / "nope.json", RULES)


def test_the_flag_is_on_match_outcomes_and_its_probabilities_become_properties():
    args = build_parser().parse_args(
        ["match-outcomes", "--directed-fork-probabilities", "p.json"])
    assert str(args.directed_fork_probabilities) == "p.json"
    args = argparse.Namespace(directed_rule_probabilities={"704.5f": 0.5})
    properties = _effect_collection_caps(args)
    assert properties["effect.directed.probability.704.5f"] == "0.5"


def test_a_bad_file_exits_1_before_any_supervisor(tmp_path, monkeypatch, capsys):
    import sealed.application.match_outcomes as module
    from sealed.infrastructure.cli import run_match_outcomes

    def no_supervisor(*args, **kwargs):
        raise AssertionError("a refused run must not start a supervisor")

    monkeypatch.setattr(module, "MatchOutcomeSupervisor", no_supervisor)
    path = _write(tmp_path, _document({"999.9z": {"fires": 0}}))
    args = build_parser().parse_args(
        ["match-outcomes", "--outcomes-dir", "out", "--directed-fork-probabilities", str(path)])
    assert run_match_outcomes(args) == 1
    assert "999.9z" in capsys.readouterr().err


def test_a_two_sided_rule_exits_1_at_start(tmp_path, monkeypatch, capsys):
    from sealed.infrastructure.cli import run_match_outcomes

    path = _write(tmp_path, _document({"704.5f": {"fires": 3, "does_not_fire": 5}}))
    args = build_parser().parse_args(
        ["match-outcomes", "--outcomes-dir", "out", "--directed-fork-rules", "704.5f",
         "--directed-fork-probabilities", str(path)])
    assert run_match_outcomes(args) == 1
    assert "two sides" in capsys.readouterr().err


def test_the_supervisor_starts_from_the_file_and_is_steered_by_it(tmp_path, monkeypatch):
    """The starting probabilities ride the caps; the counts reach the supervisor,
    which recomputes them at every log scan."""
    from unittest.mock import MagicMock

    import sealed.application.match_outcomes as module
    import sealed.infrastructure.cli as cli
    from price_predictor.infrastructure import forge_classpath

    built = {}

    def supervisor(**kwargs):
        built.update(kwargs)
        built["supervisor"] = MagicMock(collected_nothing=False)
        return built["supervisor"]

    monkeypatch.setattr(module, "MatchOutcomeSupervisor", supervisor)
    monkeypatch.setattr(forge_classpath, "check_worker_prerequisites", lambda **_: None)
    monkeypatch.setattr(cli, "directed_fork_error", lambda *_: None)
    monkeypatch.setattr(cli, "check_conversion", lambda *_: None)
    path = _write(tmp_path, _document({"704.5f": {"does_not_fire": 40},
                                       "118.12b": {"fires": 0}}))
    args = build_parser().parse_args([
        "match-outcomes", "--effect-records", str(tmp_path / "records"), "--workers", "1",
        "--cards-folder", str(tmp_path / "converted" / "cardsfolder"),
        "--directed-fork-rules", "704.5f,118.12b", "--directed-fork-probabilities", str(path)])

    assert cli.run_match_outcomes(args) == 0
    caps = built["options"].collection_caps
    assert caps["effect.directed.probability.704.5f"] == "0.6"
    assert caps["effect.directed.probability.118.12b"] == "1.0"
    built["supervisor"].steer_directed_forks.assert_called_once_with(
        read_directed_fork_needs(path, RULES))
