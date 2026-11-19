"""``match-outcomes --directed-fork-probabilities FILE`` (DF01): a JSON map from
rule id to the probability its directed forks are taken with."""

from __future__ import annotations

import argparse
import json

import pytest

from sealed.infrastructure.cli import _effect_collection_caps, build_parser
from sealed.infrastructure.directed_fork_probabilities_file import (
    read_directed_fork_probabilities,
)

RULES = ("704.5f", "118.12b")


def _write(tmp_path, text):
    path = tmp_path / "probabilities.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_map_of_listed_rules_is_read(tmp_path):
    path = _write(tmp_path, json.dumps({"704.5f": 1, "118.12b": 0.25}))
    assert read_directed_fork_probabilities(path, RULES) == {"704.5f": 1.0, "118.12b": 0.25}


@pytest.mark.parametrize("text", ["{", "[1, 2]", '{"704.5f": "often"}', '{"704.5f": true}',
                                  '{"704.5f": null}'])
def test_a_malformed_file_is_refused(tmp_path, text):
    with pytest.raises(ValueError, match="probabilities.json"):
        read_directed_fork_probabilities(_write(tmp_path, text), RULES)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "NaN"])
def test_a_value_outside_0_to_1_is_refused_naming_the_rule(tmp_path, value):
    with pytest.raises(ValueError, match="704.5f"):
        read_directed_fork_probabilities(_write(tmp_path, '{"704.5f": %s}' % value), RULES)


def test_a_rule_not_in_directed_fork_rules_is_refused_naming_it(tmp_path):
    with pytest.raises(ValueError, match="999.9z"):
        read_directed_fork_probabilities(_write(tmp_path, '{"999.9z": 0.5}'), RULES)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nope.json"):
        read_directed_fork_probabilities(tmp_path / "nope.json", RULES)


def test_the_flag_is_on_match_outcomes_and_its_probabilities_become_properties():
    args = build_parser().parse_args(
        ["match-outcomes", "--directed-fork-probabilities", "p.json"])
    assert str(args.directed_fork_probabilities) == "p.json"
    args = argparse.Namespace(directed_rule_probabilities={"704.5f": 0.5})
    properties = _effect_collection_caps(args)
    assert properties["effect.directed.probability.704.5f"] == "0.5"


def test_a_bad_file_exits_2_before_any_supervisor(tmp_path, monkeypatch, capsys):
    import sealed.application.match_outcomes as module
    from sealed.infrastructure.cli import run_match_outcomes

    def no_supervisor(*args, **kwargs):
        raise AssertionError("a refused run must not start a supervisor")

    monkeypatch.setattr(module, "MatchOutcomeSupervisor", no_supervisor)
    path = _write(tmp_path, '{"999.9z": 0.5}')
    args = build_parser().parse_args(
        ["match-outcomes", "--directed-fork-probabilities", str(path)])
    assert run_match_outcomes(args) == 1
    assert "999.9z" in capsys.readouterr().err
