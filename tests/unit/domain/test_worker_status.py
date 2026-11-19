"""The worker's status line, read back (findings 11.1, 16.4): one JSON object
per line, never split, so a refusal reason, a gap or a rule id holding a comma
reads back whole; each lifetime's latest line stands, and lifetimes and slots
add up."""

from __future__ import annotations

import json
import re

from price_predictor.domain.worker_failures import classify
from price_predictor.domain.worker_status import (
    STATUS_PREFIX,
    EngineCheck,
    SlotStatuses,
    directed_fork_tally_document,
    engine_document,
    fork_refusal_document,
    hook_failure_document,
    judge_engine,
    parse_status_line,
    render_fork_refusals,
    render_hook_failures,
)
from tests.unit.forge_sources import connector_source
from tests.unit.worker_status_line import (
    PINNED_PREFERENCES,
    directed_counts,
    status_document,
    status_line,
)

#: What a printed Java map or a split on ", " would cut: a comma, quotes, an
#: equals sign, braces and a card name past ASCII.
AWKWARD = 'replace_card remembered, "imprinted" or exiled={cards} of Lim-Dûl\'s Vault'


def _slot(*lines: str) -> SlotStatuses:
    slot = SlotStatuses()
    for line in lines:
        slot.add_line(line)
    return slot


def test_the_prefix_and_the_members_are_the_ones_the_connector_prints():
    source = connector_source("WorkerStatus.java")
    assert f'static final String LINE = "{STATUS_PREFIX}";' in source
    written = re.findall(r'\.(?:value|raw)\("([a-z_]+)"', source)
    assert set(status_document("a")) | set(PINNED_PREFERENCES) | {
        "mode", "gaps", "connector", "forge"} == set(written)


def test_a_reason_and_a_gap_holding_a_comma_quotes_and_non_ascii_round_trip():
    line = status_line(
        "a", hooks={"mode": "degraded", "gaps": [AWKWARD]},
        fork_refusals={f"directed {AWKWARD}": 2, "probe no combat": 1},
    )
    assert line.isascii(), "the Java side escapes every character past ASCII"

    status = parse_status_line(line)

    assert status is not None
    assert (status.mode, status.gaps) == ("degraded", (AWKWARD,))
    assert status.fork_refusals == {f"directed {AWKWARD}": 2, "probe no combat": 1}
    assert fork_refusal_document([_slot(line)]) == {
        f"directed {AWKWARD}": 2, "probe no combat": 1}


def test_the_line_carries_the_pinned_preferences_and_the_lifetime():
    status = parse_status_line(status_line("lx3k9"))

    assert status is not None
    assert status.lifetime == "lx3k9"
    assert status.pinned_preferences == PINNED_PREFERENCES
    assert status.gaps == ()


def test_the_line_is_found_after_another_stream_s_text_and_read_up_to_its_object():
    line = "Worker: 10 matches gener" + status_line(
        "a", fork_refusals={"directed dies": 1}) + "ated"

    status = parse_status_line(line)

    assert status is not None and status.fork_refusals == {"directed dies": 1}


def test_a_cut_or_malformed_line_is_skipped():
    whole = status_line("a", fork_refusals={"directed dies": 1})
    for line in (
        whole[: len(whole) // 2],
        STATUS_PREFIX + "[1]",
        STATUS_PREFIX + json.dumps({"lifetime": "a"}),
        status_line("a", fork_refusals={"directed dies": "many"}),
        status_line("a", fork_refusals={"directed dies": True}),
        status_line("a", directed={"704.5f": [1, 2]}),
        "Effect records: x/run.0-a.jsonl.gz [mode=patched]",
    ):
        assert parse_status_line(line) is None, line


def test_a_status_line_is_no_failure_line():
    assert classify(status_line("a", hook_failures={
        "connector": {"RESOLVE": 3}, "forge": {"resolve": 3}})) is None


def test_a_lifetime_s_latest_line_stands_and_lifetimes_add_up():
    slot = _slot(
        status_line("a"),
        status_line("a", hook_failures={"connector": {"RESOLVE": 1}, "forge": {"resolve": 1}}),
        status_line("a", hook_failures={
            "connector": {"RESOLVE": 3, "game-end marker": 1}, "forge": {"resolve": 3}}),
        status_line("b"),
        status_line("b", hook_failures={"connector": {"RESOLVE": 2}, "forge": {}}),
    )

    assert hook_failure_document([slot]) == {
        "connector": {"RESOLVE": 5, "game-end marker": 1},
        "forge": {"resolve": 3},
    }


def test_hook_failures_add_up_over_slots_and_render_on_one_line():
    first = _slot(status_line("a", hook_failures={"connector": {"RESOLVE": 1}, "forge": {}}))
    second = _slot(status_line("b", hook_failures={
        "connector": {"RESOLVE": 2, "subscriber X.y": 1}, "forge": {"s": 4}}))
    document = hook_failure_document([first, second])

    assert document == {"connector": {"RESOLVE": 3, "subscriber X.y": 1}, "forge": {"s": 4}}
    assert render_hook_failures(document) == (
        "3 RESOLVE, 1 subscriber X.y; contained by Forge: 4 s")
    assert render_hook_failures(hook_failure_document([])) == "no hook failures"


def test_fork_refusals_add_up_in_name_order_and_render_on_one_line():
    first = _slot(status_line("a", fork_refusals={"probe no combat": 2}))
    second = _slot(status_line("b", fork_refusals={"directed dies": 1, "probe no combat": 1}))
    document = fork_refusal_document([first, second])

    assert list(document.items()) == [("directed dies", 1), ("probe no combat", 3)]
    assert render_fork_refusals(document) == "1 directed dies, 3 probe no combat"
    assert render_fork_refusals(fork_refusal_document([])) == "no fork refusals"


def test_the_directed_tally_adds_up_over_lifetimes_and_slots_in_rule_name_order():
    first = _slot(
        status_line("a", directed={"704.5f": directed_counts(matched=3, taken=1)}),
        status_line("a", directed={"704.5f": directed_counts(matched=5, taken=2)}),
        status_line("b", directed={"704.5f": directed_counts(matched=4),
                                   "a, b=c": directed_counts(drawn=1)}),
    )
    second = _slot(status_line("c", directed={"118.12b": directed_counts(matched=2)}))

    document = directed_fork_tally_document([first, second])

    assert list(document) == ["118.12b", "704.5f", "a, b=c"]
    assert document["704.5f"] == directed_counts(matched=9, taken=2)
    assert document["a, b=c"] == directed_counts(drawn=1)
    assert directed_fork_tally_document([]) == {}


# ── The engine a run's workers link (finding 11.1, task K01c) ───────────


def _status(lifetime: str = "a", **members):
    return parse_status_line(status_line(lifetime, **members))


def _hooks(mode: str, *gaps: str) -> dict:
    return {"hooks": {"mode": mode, "gaps": list(gaps)}}


def test_a_whole_engine_is_never_refused():
    verdict = judge_engine(_status(), [_slot(status_line("a"))])

    assert verdict.shortfall == ()
    assert verdict.refusal(allow_degraded=False) is None
    assert verdict.warning() is None


def test_no_status_line_yet_is_no_verdict():
    verdict = judge_engine(None, [])
    assert verdict.refusal(allow_degraded=False) is None


def test_a_degraded_engine_is_refused_naming_the_mode_and_each_gap():
    first = _status(**_hooks("degraded", "ApiEvents missing", AWKWARD))

    refusal = judge_engine(first, []).refusal(allow_degraded=False)

    assert "mode degraded (no effect-record hooks); ApiEvents missing; " + AWKWARD in refusal
    assert "worker lifetime a" in refusal
    assert "--allow-degraded" in refusal


def test_a_patched_engine_with_a_gap_is_refused_naming_the_gap_alone():
    refusal = judge_engine(_status(**_hooks("patched", "Listener.onX missing")), []).refusal(
        allow_degraded=False)

    assert "): Listener.onX missing." in refusal
    assert "mode" not in refusal.partition("): ")[2].partition(".")[0]


def test_allowed_a_shortfall_is_a_warning_and_no_refusal():
    verdict = judge_engine(_status(**_hooks("degraded")), [])

    assert verdict.refusal(allow_degraded=True) is None
    assert verdict.warning() == (
        "--allow-degraded: collecting on an engine that is not whole (worker "
        "lifetime a): mode degraded (no effect-record hooks)")


def test_a_later_lifetime_on_another_engine_is_refused_even_when_allowed():
    first = _status("a")
    slots = [_slot(status_line("a"), status_line(
        "b", pinned_preferences={**PINNED_PREFERENCES, "ui_language": "de-DE"}))]

    verdict = judge_engine(first, slots)

    assert set(verdict.changed) == {"b"}
    refusal = verdict.refusal(allow_degraded=True)
    assert refusal.startswith("worker lifetime b runs another engine than the run's first "
                              "(lifetime a): pinned_preferences ")
    assert "'de-DE'" in refusal


def test_the_engine_document_is_the_three_members_run_json_keeps():
    assert engine_document(_status(**_hooks("patched", "g"))) == {
        "mode": "patched", "gaps": ["g"], "pinned_preferences": PINNED_PREFERENCES}


def test_the_check_keeps_its_first_refusal_and_warns_once():
    degraded = judge_engine(_status(**_hooks("degraded")), [])
    allowed = EngineCheck(allow_degraded=True)

    assert allowed.review(degraded) is not None
    assert allowed.review(degraded) is None, "once a run"
    assert allowed.refusal is None

    refusing = EngineCheck()
    assert refusing.review(degraded) is None, "no warning without the flag"
    first_reason = refusing.refusal
    refusing.review(judge_engine(_status(), []))
    assert refusing.refusal == first_reason is not None
