"""--agent-mix parsing + per-seat categorical sampling (FR-006)."""

from __future__ import annotations

import random

import pytest

from draft.application.agent_mix import (
    AgentMixError,
    format_agent_mix,
    parse_agent_mix,
    sample_agents,
)


def test_parse_default_spec() -> None:
    mix = parse_agent_mix("forge-full:6,forge-r30:1,forge-r100:1")
    assert mix == [("forge-full", 6), ("forge-r30", 1), ("forge-r100", 1)]


def test_format_round_trip() -> None:
    spec = "forge-full:6,forge-r30:1,forge-r100:1"
    assert format_agent_mix(parse_agent_mix(spec)) == spec


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "forge-full", "forge-full:0", "forge-full:-1",
     "forge-full:x", ":3", "forge-full:1:2"],
)
def test_malformed_specs_raise(bad: str) -> None:
    with pytest.raises(AgentMixError):
        parse_agent_mix(bad)


def test_sample_returns_pod_size_labels_verbatim() -> None:
    mix = parse_agent_mix("forge-full:6,forge-r30:1,forge-r100:1")
    rng = random.Random(0)
    agents = sample_agents(mix, pod_size=8, rng=rng)
    assert len(agents) == 8
    assert set(agents) <= {"forge-full", "forge-r30", "forge-r100"}


def test_single_label_mix_assigns_all_seats() -> None:
    mix = parse_agent_mix("forge-full:1")
    agents = sample_agents(mix, pod_size=8, rng=random.Random(1))
    assert agents == ["forge-full"] * 8


def test_weights_are_honored_in_distribution() -> None:
    # forge-full weight 9 vs forge-r100 weight 1 -> ~90% forge-full over many draws.
    mix = parse_agent_mix("forge-full:9,forge-r100:1")
    rng = random.Random(42)
    draws = sample_agents(mix, pod_size=5000, rng=rng)
    share_full = draws.count("forge-full") / len(draws)
    assert 0.85 < share_full < 0.95


def test_draws_are_independent_per_seat() -> None:
    # With a balanced 1:1 mix, an 8-seat pod should not be forced to all-same.
    mix = parse_agent_mix("a:1,b:1")
    rng = random.Random(7)
    # Across many pods, mixed pods dominate (P(all same) = 2 * 0.5^8 per pod).
    mixed = sum(
        len(set(sample_agents(mix, 8, rng))) > 1 for _ in range(200)
    )
    assert mixed > 190
