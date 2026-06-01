"""Parse and sample the ``--agent-mix`` categorical (FR-006).

``--agent-mix`` is a comma-separated ``label:weight`` spec, e.g.
``forge-full:6,forge-r30:1,forge-r100:1``. Each pod seat draws its agent
independently from this categorical (weights honored, identifier recorded
verbatim). The supervisor validates and normalizes the spec here, then forwards
it to the Java worker, which performs the actual per-seat sampling + override
application; this module is the single owner of the spec's grammar and the
sampling contract (also used by tests and any Python-side draw).
"""

from __future__ import annotations

import random


class AgentMixError(ValueError):
    """Raised when an ``--agent-mix`` spec is malformed."""


def parse_agent_mix(spec: str) -> list[tuple[str, int]]:
    """Parse ``label:weight,label:weight,…`` into ``[(label, weight), …]``.

    Weights must be positive integers; labels must be non-empty and free of the
    ``:`` / ``,`` separators. Order is preserved.
    """
    if not spec or not spec.strip():
        raise AgentMixError("agent mix must be a non-empty 'label:weight,...' spec")
    out: list[tuple[str, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.count(":") != 1:
            raise AgentMixError(
                f"agent-mix entry {token!r} must be exactly 'label:weight'"
            )
        label, _, weight_str = token.partition(":")
        label = label.strip()
        if not label:
            raise AgentMixError(f"agent-mix entry {token!r} has an empty label")
        try:
            weight = int(weight_str)
        except ValueError:
            raise AgentMixError(
                f"agent-mix weight for {label!r} must be an integer, got {weight_str!r}"
            ) from None
        if weight < 1:
            raise AgentMixError(
                f"agent-mix weight for {label!r} must be >= 1, got {weight}"
            )
        out.append((label, weight))
    if not out:
        raise AgentMixError("agent mix must contain at least one 'label:weight' entry")
    return out


def format_agent_mix(mix: list[tuple[str, int]]) -> str:
    """Render the canonical ``label:weight,…`` form (for forwarding to the worker)."""
    return ",".join(f"{label}:{weight}" for label, weight in mix)


def sample_agents(
    mix: list[tuple[str, int]], pod_size: int, rng: random.Random,
) -> list[str]:
    """Draw ``pod_size`` agent labels, each an independent weighted categorical draw."""
    labels = [label for label, _ in mix]
    weights = [weight for _, weight in mix]
    return [rng.choices(labels, weights=weights, k=1)[0] for _ in range(pod_size)]
