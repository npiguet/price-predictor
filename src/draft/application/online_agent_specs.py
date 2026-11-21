"""Parsing of the gen-3 online trainer's agent flags: ``--learner``, ``--frozen``
and ``--agent-temp``.
"""

from __future__ import annotations

from pathlib import Path

from draft.application.online_config_validation import OnlineConfigError


def parse_learner_spec(specs: list[str] | None) -> tuple[str, Path]:
    """Parse the single ``--learner LABEL=PATH``.

    A bare ``PATH`` is deliberately **not** accepted (unlike
    ``--agent-checkpoint``): the learner's mix label is load-bearing — it names
    the seats that feed the gradient and the anchor margin's learner side.
    """
    specs = list(specs or [])
    if not specs:
        raise OnlineConfigError(
            "--learner LABEL=PATH is required: exactly one learner agent names "
            "the mix label piloted by the live policy and the checkpoint that "
            "warm-starts it"
        )
    if len(specs) > 1:
        raise OnlineConfigError(
            f"exactly one --learner is allowed, got {len(specs)}: "
            f"{', '.join(specs)}"
        )
    label, sep, path_str = specs[0].partition("=")
    label, path_str = label.strip(), path_str.strip()
    if not sep or not label or not path_str:
        raise OnlineConfigError(
            f"--learner {specs[0]!r} must be 'LABEL=PATH' — the mix label is "
            "load-bearing, so a bare PATH is not accepted"
        )
    return label, Path(path_str)


def parse_frozen_specs(specs: list[str] | None) -> dict[str, Path]:
    """Parse the repeatable ``--frozen LABEL=PATH`` bindings."""
    frozen: dict[str, Path] = {}
    for spec in specs or []:
        label, sep, path_str = spec.partition("=")
        label, path_str = label.strip(), path_str.strip()
        if not sep or not label or not path_str:
            raise OnlineConfigError(f"--frozen {spec!r} must be 'LABEL=PATH'")
        if label in frozen:
            raise OnlineConfigError(f"duplicate --frozen label {label!r}")
        frozen[label] = Path(path_str)
    return frozen


def parse_agent_temperatures(spec: str | None) -> dict[str, float]:
    """Parse ``--agent-temp "LABEL=T,…"`` into a per-label sampling temperature.

    Same ``LABEL=VALUE`` dialect as ``--learner`` / ``--frozen`` /
    ``--agent-checkpoint``, comma-separated. A label the map omits plays argmax,
    so the map is the *whole* statement of how the field plays — which is why an
    empty or malformed entry is an error rather than a silently-skipped one.
    """
    if spec is None or not spec.strip():
        raise OnlineConfigError(
            "--agent-temp \"LABEL=T,…\" is required (no default): it names every "
            "label that samples, and the learner's entry is also the temperature "
            "the update's policy distributions are evaluated at"
        )
    temperatures: dict[str, float] = {}
    for entry in spec.split(","):
        label, sep, value_str = entry.partition("=")
        label, value_str = label.strip(), value_str.strip()
        if not sep or not label or not value_str:
            raise OnlineConfigError(
                f"--agent-temp entry {entry.strip()!r} must be 'LABEL=VALUE' "
                f"(the whole map was {spec!r})"
            )
        try:
            value = float(value_str)
        except ValueError:
            raise OnlineConfigError(
                f"--agent-temp {label!r} temperature must be a number, got "
                f"{value_str!r}"
            ) from None
        if label in temperatures:
            raise OnlineConfigError(f"duplicate --agent-temp label {label!r}")
        temperatures[label] = value
    return temperatures
