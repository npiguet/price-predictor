"""Parsing of the gen-3 online trainer's agent flags: ``--learner``, ``--frozen``
and ``--agent-temp``.
"""

from __future__ import annotations

from pathlib import Path

from draft.application.online_config_validation import OnlineConfigError


def _split_binding(text: str) -> tuple[str, str] | None:
    """``LABEL=VALUE`` as its stripped label and value, or ``None`` when malformed."""
    label, sep, value = text.partition("=")
    label, value = label.strip(), value.strip()
    if _malformed(sep, label, value):
        return None
    return label, value


def _malformed(sep: str, label: str, value: str) -> bool:
    return not sep or not label or not value


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
    binding = _split_binding(specs[0])
    if binding is None:
        raise OnlineConfigError(
            f"--learner {specs[0]!r} must be 'LABEL=PATH' — the mix label is "
            "load-bearing, so a bare PATH is not accepted"
        )
    label, path_str = binding
    return label, Path(path_str)


def parse_frozen_specs(specs: list[str] | None) -> dict[str, Path]:
    """Parse the repeatable ``--frozen LABEL=PATH`` bindings."""
    frozen: dict[str, Path] = {}
    for spec in specs or []:
        binding = _split_binding(spec)
        if binding is None:
            raise OnlineConfigError(f"--frozen {spec!r} must be 'LABEL=PATH'")
        label, path_str = binding
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
        label, value = _parse_temperature_entry(entry, spec)
        if label in temperatures:
            raise OnlineConfigError(f"duplicate --agent-temp label {label!r}")
        temperatures[label] = value
    return temperatures


def _parse_temperature_entry(entry: str, spec: str) -> tuple[str, float]:
    """One ``LABEL=T`` entry of ``--agent-temp`` (``spec`` is the whole map, for the message)."""
    binding = _split_binding(entry)
    if binding is None:
        raise OnlineConfigError(
            f"--agent-temp entry {entry.strip()!r} must be 'LABEL=VALUE' "
            f"(the whole map was {spec!r})"
        )
    label, value_str = binding
    try:
        value = float(value_str)
    except ValueError:
        raise OnlineConfigError(
            f"--agent-temp {label!r} temperature must be a number, got "
            f"{value_str!r}"
        ) from None
    return label, value
