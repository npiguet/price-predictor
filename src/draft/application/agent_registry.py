"""Registry of model-piloted mix labels → their pick services (data-model §2.6).

Parses the ``--agent-checkpoint LABEL=PATH`` bindings, validates every mix label
(a Forge built-in or a bound checkpoint — FR-011) and each checkpoint's geometry
against the live draft (FR-012), and builds one :class:`AgentPickService` per
bound label. Exposes the set of external (model) labels the worker routes
through the pick side-channel. Built once at supervisor startup.

Geometry validation mirrors the fail-fast width checks in
``imitation_model_setup._check_dims`` (research D5): ``config.packs`` must equal the
live ``PACKS`` and ``config.P`` must be ≥ the live pack size when it is known
(``--set`` runs); for random-set runs the live pack size is unknown at startup
and the per-request ``pick_number ≤ P`` guard in :class:`AgentPickService`
catches any overflow as a pick fault instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from draft.application.agent_pick_service import AgentPickService, PickFault
from draft.application.generate_draft_data import PACKS

# Forge AI built-in agent labels (no checkpoint binding required).
FORGE_BUILTINS = frozenset({"forge-full", "forge-r30", "forge-r100"})


class AgentRegistryError(ValueError):
    """Raised on an invalid label binding or geometry mismatch (FR-011/FR-012)."""


def _split_spec(spec: str) -> tuple[str, str]:
    """One ``--agent-checkpoint`` spec as ``(label, path text)``, both stripped."""
    label, sep, path_str = spec.partition("=")
    if sep:
        return label.strip(), path_str.strip()
    return "draft-agent", spec.strip()


def parse_agent_checkpoints(specs: list[str]) -> dict[str, Path]:
    """Parse repeatable ``LABEL=PATH`` (bare ``PATH`` ⇒ label ``draft-agent``).

    Raises :class:`AgentRegistryError` on a malformed spec or a duplicate label.
    """
    bindings: dict[str, Path] = {}
    for spec in specs:
        label, path_str = _split_spec(spec)
        if not label or not path_str:
            raise AgentRegistryError(
                f"--agent-checkpoint {spec!r} must be 'LABEL=PATH' or a bare PATH"
            )
        if label in bindings:
            raise AgentRegistryError(f"duplicate --agent-checkpoint label {label!r}")
        bindings[label] = Path(path_str)
    return bindings


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """How :meth:`AgentRegistry.build` constructs and checks the services.

    ``locator`` and ``device`` reach every service built from a checkpoint;
    ``pick_mode``, ``temperature`` and ``seed`` set how those services pick.
    ``pack_size`` (when known, e.g. ``--set`` runs) checks each checkpoint's
    ``P`` (FR-012). ``preloaded`` supplies already-constructed services (the
    online trainer's live-model learner), which are never rebuilt.
    """

    locator: Any
    pick_mode: str = "argmax"
    temperature: float = 1.0
    seed: int | None = None
    pack_size: int | None = None
    device: Any = None
    preloaded: dict[str, AgentPickService] | None = None


def _bound_labels(agent_checkpoints: dict[str, Path], preloaded: dict) -> set[str]:
    """Every bound label; a label bound both ways is refused."""
    clash = sorted(set(preloaded) & set(agent_checkpoints))
    if clash:
        raise AgentRegistryError(
            f"label(s) {', '.join(clash)} bound both to a checkpoint path "
            "and to a preloaded service"
        )
    return set(agent_checkpoints) | set(preloaded)


def _check_mix_labels(mix_labels: set[str], bound: set[str]) -> None:
    for label in mix_labels:
        if label not in FORGE_BUILTINS and label not in bound:
            raise AgentRegistryError(
                f"mix label {label!r} is neither a Forge built-in "
                f"({', '.join(sorted(FORGE_BUILTINS))}) nor bound via "
                "--agent-checkpoint"
            )


def _load_services(
    agent_checkpoints: dict[str, Path], options: BuildOptions,
) -> dict[str, AgentPickService]:
    services: dict[str, AgentPickService] = {}
    for label, path in agent_checkpoints.items():
        if not path.exists():
            raise AgentRegistryError(f"checkpoint for label {label!r} not found: {path}")
        services[label] = AgentPickService(
            path, options.locator, device=options.device,
            pick_mode=options.pick_mode, temperature=options.temperature,
            seed=options.seed,
        )
    return services


def _check_geometry(label: str, service: AgentPickService, pack_size: int | None) -> None:
    config = service.config
    if config.packs != PACKS:
        raise AgentRegistryError(
            f"checkpoint {label!r} trained for {config.packs} packs, but the "
            f"live draft has {PACKS}"
        )
    if pack_size is not None and config.P < pack_size:
        raise AgentRegistryError(
            f"checkpoint {label!r} capacity P={config.P} is smaller than the "
            f"live pack size {pack_size}"
        )


class AgentRegistry:
    """Maps each model label to its :class:`AgentPickService`."""

    def __init__(self, services: dict[str, AgentPickService]) -> None:
        self._services = services

    @property
    def external_labels(self) -> frozenset[str]:
        """Mix labels routed through the pick side-channel (the bound labels)."""
        return frozenset(self._services)

    @classmethod
    def build(
        cls,
        agent_checkpoints: dict[str, Path],
        mix_labels: set[str],
        options: BuildOptions,
    ) -> "AgentRegistry":
        """Validate labels + geometry and construct one service per bound label.

        ``mix_labels`` is every label in ``--agent-mix``; each must be a Forge
        built-in or a bound checkpoint (FR-011). ``options.pack_size`` checks
        each checkpoint's ``P`` (FR-012).

        ``options.preloaded`` labels count as **bound** for label validation
        and go through the same geometry checks, so there is still exactly one
        place that owns "every mix label is a built-in, a frozen agent, or the
        learner".
        """
        preloaded = dict(options.preloaded or {})
        _check_mix_labels(mix_labels, _bound_labels(agent_checkpoints, preloaded))
        services = _load_services(agent_checkpoints, options)
        services.update(preloaded)
        for label, service in services.items():
            _check_geometry(label, service, options.pack_size)
        return cls(services)

    def pick(self, request) -> str:
        """Dispatch a request to its label's service (raises if unbound)."""
        service = self._services.get(request.agent)
        if service is None:
            raise PickFault(f"no model bound for label {request.agent!r}")
        return service.pick(request)

    def reset_draft(self, draft_id: str) -> None:
        for service in self._services.values():
            service.reset_draft(draft_id)
