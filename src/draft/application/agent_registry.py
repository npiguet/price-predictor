"""Registry of model-piloted mix labels → their pick services (data-model §2.6).

Parses the ``--agent-checkpoint LABEL=PATH`` bindings, validates every mix label
(a Forge built-in or a bound checkpoint — FR-011) and each checkpoint's geometry
against the live draft (FR-012), and builds one :class:`AgentPickService` per
bound label. Exposes the set of external (model) labels the worker routes
through the pick side-channel. Built once at supervisor startup.

Geometry validation mirrors the fail-fast width checks in
``train_draft_agent._check_dims`` (research D5): ``config.packs`` must equal the
live ``PACKS`` and ``config.P`` must be ≥ the live pack size when it is known
(``--set`` runs); for random-set runs the live pack size is unknown at startup
and the per-request ``pick_number ≤ P`` guard in :class:`AgentPickService`
catches any overflow as a pick fault instead.
"""

from __future__ import annotations

from pathlib import Path

from draft.application.agent_pick_service import AgentPickService, PickFault
from draft.application.generate_draft_data import PACKS

# Forge AI built-in agent labels (no checkpoint binding required).
FORGE_BUILTINS = frozenset({"forge-full", "forge-r30", "forge-r100"})


class AgentRegistryError(ValueError):
    """Raised on an invalid label binding or geometry mismatch (FR-011/FR-012)."""


def parse_agent_checkpoints(specs: list[str]) -> dict[str, Path]:
    """Parse repeatable ``LABEL=PATH`` (bare ``PATH`` ⇒ label ``draft-agent``).

    Raises :class:`AgentRegistryError` on a malformed spec or a duplicate label.
    """
    bindings: dict[str, Path] = {}
    for spec in specs:
        label, sep, path_str = spec.partition("=")
        if sep:
            label = label.strip()
            path_str = path_str.strip()
        else:
            label, path_str = "draft-agent", spec.strip()
        if not label or not path_str:
            raise AgentRegistryError(
                f"--agent-checkpoint {spec!r} must be 'LABEL=PATH' or a bare PATH"
            )
        if label in bindings:
            raise AgentRegistryError(f"duplicate --agent-checkpoint label {label!r}")
        bindings[label] = Path(path_str)
    return bindings


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
        *,
        locator,
        pick_mode: str = "argmax",
        temperature: float = 1.0,
        seed: int | None = None,
        pack_size: int | None = None,
        device=None,
    ) -> "AgentRegistry":
        """Validate labels + geometry and construct one service per bound label.

        ``mix_labels`` is every label in ``--agent-mix``; each must be a Forge
        built-in or a bound checkpoint (FR-011). ``pack_size`` (when known, e.g.
        ``--set`` runs) checks each checkpoint's ``P`` (FR-012).
        """
        for label in mix_labels:
            if label not in FORGE_BUILTINS and label not in agent_checkpoints:
                raise AgentRegistryError(
                    f"mix label {label!r} is neither a Forge built-in "
                    f"({', '.join(sorted(FORGE_BUILTINS))}) nor bound via "
                    "--agent-checkpoint"
                )
        services: dict[str, AgentPickService] = {}
        for label, path in agent_checkpoints.items():
            if not path.exists():
                raise AgentRegistryError(f"checkpoint for label {label!r} not found: {path}")
            service = AgentPickService(
                path, locator, device=device,
                pick_mode=pick_mode, temperature=temperature, seed=seed,
            )
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
            services[label] = service
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
