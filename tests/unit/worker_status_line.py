"""Worker status lines as ``WorkerStatus`` prints them, for the tests that feed
a supervisor or a log scan worker logs (``price_predictor.domain.worker_status``)."""

from __future__ import annotations

import json

from price_predictor.domain.worker_status import STATUS_PREFIX

#: ``PinnedForgePreferences.PINNED`` as the line spells it.
PINNED_PREFERENCES = {
    "mulligan_rule": "London", "filtered_hands": False, "performance_mode": False,
    "ui_language": "en-US",
}


#: The Forge commit a whole worker's builds name: its engine's and its connector's.
FORGE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def hooks_document(mode: str = "patched", gaps: tuple[str, ...] = (), **builds: object) -> dict:
    """A status line's ``hooks``: ``mode``, ``gaps``, and both builds at
    :data:`FORGE_COMMIT` from a clean checkout, ``builds`` replacing them."""
    return {
        "mode": mode, "gaps": list(gaps), "forge_build_commit": FORGE_COMMIT,
        "forge_build_dirty": False, "connector_forge_commit": FORGE_COMMIT, **builds,
    }


def status_document(lifetime: str) -> dict:
    """A patched, whole worker's status object with every count empty."""
    return {
        "lifetime": lifetime,
        "hooks": hooks_document(),
        "pinned_preferences": dict(PINNED_PREFERENCES),
        "hook_failures": {"connector": {}, "forge": {}},
        "fork_refusals": {},
        "directed": {},
    }


def status_line(lifetime: str = "a", **members: object) -> str:
    """The status line of ``lifetime``, ``members`` replacing the document's
    own: ASCII, compact, one line, as the Java side writes it."""
    document = status_document(lifetime)
    document.update(members)
    return STATUS_PREFIX + json.dumps(document, separators=(",", ":"))


def directed_counts(**given: int) -> dict[str, int]:
    """A directed rule's six counts in the tally's order, those not given at 0."""
    names = ("matched", "drawn", "budget_blocked", "taken", "kept", "precondition_threw")
    return {name: given.get(name, 0) for name in names}
