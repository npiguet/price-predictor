"""The startup check of the directed-fork flags (FR-014).

Shared by ``sealed match-outcomes`` and ``effects collect-variants``. Loads no
torch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

#: The rules catalogue the startup check reads (``rules/README.md``).
DEFAULT_RULES_DIR = Path("rules")
#: Prints the rule ids the worker holds a board precondition for.
DIRECTED_RULES_MAIN = "com.pricepredictor.connector.DirectedRulesMain"

_DETECTOR_RANK = {"missing": 0, "none": 1, "untaggable": 2, "same_as": 3, "declared": 4}

#: The scenarios whose declaration names an event its record kind never
#: lists (the effects package's ``DetectorIndex.untaggable``, read off its
#: emittable-events table): no record of theirs can be tagged.
#: ``test_directed_fork_flags.py`` pins this set to that index over the real
#: catalogue.
UNTAGGABLE_SCENARIOS = frozenset({"104.4c-1"})


def _scenarios(rules_dir: Path) -> Iterator[dict]:
    for path in sorted(Path(rules_dir, "scenarios").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        yield from doc.get("scenarios") or ()


def _detector_state(scenario: dict) -> str:
    declaration = scenario.get("detector")
    if declaration is None:
        return "missing"
    for state in ("none", "same_as"):
        if state in declaration:
            return state
    return "untaggable" if scenario.get("id") in UNTAGGABLE_SCENARIOS else "declared"


def _has_classified_variant(scenario: dict) -> bool:
    """Whether a contrasting variant, any after the base one, is classified
    into a fork operation."""
    return any(
        isinstance(variant.get("operation"), str)
        for variant in (scenario.get("variants") or ())[1:]
    )


class _Readiness:
    """The best detector state and the fork classification seen so far, per rule."""

    def __init__(self) -> None:
        self._detector: dict[str, str] = {}
        self._forkable: dict[str, bool] = {}

    def note(self, rule: str, state: str, classified: bool) -> None:
        """One scenario's word on a rule: the higher-ranked state wins, and any
        classified scenario makes the rule forkable."""
        if _DETECTOR_RANK[state] >= _DETECTOR_RANK[self._detector.get(rule, "missing")]:
            self._detector[rule] = state
        self._forkable[rule] = self._forkable.get(rule, False) or classified

    def by_rule(self) -> dict[str, tuple[str, bool]]:
        return {rule: (state, self._forkable[rule]) for rule, state in self._detector.items()}


def rule_readiness(rules_dir: Path = DEFAULT_RULES_DIR) -> dict[str, tuple[str, bool]]:
    """Per rule id of the catalogue: its detector state, and whether a
    contrasting variant of one of its scenarios is classified into a fork
    operation.

    The detector state is ``"declared"`` when some scenario listing the rule
    declares a detector family, else ``"same_as"`` when one declares the
    same fact as another scenario (``{"same_as": id, "reason": ...}``, its
    rule counted under that scenario's), else ``"untaggable"`` when one
    declares a family no record can show (:data:`UNTAGGABLE_SCENARIOS`), else
    ``"none"`` when one declares ``{"none": reason}``, else ``"missing"``.
    The base variant, the first, makes nothing, and an unclassifiable
    variant's operation is an object.

    Read here rather than through ``effects.infrastructure.rules_catalogue_io``,
    which this package may not import; ``test_directed_fork_flags.py`` pins the
    two readings together over the real catalogue.
    """
    readiness = _Readiness()
    for scenario in _scenarios(rules_dir):
        state = _detector_state(scenario)
        classified = _has_classified_variant(scenario)
        for rule in scenario.get("rules") or ():
            readiness.note(rule, state, classified)
    return readiness.by_rule()


def _listed_rules(stdout: str) -> frozenset[str]:
    return frozenset(line.strip() for line in stdout.splitlines() if line.strip())


def registered_directed_rules() -> frozenset[str]:
    """The rule ids one run of ``DirectedRulesMain`` prints: those the worker
    holds a board precondition for (FR-014).

    Raises ``RuntimeError`` when the query fails, and ``FileNotFoundError``
    when the connector JAR or java is missing.
    """
    from price_predictor.infrastructure.jvm_processes import run_forge_worker

    result = run_forge_worker(DIRECTED_RULES_MAIN, input_text="")
    if result.returncode != 0:
        raise RuntimeError(
            f"DirectedRulesMain exited {result.returncode}: {(result.stderr or '').strip()}"
        )
    return _listed_rules(result.stdout)


#: Why a rule's detector state refuses it, by state; ``{rule}`` and
#: ``{scenarios}`` (the catalogue's scenarios directory) are filled in.
_DECLARATION_REFUSALS = {
    "missing": (
        "--directed-fork-rules: {rule} has no detector declaration in {scenarios}"
    ),
    "none": (
        "--directed-fork-rules: {rule} declares its detector none, so "
        "no record of its fork could be tagged"
    ),
    "untaggable": (
        "--directed-fork-rules: {rule} declares a detector only for an event "
        "its record kind never lists, so no record of its fork could be tagged"
    ),
    "same_as": (
        "--directed-fork-rules: {rule} states the same fact as another "
        "scenario (same_as) and is counted under that scenario's rule, so "
        "no record of its fork would be tagged for it: aim at that rule"
    ),
}


def _declaration_error(rule: str, detector: str, rules_dir: Path) -> str | None:
    """Why a rule's detector state refuses it, or None when it is declared."""
    refusal = _DECLARATION_REFUSALS.get(detector)
    if refusal is None:
        return None
    return refusal.format(rule=rule, scenarios=Path(rules_dir, "scenarios"))


def _catalogue_error(rules: Sequence[str], rules_dir: Path) -> str | None:
    """The first listed rule the catalogue refuses: no usable detector, or no
    contrasting variant classified into a fork operation."""
    readiness = rule_readiness(rules_dir)
    for rule in rules:
        detector, forkable = readiness.get(rule, ("missing", False))
        refusal = _declaration_error(rule, detector, rules_dir)
        if refusal is not None:
            return refusal
        if not forkable:
            return (
                f"--directed-fork-rules: {rule} has no contrasting variant "
                "classified into a fork operation"
            )
    return None


def _registration_error(
    rules: Sequence[str], registered: Callable[[], frozenset[str]] | None,
) -> str | None:
    """The first listed rule with no board precondition in the connector."""
    try:
        known = (registered or registered_directed_rules)()
    except (OSError, RuntimeError) as exc:
        return f"--directed-fork-rules: cannot list the registered rules: {exc}"
    for rule in rules:
        if rule not in known:
            return (
                f"--directed-fork-rules: {rule} has no board precondition "
                "registered in forge-connector (DirectedRulesMain does not list it)"
            )
    return None


def directed_fork_error(
    rules: Sequence[str],
    probability: float,
    *,
    rules_dir: Path = DEFAULT_RULES_DIR,
    registered: Callable[[], frozenset[str]] | None = None,
) -> str | None:
    """Why the directed-fork flags cannot run, naming the rule, or None when
    they can (FR-014): the startup check of ``match-outcomes`` and
    ``collect-variants``, never of ``collect-coverage``, whose
    ``--directed-fork-rules`` is a label only.

    A listed rule needs a detector declared in its scenarios (not ``none``,
    not ``same_as``, not one only for an event its record never lists),
    a contrasting variant classified into a fork operation, and a board
    precondition registered in ``forge-connector``, which one run of
    ``DirectedRulesMain`` lists. The JVM is asked once the catalogue's checks
    pass, and not at all when no rule is listed; ``registered`` replaces that
    query, for a test.
    """
    if not 0.0 < probability <= 1.0:
        return f"--directed-fork-probability must be in (0, 1], got {probability}"
    if not rules:
        return None
    return _catalogue_error(rules, rules_dir) or _registration_error(rules, registered)
