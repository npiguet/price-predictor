"""Read the rules catalogue under ``rules/``: its scenarios and its heads map.

A loader without the checks: ``scripts/rules_catalogue.py validate`` is what
refuses a malformed sidecar, and every reader here assumes the tree it reads
validates. It lives in the package so ``rule-counts`` and ``build-corpus`` read
the catalogue without importing a script. Prior art: the validators
``validate_scenarios`` and ``validate_heads`` in ``scripts/rules_catalogue.py``,
which parse the same files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from effects.domain.canonical_digest import canonical_sha256
from effects.domain.rule_catalogue import (
    Departure,
    DetectorDeclaration,
    ForkOperation,
    MapVerdict,
    NoDetector,
    Polarity,
    RuleMapEntry,
    Scenario,
    ScenarioCatalogue,
    ScenarioVariant,
    ScriptEdit,
    Unclassifiable,
    is_model_output,
)

DEFAULT_RULES_DIR = Path("rules")


def _detector(data: Mapping[str, Any] | None) -> DetectorDeclaration | NoDetector | None:
    if data is None:
        return None
    if "none" in data:
        return NoDetector(str(data["none"]))
    return DetectorDeclaration.from_json(data)


def _operation(data: Any) -> ForkOperation | Unclassifiable | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        return Unclassifiable(str(data["unclassifiable"]))
    return ForkOperation(data)


def _variant(data: Mapping[str, Any], *, is_base: bool) -> ScenarioVariant:
    return ScenarioVariant(
        label=data["label"],
        polarity=Polarity(data["polarity"]),
        delta=data.get("delta") or "",
        expected=data.get("expected") or "",
        operation=_operation(data.get("operation")),
        is_base=is_base,
        script_edit=ScriptEdit.from_json(edit) if (edit := data.get("script_edit")) else None,
    )


def _scenario(unit: str, data: Mapping[str, Any]) -> Scenario:
    departure = data.get("departure")
    return Scenario(
        id=data["id"],
        unit=unit,
        rules=tuple(data.get("rules") or ()),
        record_kind=data["record_kind"],
        variants=tuple(_variant(v, is_base=i == 0)
                       for i, v in enumerate(data.get("variants") or ())),
        detector=_detector(data.get("detector")),
        departure=Departure.from_json(departure) if departure else None,
        cards=tuple(data.get("cards") or ()),
    )


def load_scenarios(rules_dir: Path = DEFAULT_RULES_DIR) -> ScenarioCatalogue:
    """Every scenario under ``rules_dir/scenarios``, in unit-file then file order."""
    scenarios: list[Scenario] = []
    for path in sorted(Path(rules_dir, "scenarios").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        unit = doc.get("unit") or path.stem
        scenarios.extend(_scenario(unit, s) for s in doc.get("scenarios") or ())
    return ScenarioCatalogue(tuple(scenarios))


def _heads_lines(rules_dir: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for path in sorted(Path(rules_dir, "heads").glob("*.jsonl")):
        for text in path.read_text(encoding="utf-8").splitlines():
            if text.strip():
                lines.append(json.loads(text))
    return lines


def load_heads(rules_dir: Path = DEFAULT_RULES_DIR,
               rules: frozenset[str] | None = None) -> dict[str, RuleMapEntry]:
    """``rule id → RuleMapEntry`` from ``rules_dir/heads``.

    ``outputs`` keeps only the untagged outputs the model has; a ``new:``,
    ``record:`` or ``planned:`` entry is dropped until a rerun of the map lists
    it as an existing output (research R4). A rule in ``rules`` with no heads
    line reads ``missing``.
    """
    entries = {
        line["id"]: RuleMapEntry(
            rule=line["id"],
            verdict=MapVerdict(line["output"]),
            outputs=tuple(o for o in line.get("outputs") or () if is_model_output(o)),
        )
        for line in _heads_lines(rules_dir)
    }
    for rule in sorted(rules or ()):
        entries.setdefault(rule, RuleMapEntry(rule=rule, verdict=MapVerdict.MISSING))
    return entries


def heads_digest(rules_dir: Path = DEFAULT_RULES_DIR) -> str:
    """The sha256 canonical digest of every parsed heads line, in id order."""
    return canonical_sha256(sorted(_heads_lines(rules_dir), key=lambda line: line["id"]))


def map_snapshot(entries: Mapping[str, RuleMapEntry]) -> dict[str, dict[str, Any]]:
    """The manifest's ``rule_map``: ``rule → {verdict, outputs}``."""
    return {rule: {"verdict": str(e.verdict), "outputs": list(e.outputs)}
            for rule, e in sorted(entries.items())}
