"""Board abilities: the model's error where a permanent's own text decides (FR-085b).

The state snapshot gives each permanent its spell and activated abilities as
card-slot lines, and none of its statics, triggers or replacements, so the
model reads those only through what they have already done to the board: a
pump already in the toughness, a granted keyword already in the flags. What a
static decides at the moment of the record, such as who may block or what a
destroy turns into, reaches the model as no text at all. This probe measures
what that costs, so gen-3's fix has a baseline: on each subset below, every
output field's loss and the ablation of ``e`` (``ablation.run_ablation``),
beside a control subset of the same kind.

======================  ====================================================
subset                  records
======================  ====================================================
legality-static         ``attackers``/``blockers``: a creature is kept out by
                        a static that is not a keyword line
legality-keyword        kept out by keyword lines only (protection, say),
                        which the keyword flags carry
legality-none           nobody is kept out
resolution-replacement  a resolution half, with a replacement line on a
                        permanent other than the ability's source
resolution-static       a non-keyword static line on another permanent, and
                        no replacement
resolution-none         no other permanent carries a static, trigger or
                        replacement line
======================  ====================================================

Records come from the corpus's validation strata, read in shard order, the
first ``per_subset`` of each subset per stratum, so every arm on one corpus
reads the same records without a frozen list.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

SUBSETS: tuple[str, ...] = (
    "legality-static", "legality-keyword", "legality-none",
    "resolution-replacement", "resolution-static", "resolution-none",
)
#: Records per subset per validation stratum.
RECORDS_PER_SUBSET = 300
#: Validation shards read per stratum before giving up on a short subset.
SHARDS_PER_STRATUM = 200
#: The trait kinds the snapshot leaves out of a permanent's card slots.
UNLISTED_KINDS = frozenset({"static", "trigger", "replacement"})


def _sidecar_errors() -> tuple[type[Exception], ...]:
    from effects.infrastructure.sidecar_io import UnconfiguredTree, UnconvertedScript

    return UnconvertedScript, UnconfiguredTree


def unlisted_kinds(entity, sidecars) -> frozenset[str]:
    """The non-keyword static, trigger and replacement lines on ``entity``'s face.

    Read from the sidecar of the script its printed keys name. A keyword line
    is left out whatever trait it folds in, because the keyword flags already
    carry it.
    """
    from effects.domain.rule_families import KEYWORD_API_TYPE

    if not entity.printed:
        return frozenset()
    try:
        sidecar = sidecars.get(entity.printed[0].script_file)
    except _sidecar_errors():
        return frozenset()
    kinds: set[str] = set()
    for line in sidecar.lines:
        if line.script_api_type == KEYWORD_API_TYPE:
            continue
        for key in line.provenance:
            if key.face == entity.face and key.trait_kind in UNLISTED_KINDS:
                kinds.add(key.trait_kind)
    return frozenset(kinds)


def _keyword_static(keys, sidecars) -> bool | None:
    """Whether a responsible static is a keyword line; None where it resolves to none."""
    from effects.domain.rule_families import KEYWORD_API_TYPE

    for key in keys:
        line = sidecars.line_or_none(key)
        if line is not None:
            return line.script_api_type == KEYWORD_API_TYPE
    return None


def subset_of(record, sidecars) -> str | None:
    """The subset ``record`` belongs to, or None."""
    from effects.domain.records import Moment, PlayabilitySubkind, RecordKind

    if record.kind is RecordKind.PLAYABILITY and record.subkind in (
            PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS):
        verdicts = [_keyword_static(entry.responsible_static, sidecars)
                    for entry in record.payload.forbidden if entry.responsible_static]
        if not record.payload.forbidden:
            return "legality-none"
        if any(verdict is False for verdict in verdicts):
            return "legality-static"
        if verdicts and all(verdict is True for verdict in verdicts):
            return "legality-keyword"
        return None
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        source = record.state.refs.source
        kinds: set[str] = set()
        for entity in record.state.entities:
            if entity.zone == "battlefield" and entity.id != source:
                kinds |= unlisted_kinds(entity, sidecars)
        if "replacement" in kinds:
            return "resolution-replacement"
        if "static" in kinds:
            return "resolution-static"
        if not kinds:
            return "resolution-none"
    return None


def collect(records: Iterable, sidecars, *, per_subset: int = RECORDS_PER_SUBSET
            ) -> dict[str, list]:
    """The first ``per_subset`` records of each subset, in the order read."""
    chosen: dict[str, list] = {name: [] for name in SUBSETS}
    for record in records:
        name = subset_of(record, sidecars)
        if name is not None and len(chosen[name]) < per_subset:
            chosen[name].append(record)
            if all(len(rows) >= per_subset for rows in chosen.values()):
                break
    return chosen


def stratum_records(directory, *, shards: int = SHARDS_PER_STRATUM):
    """The records of the first ``shards`` shards under ``directory``, in path order."""
    from effects.infrastructure.shard_io import iter_shards, read_shard

    for shard in iter_shards(directory)[:shards]:
        yield from read_shard(shard)


def run(probe_model, chosen: Mapping[str, Mapping[str, list]], stats) -> dict:
    """``{subset: {"n": {stratum: n}, "ablation": ...}}``, both strata pooled."""
    from effect_knowledge_probes import ablation

    out: dict[str, dict] = {}
    for name in SUBSETS:
        records = [r for by_subset in chosen.values() for r in by_subset.get(name, [])]
        out[name] = {"n": {stratum: len(by_subset.get(name, []))
                           for stratum, by_subset in chosen.items()}}
        if records:
            out[name]["ablation"] = ablation.run_ablation(probe_model, records, stats)
    return out


def table(result: Mapping[str, dict]) -> str:
    """Per field, each subset's loss and the cost of noise in the card slots."""
    from math import isnan

    def fmt(value) -> str:
        return "" if value is None or isnan(value) else f"{value:.3f}"

    def shown(values) -> bool:
        """A row is worth a line when some subset measured more than rounding."""
        return any(v is not None and not isnan(v) and abs(v) >= 0.0005 for v in values)

    present = [name for name in SUBSETS if result.get(name, {}).get("ablation")]
    fields = sorted({field for name in present for field in result[name]["ablation"]})
    lines = ["# Board abilities", "",
             "Records per subset: " + ", ".join(
                 f"{name} {sum(result[name]['n'].values())}" for name in SUBSETS
                 if name in result) + ".", "",
             "## Loss per field", "",
             "| field | " + " | ".join(present) + " |",
             "|---|" + "---:|" * len(present)]
    for field in fields:
        values = [result[name]["ablation"].get(field, {}).get("base_loss") for name in present]
        if shown(values):
            lines.append(f"| {field} | " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "## Loss increase with noise in the card slots", "",
              "| field | " + " | ".join(present) + " |",
              "|---|" + "---:|" * len(present)]
    for field in fields:
        values = [result[name]["ablation"].get(field, {}).get("noise", {}).get("cards")
                  for name in present]
        if shown(values):
            lines.append(f"| {field} | " + " | ".join(fmt(v) for v in values) + " |")
    return "\n".join(lines) + "\n"
