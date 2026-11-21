"""The conversions a records directory was collected against (finding 11.3).

Each collecting command keeps the conversion it ran against in its ``run.json``
(``record_conversion``). A directory holds one conversion: a collection into
it refuses a tree of another, and ``build-corpus`` joins its shards only
against the tree they were collected against.
"""

from __future__ import annotations

from pathlib import Path

from price_predictor.domain.converted_provenance import (
    CONVERSION_MEMBER,
    Conversion,
    ConversionRefused,
    recorded_conversions,
    require_same_conversion,
)
from price_predictor.infrastructure.converted_provenance import read_document, record_conversion
from price_predictor.infrastructure.run_manifest import MANIFEST_NAME


def records_conversions(records_dir: Path) -> dict[Path, list[Conversion]]:
    """``run.json`` -> the conversions it records, for every ``run.json`` at or
    under ``records_dir`` that records one."""
    found = {}
    for manifest in sorted(Path(records_dir).rglob(MANIFEST_NAME)):
        recorded = recorded_conversions(read_document(manifest), manifest)
        if recorded:
            found[manifest] = recorded
    return found


def require_collectable(records_dir: Path, wanted: Conversion) -> None:
    """Refuse a collection into ``records_dir`` against ``wanted`` when a
    conversion already recorded there differs (nothing recorded: nothing to
    disagree with)."""
    for recorded in records_conversions(records_dir).values():
        for conversion in recorded:
            require_same_conversion(conversion, wanted)


def require_joinable(records_dir: Path, shards: list[Path], wanted: Conversion) -> None:
    """Refuse a ``build-corpus`` over ``shards`` under ``records_dir`` when a
    recorded conversion differs from ``wanted``, or when a shard's directory has
    no ``run.json`` at or above it (within ``records_dir``) recording one."""
    root = Path(records_dir)
    recorded = records_conversions(root)
    for conversions in recorded.values():
        for conversion in conversions:
            require_same_conversion(conversion, wanted)
    _require_recorded(root, {shard.parent for shard in shards}, set(map(_parent, recorded)))


def _parent(manifest: Path) -> Path:
    return manifest.parent


def _require_recorded(root: Path, directories: set[Path], covered: set[Path]) -> None:
    """Refuse the first of ``directories`` whose nearest ``run.json`` is not in ``covered``."""
    owners = {manifest.parent for manifest in root.rglob(MANIFEST_NAME)}
    unrecorded = [d for d in sorted(directories) if _owner(d, root, owners) not in covered]
    if unrecorded:
        raise ConversionRefused(
            f"the shards in {unrecorded[0]} name no conversion: the nearest run.json "
            f"at or above it under {root} records none (the '{CONVERSION_MEMBER}' "
            "member each collecting command writes), so they were collected before "
            "the record existed. Recollect them with this version"
        )


def _owner(directory: Path, root: Path, owners: set[Path]) -> Path | None:
    """The nearest directory at or above ``directory``, up to ``root``, holding a ``run.json``."""
    within = (directory, *directory.parents)[: len(directory.relative_to(root).parts) + 1]
    return next((path for path in within if path in owners), None)


def recorded_forge_commit(run_dir: Path) -> str | None:
    """The Forge commit of the last conversion ``run_dir``'s own ``run.json``
    records (a collection records its own before any worker starts), or None
    when it records none or cannot be read."""
    manifest = Path(run_dir) / MANIFEST_NAME
    try:
        recorded = recorded_conversions(read_document(manifest), manifest)
    except ConversionRefused:
        return None
    return recorded[-1].forge_commit if recorded else None


def record_collection(run_dir: Path, conversion: Conversion) -> None:
    """Keep ``conversion`` in ``run_dir``'s ``run.json`` once ``run_dir`` holds
    no other: refused (:class:`ConversionRefused`) before anything is written
    when a conversion already recorded there differs."""
    require_collectable(run_dir, conversion)
    record_conversion(run_dir, conversion)
