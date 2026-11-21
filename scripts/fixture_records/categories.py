"""The categories a fixture is filled with, most specific first: a record
lands in the first category it fits. The gen-3 categories are matched first,
then the detector categories, then the shared ones, then the rare members,
which take only what no other category took."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from effects.infrastructure.converted_trees import SourceTrees
from fixture_records.detector_categories import detector_categories
from fixture_records.gen3 import gen3_categories
from fixture_records.rare_members import rare_member_categories
from fixture_records.shapes import acting_key
from fixture_records.sidecars import Sidecars

Test = Callable[[dict], bool]


def _is_kind(kind: str, discriminator: str | None = None) -> Test:
    """A record of ``kind`` whose moment or subkind is ``discriminator``, if given."""
    def test(record: dict) -> bool:
        if record["kind"] != kind:
            return False
        return discriminator is None or (
            record.get("moment") or record.get("subkind")) == discriminator
    return test


_OBSERVED_EFFECT = _is_kind("resolution", "resolution")


def _modal(sidecars: Sidecars) -> Test:
    """A resolution of a ``Charm`` (a modal spell or ability)."""
    def test(record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "resolution" and key is not None
                and sidecars.api_type(key) == "Charm")
    return test


def _interventional(record: dict) -> bool:
    return _OBSERVED_EFFECT(record) and record["interventional"]


def _combat_probe(record: dict) -> bool:
    return record["kind"] == "combat" and bool(record["payload"].get("probed_keyword"))


def _blockers_forbidden(record: dict) -> bool:
    return _is_kind("playability", "blockers")(record) and bool(
        record["payload"].get("forbidden"))


def _linked_effect(record: dict) -> bool:
    return _OBSERVED_EFFECT(record) and not record["fork"] and bool(record.get("link_id"))


def _trigger(fired: bool) -> Test:
    return lambda record: record["kind"] == "trigger" and bool(
        record["payload"].get("fired")) == fired


def _observed_combat(record: dict) -> bool:
    return record["kind"] == "combat" and not record["fork"]


def shared_categories(sidecars: Sidecars) -> dict[str, Test]:
    """The categories every fixture fills, in the order they are matched."""
    return {
        "modal": _modal(sidecars),
        "interventional": _interventional,
        "combat-probe": _combat_probe,
        "blockers-forbidden": _blockers_forbidden,
        "activation": _is_kind("resolution", "activation"),
        "effect": _linked_effect,
        "trigger-fired": _trigger(True),
        "trigger-unfired": _trigger(False),
        "continuous": _is_kind("continuous"),
        "rewrite": _is_kind("rewrite"),
        "combat": _observed_combat,
        "decision": _is_kind("playability", "decision"),
        "attackers": _is_kind("playability", "attackers"),
        "blockers": _is_kind("playability", "blockers"),
    }


def categories_for(sidecars: Sidecars, folders: list[Path]) -> dict[str, Test]:
    """The fixture's categories, in the order they are matched; ``folders``
    are the converted trees, which gen 3's detectors read through a sidecar
    cache of their own."""
    lookup = SourceTrees.of(folders).sidecar_cache()
    return {**gen3_categories(sidecars, lookup), **detector_categories(sidecars),
            **shared_categories(sidecars), **rare_member_categories(sidecars)}
