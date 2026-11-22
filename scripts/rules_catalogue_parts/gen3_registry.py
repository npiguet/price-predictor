"""``rules/gen3-fields.json``: the registry of gen-3 inputs, outputs, extensions and fixes."""

import json
from collections import defaultdict

from .locations import RULES


def requirement_closure(entries, ids):
    """``ids`` and every entry they require, directly or through another."""
    out, todo = set(ids), list(ids)
    while todo:
        for req in entries[todo.pop()].get("requires", ()):
            if req not in out:
                out.add(req)
                todo.append(req)
    return out


_GEN3_IDS = None


def gen3_ids():
    global _GEN3_IDS
    if _GEN3_IDS is None:
        _GEN3_IDS = gen3_registry()[1].keys()
    return _GEN3_IDS


GEN3_FIELDS_NAME = "gen3-fields.json"
GEN3_FIELDS = RULES / GEN3_FIELDS_NAME
GEN3_KEYS = ("gen3_inputs", "gen3_outputs", "gen3_fixes", "gen3_dropped", "learnable_after_gen3")
#: The registry's member listing the rules learnable at the last run, written by gen3-fields.
LEARNABLE = "learnable"

_OUTPUT_PLANNED = ("planned.timing", "planned.failed_checks")


def _entry_contexts(reg):
    """``(entry, contexts)`` of every registry entry, the contexts its names are looked up in."""
    for e in reg["inputs"]:
        yield e, ["input"]
    for e in reg["outputs"] + reg["fixes"]:
        yield e, ["output"]
    for e in reg["extensions"]:
        yield e, [c for c in ("input", "output") if c in e["applies_to"]]
    for e in reg["planned"]:
        yield e, ["output"] if e["id"] in _OUTPUT_PLANNED else ["input"]


def _index_names(reg):
    """``(entries by id, {context: {name: {entry ids}}})``."""
    entries = {}
    by_context = {"input": defaultdict(set), "output": defaultdict(set)}
    for entry, contexts in _entry_contexts(reg):
        entries[entry["id"]] = entry
        for c in contexts:
            for name in entry["names"]:
                by_context[c][name].add(entry["id"])
    return entries, by_context


def _context_lookup(context, names, reg):
    """``{name: id, "covered" or None}`` of one context."""
    lookup = {}
    for name, ids in names.items():
        if len(ids) > 1:
            raise ValueError(f"{name} maps to {sorted(ids)} as an {context}")
        lookup[name] = next(iter(ids))
    for name in reg["covered"]["names"]:
        lookup.setdefault(name, "covered")
    for name in (n for group in reg["dropped"] for n in group["names"]):
        lookup.setdefault(name, None)
    return lookup


def gen3_registry(path=None):
    """``(entries by id, {context: {name: id or None}})`` from gen3-fields.json.

    A name is looked up in its own context: a missing input among the inputs, an
    untrained output among the outputs, because a few names (``new:text_change``,
    ``new:room_unlocked``) were proposed in both roles. Within a context a name maps
    to one entry; ``None`` is covered or dropped.
    """
    reg = json.loads((path or GEN3_FIELDS).read_text(encoding="utf-8"))
    entries, by_context = _index_names(reg)
    lookup = {c: _context_lookup(c, names, reg) for c, names in by_context.items()}
    overrides = {(o["rule"], o["name"]): o["to"] for o in reg["overrides"]}
    return reg, entries, lookup, overrides
