"""Every event parameter ``EVENT_PARAMS`` declares is one a writer writes.

A declared parameter no writer writes is a dead field: ``field-coverage``
finds it only after a collection, and a reader that reads it reads nothing.
The writers are the connector's Java sources and Forge's own outcome notes
(``EffectRecordOutcomes``), read here as text, so the check needs no JVM.

A type and a parameter are paired where one source writes both: a file that
names the type (``EventType.X``) and writes the parameter (``EventParam.Y``
or an ``EffectEvent`` helper that writes it); a mode row (``ModeRow.of``) by
the slots it lists; a Forge note by the names its ``params(...)`` call spells.
A pair written across two files is listed with the file that writes it.
"""

from __future__ import annotations

import re

from effects.domain.event_schema import EVENT_PARAMS, Event
from tests.unit.forge_sources import CONNECTOR, FORGE_GAME, forge_source

_EFFECTS = CONNECTOR / "effects"
#: The mode rows and their slots, paired by :func:`_mode_row_pairs` instead.
_ROW_FILES = frozenset({"RunParamRows.java", "RunParamSlots.java", "CounterSlot.java"})
_TABLES = frozenset({"EventType.java", "EventParam.java"})
#: A parameter named only to be read or dropped, which writes nothing.
_READ_FORMS = re.compile(r"(?:\.param|hasParam|removeParam)\(\s*EventParam\.([A-Z_]+)\s*\)")

#: The writers that stamp a provenance parameter (``cause``) on an event of
#: any type: the bracket's attribution and the run-parameter reading.
_ANY_TYPE_WRITERS = ("EventAttribution.java", "RunParamNormalizer.java")

#: ``(type, parameter)`` written on an event another file builds, with the file.
WRITTEN_ON_ANOTHER_FILES_EVENT: dict[tuple[str, str], str] = {
    # BusEvents builds the move, BusEventReader.placeInLibrary adds its position.
    ("zone_change", "library_position"): "BusEventReader.java",
}

#: Declared pairs no writer writes, each with why it stays declared.
KNOWN_UNWRITTEN: dict[tuple[str, str], str] = {}


def _constants(name: str) -> dict[str, str]:
    """An enum's constant names to their wire names."""
    text = (_EFFECTS / name).read_text(encoding="utf-8")
    return dict(re.findall(r'^\s*([A-Z_]+)\("([a-z_]+)"', text, re.MULTILINE))


_TYPES = _constants("EventType.java")
_PARAMS = _constants("EventParam.java")


def _body(text: str, start: int) -> str:
    """The brace-delimited block that opens first at or after ``start``."""
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[opening:index + 1]
    return text[opening:]


def _call_arguments(text: str, start: int) -> str:
    """The text up to the parenthesis closing the call open at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth < 0:
            return text[start:index]
    return text[start:]


def _params_written(text: str) -> set[str]:
    """The parameters ``text`` names as ``EventParam`` constants, the read forms aside."""
    stripped = _READ_FORMS.sub("", text)
    return {_PARAMS[name] for name in re.findall(r"EventParam\.([A-Z_]+)\b", stripped)
            if name in _PARAMS}


def _helpers() -> dict[str, set[str]]:
    """``EffectEvent``'s own writing methods (``flag``, ``cause``) and what each writes."""
    text = (_EFFECTS / "EffectEvent.java").read_text(encoding="utf-8")
    helpers = {}
    for match in re.finditer(r"EffectEvent (\w+)\([^)]*\)\s*\{", text):
        if match.group(1) not in {"param", "paramNamed"}:
            helpers[match.group(1)] = _params_written(_body(text, match.start()))
    return {name: written for name, written in helpers.items() if written}


def _file_pairs(helpers: dict[str, set[str]]) -> set[tuple[str, str]]:
    """Every type a source names, paired with every parameter it writes."""
    pairs = set()
    for path in sorted(_EFFECTS.rglob("*.java")):
        if path.name in _ROW_FILES | _TABLES:
            continue
        text = path.read_text(encoding="utf-8")
        types = {_TYPES[name] for name in re.findall(r"EventType\.([A-Z_]+)\b", text)
                 if name in _TYPES}
        written = _params_written(text) | {
            param for name, params in helpers.items()
            if re.search(rf"\.{name}\(", text) for param in params}
        pairs |= {(event_type, param) for event_type in types for param in written}
    return pairs


def _slot_writes() -> dict[str, set[str]]:
    """Each slot ``RunParamSlots`` declares, or makes, to the parameters it writes."""
    text = (_EFFECTS / "RunParamSlots.java").read_text(encoding="utf-8")
    slots = {}
    for name, method in re.findall(r"ModeRow\.Slot (\w+) = RunParamSlots::(\w+);", text):
        slots[name] = _params_written(_body(text, re.search(rf"\bvoid {method}\(", text).start()))
    for match in re.finditer(r"static ModeRow\.Slot (\w+)\(", text):
        slots[match.group(1)] = _params_written(_body(text, match.start()))
    slots["counters"] = _params_written((_EFFECTS / "CounterSlot.java").read_text(encoding="utf-8"))
    return slots


def _mode_row_pairs() -> set[tuple[str, str]]:
    """Each mode row's type paired with what the slots it lists write."""
    text = (_EFFECTS / "RunParamRows.java").read_text(encoding="utf-8")
    slots = _slot_writes()
    arrays = {name: re.findall(r"RunParamSlots\.(\w+)", members) for name, members
              in re.findall(r"ModeRow\.Slot\[\] (\w+) = \{([^}]*)\}", text)}
    pairs = set()
    for match in re.finditer(r"ModeRow\.of\(EventType\.([A-Z_]+)", text):
        for word in re.findall(r"\w+", _call_arguments(text, match.end())):
            for slot in arrays.get(word, [word]):
                pairs |= {(_TYPES[match.group(1)], param) for param in slots.get(slot, ())}
    return pairs


def _forge_note_pairs() -> set[tuple[str, str]]:
    """Each Forge outcome note's type paired with the names its call spells."""
    path = forge_source(FORGE_GAME / "ability" / "EffectRecordOutcomes.java")
    text = path.read_text(encoding="utf-8")
    types = dict(re.findall(r'static final String ([A-Z_]+) = "([a-z_]+)";', text))
    pairs = set()
    for match in re.finditer(r"\bnote\(sa, ([A-Z_]+),", text):
        names = re.findall(r'"([a-z_]+)"', _call_arguments(text, match.end()))
        pairs |= {(types[match.group(1)], name) for name in names}
    return pairs


def _provenance_pairs(helpers: dict[str, set[str]]) -> set[tuple[str, str]]:
    """Every type paired with each provenance parameter an any-type writer stamps."""
    stamped = set()
    for name in _ANY_TYPE_WRITERS:
        text = (_EFFECTS / name).read_text(encoding="utf-8")
        stamped |= {param for helper, params in helpers.items()
                    if re.search(rf"\.{helper}\(", text) for param in params}
    return {(event_type, param) for event_type in _TYPES.values()
            for param in stamped & Event.PROVENANCE_PARAMS}


def _written_pairs() -> set[tuple[str, str]]:
    helpers = _helpers()
    return (_file_pairs(helpers) | _provenance_pairs(helpers) | _mode_row_pairs()
            | _forge_note_pairs())


def _declared_pairs() -> set[tuple[str, str]]:
    return {(event_type.value, param) for event_type, params in EVENT_PARAMS.items()
            for param in params}


def test_every_declared_parameter_is_written():
    unwritten = _declared_pairs() - _written_pairs() - set(WRITTEN_ON_ANOTHER_FILES_EVENT)
    assert unwritten == set(KNOWN_UNWRITTEN), (
        f"declared and written by no writer: {sorted(unwritten - set(KNOWN_UNWRITTEN))}; "
        f"written now, so off the known list: {sorted(set(KNOWN_UNWRITTEN) - unwritten)}")


def test_a_pair_written_on_another_file_s_event_is_written_there():
    for (_, param), name in WRITTEN_ON_ANOTHER_FILES_EVENT.items():
        assert param in _params_written((_EFFECTS / name).read_text(encoding="utf-8")), name


def test_the_known_lists_name_declared_pairs():
    declared = _declared_pairs()
    assert set(WRITTEN_ON_ANOTHER_FILES_EVENT) <= declared
    assert set(KNOWN_UNWRITTEN) <= declared


def test_the_scan_pairs_a_mode_row_by_its_slots_alone():
    """A row's type takes only its own slots' parameters, never one another
    row in the same table writes: ``destroyed`` reads ``regenerable``, and
    ``sacrificed``, whose row lists no slot, reads nothing."""
    pairs = _mode_row_pairs()
    assert ("destroyed", "regenerable") in pairs
    assert not {pair for pair in pairs if pair[0] == "sacrificed"}


def test_a_read_of_a_parameter_writes_nothing():
    text = ("if (e.hasParam(EventParam.SOURCE)) e.param(EventParam.AMOUNT); "
            "e.param(EventParam.COMBAT, x);")
    assert _params_written(text) == {"combat"}
