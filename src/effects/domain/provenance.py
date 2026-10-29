"""The join between runtime Forge traits and converted ability lines.

Records name abilities by :class:`ProvenanceKey`; the sidecar maps keys to
rendered lines and aligns the ability cache. Pure dataclasses; the JSON is
``infrastructure/sidecar_io.py``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class SourceTree(StrEnum):
    """The tree a script file's first path segment names.

    ``cardsfolder`` is letter-keyed, ``tokenscripts`` and ``variant-scripts``
    are flat, and ``counterfactual`` is no folder at all: it prefixes the
    copies the trainer stages in memory.
    """

    CARDSFOLDER = "cardsfolder"
    TOKENSCRIPTS = "tokenscripts"
    VARIANT_SCRIPTS = "variant-scripts"
    COUNTERFACTUAL = "counterfactual"


def split_script_file(script_file: str) -> tuple[str, str]:
    """``(tree, path within the tree)`` of a tree-prefixed script file."""
    tree, _, relative = script_file.partition("/")
    return tree, relative


#: The fields every key's dict form carries (``option`` is written only when
#: set); what tells a key dict from any other mapping in a record's raw JSON.
KEY_DICT_FIELDS: frozenset[str] = frozenset({
    "script_file", "face", "trait_kind", "index_within_kind",
})

class KeyResolution(StrEnum):
    """What a provenance key resolves to at the join.

    ``DROPPED`` has no text by design (Forge's implicit permanent spell, most
    often); ``RUNTIME_ONLY`` has text the join cannot reach
    (:meth:`ProvenanceSidecar.is_runtime_only`); ``UNCONVERTED`` is a script no
    sidecar describes. A key inside a declared range and in neither list is not
    a value: it raises.
    """

    LINE = "line"
    DROPPED = "dropped"
    RUNTIME_ONLY = "runtime-only"
    UNCONVERTED = "unconverted"


@dataclass(frozen=True, slots=True)
class ProvenanceKey:
    """``(script_file, face, trait_kind, index_within_kind, option?)``.

    ``script_file`` includes its tree. ``index_within_kind`` indexes that
    kind's slice of the face's raw trait list, not a global ordinal.
    ``option`` is a charm mode's 0-based position in the root's ``Choices$`` —
    Forge gives a mode no identity of its own, so the position is the one both
    converter and collector can compute. A key without it names the root.
    """

    script_file: str
    face: int
    trait_kind: str
    index_within_kind: int
    option: int | None = None

    @property
    def tree(self) -> str:
        """The source tree this key's script file belongs to."""
        return split_script_file(self.script_file)[0]

    @property
    def label(self) -> str:
        """The key as a report names it."""
        suffix = f" option {self.option}" if self.option is not None else ""
        return (
            f"{self.script_file} face {self.face} {self.trait_kind}"
            f"[{self.index_within_kind}]{suffix}"
        )

    @property
    def root(self) -> ProvenanceKey:
        """The charm's own key for a mode key; the key itself otherwise."""
        if self.option is None:
            return self
        return ProvenanceKey(
            self.script_file, self.face, self.trait_kind, self.index_within_kind,
        )

    def as_dict(self) -> dict:
        out = {
            "script_file": self.script_file,
            "face": self.face,
            "trait_kind": self.trait_kind,
            "index_within_kind": self.index_within_kind,
        }
        if self.option is not None:
            out["option"] = self.option
        return out

    def sidecar_dict(self) -> dict:
        """The key as a sidecar spells it: without the shared ``script_file``."""
        out = self.as_dict()
        del out["script_file"]
        return out

    @classmethod
    def from_dict(cls, data: dict, *, script_file: str | None = None) -> ProvenanceKey:
        """Parse a key; a sidecar entry omits ``script_file``, so pass its header's."""
        resolved = data.get("script_file", script_file)
        if resolved is None:
            raise ValueError(
                "provenance key has no script_file and none was supplied by the "
                f"enclosing sidecar: {data!r}"
            )
        return cls(
            script_file=resolved,
            face=int(data["face"]),
            trait_kind=str(data["trait_kind"]),
            index_within_kind=int(data["index_within_kind"]),
            option=None if data.get("option") is None else int(data["option"]),
        )


@dataclass(frozen=True, slots=True)
class SubAbilityLink:
    """One sub-ability below a trait, addressed by its index path."""

    path: tuple[int, ...]
    label: str


class Role(StrEnum):
    """What a span of converted prose does, as ``convert`` tags it.

    Declaration order is the encoder's role-embedding row order, after row 0
    for no role, so a role is appended and never reordered.
    """

    COST = "cost"
    EFFECT = "effect"
    TRIGGER_CONDITION = "trigger-condition"
    TARGET_SPEC = "target-spec"


@dataclass(frozen=True, slots=True)
class RoleSpan:
    """A character range over the converted prose, tagged with its role."""

    start: int
    end: int
    role: Role


@dataclass(frozen=True, slots=True)
class SidecarLine:
    """One rendered line of a converted card, with everything that produced it.

    ``provenance`` holds several keys where the converter merged several
    traits into one line. ``line_index`` points into the converted ``.txt``;
    the ability-cache row is instead this entry's position in ``lines``.
    ``prose_offset`` is ``None`` on a sidecar written before the converter
    recorded it, whose readers cut at the first ``": "``.
    """

    line_index: int
    line_kind: str
    provenance: tuple[ProvenanceKey, ...]
    sub_ability_links: tuple[SubAbilityLink, ...] = ()
    script_api_type: str | None = None
    script_param_keys: tuple[str, ...] = ()
    script_text: str | None = None
    role_spans: tuple[RoleSpan, ...] = ()
    prose_offset: int | None = None


def _note_declared(
    declared: dict[tuple[int, str], int], key: ProvenanceKey,
) -> None:
    slot = (key.face, key.trait_kind)
    highest = declared.get(slot)
    if highest is None or key.index_within_kind > highest:
        declared[slot] = key.index_within_kind


@dataclass(frozen=True, slots=True)
class ProvenanceSidecar:
    """``<name>.provenance.json``, read beside the converted ``<name>.txt``.

    ``dropped_keys`` lists traits that map to no rendered line. The collector
    computes keys from runtime traits, so without it an expected dedup and a
    genuine sidecar mismatch — the one case that must fail loudly — would look
    identical at the join.
    """

    card: str
    script_file: str
    lines: tuple[SidecarLine, ...]
    dropped_keys: tuple[ProvenanceKey, ...] = ()
    _row_by_key: dict[ProvenanceKey, int] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _declared_max: dict[tuple[int, str], int] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _dropped: frozenset[ProvenanceKey] = field(
        default=frozenset(), init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        index: dict[ProvenanceKey, int] = {}
        declared: dict[tuple[int, str], int] = {}
        for row, line in enumerate(self.lines):
            for key in line.provenance:
                index.setdefault(key, row)
                _note_declared(declared, key)
        for key in self.dropped_keys:
            _note_declared(declared, key)
        object.__setattr__(self, "_row_by_key", index)
        object.__setattr__(self, "_declared_max", declared)
        object.__setattr__(self, "_dropped", frozenset(self.dropped_keys))

    def is_runtime_only(self, key: ProvenanceKey) -> bool:
        """Is this a trait Forge attached to a live card rather than one parsed?

        True when the sidecar declared nothing for the key's
        ``(face, trait_kind)``, or not this far along it: a trigger granted by
        another card's static, a disguise creature's face-down face, a level-up
        or bestow ability appended past the parsed list. False inside a
        declared range, where ``dropped_keys`` is declared-minus-claimed, so a
        gap means the sidecar describes a different card.
        """
        highest = self._declared_max.get((key.face, key.trait_kind))
        return highest is None or key.index_within_kind > highest

    @property
    def tree(self) -> str:
        return split_script_file(self.script_file)[0]

    def row_for(self, key: ProvenanceKey) -> int | None:
        """The ability-cache row a key resolves to, or None where it was dropped."""
        row = self._row_by_key.get(key)
        if row is not None:
            return row
        if key in self._dropped or self.is_runtime_only(key):
            return None
        raise KeyError(self._mismatch_message(key))

    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """Which of the join's answers this key gets; raises on a mismatch.

        Unlike ``row_for``, tells a dropped key from a runtime-only one.
        """
        if key in self._row_by_key:
            return KeyResolution.LINE
        if key in self._dropped:
            return KeyResolution.DROPPED
        if self.is_runtime_only(key):
            return KeyResolution.RUNTIME_ONLY
        raise KeyError(self._mismatch_message(key))

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The rendered line a key resolves to, or None where it was dropped.

        Raises on a key inside a declared range in neither list: the corpus was
        reconverted between collection and training.
        """
        row = self.row_for(key)
        return None if row is None else self.lines[row]

    def option_rows_after(self, row: int) -> list[int]:
        """The keyed ``option`` rows that immediately follow ``row``.

        A charm's modes follow its root line in ``Choices$`` order; a die-roll
        ``option`` line carries no key and ends the run.
        """
        rows: list[int] = []
        for following in range(row + 1, len(self.lines)):
            line = self.lines[following]
            if line.line_kind != "option" or not any(
                key.option is not None for key in line.provenance
            ):
                break
            rows.append(following)
        return rows

    def _mismatch_message(self, key: ProvenanceKey) -> str:
        return (
            f"provenance key {key} appears in neither the lines nor the "
            f"dropped_keys of {self.script_file}; the sidecar does not describe "
            "the card this record was collected against (reconversion between "
            "collection and training?)"
        )


class SidecarLookup(Protocol):
    """The sidecar join as its readers use it, by provenance key.

    ``infrastructure.sidecar_io.SidecarCache`` implements it over the converted
    trees; a test double need implement only what its caller reads.
    """

    def get(self, script_file: str) -> ProvenanceSidecar:
        """The sidecar of one script file; ``KeyError`` where none was written."""
        ...

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The line a key names, None where there is none; ``KeyError`` on a key
        the sidecar does not describe."""
        ...

    def line_or_none(self, key: ProvenanceKey) -> SidecarLine | None:
        """:meth:`line_for`, with every join failure read as no line."""
        ...

    def first_line(self, keys: Iterable[ProvenanceKey]) -> SidecarLine | None:
        """The first line any of ``keys`` names, under :meth:`line_or_none`."""
        ...

    def prose_for(self, key: ProvenanceKey) -> str | None:
        """The converted prose of the line a key names."""
        ...

    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """The join's answer for a key."""
        ...

    def script_text(self, key: ProvenanceKey) -> str | None:
        """The normalized script text of the line a key names, None where
        :meth:`line_or_none` gives no line or the line has no script."""
        ...


def first_text(
    keys: Iterable[ProvenanceKey], text_of: Callable[[ProvenanceKey], str | None],
) -> str | None:
    """The text of the first key ``text_of`` names one for, or None.

    The acting line is the first acting key that resolves, so every reader of
    "the text this record acts through" folds a key tuple this way, whatever
    ``text_of`` resolves to.
    """
    for key in keys:
        text = text_of(key)
        if text is not None:
            return text
    return None
