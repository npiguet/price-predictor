"""The join between runtime Forge traits and converted ability lines.

Records name abilities by :class:`ProvenanceKey`; the sidecar maps keys to
rendered lines and aligns the ability cache. Pure dataclasses; the JSON is
``infrastructure/sidecar_io.py``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import NamedTuple, Protocol

from effects.domain.keyword_names import display_name_of, normalize_keyword
from effects.domain.script_text import is_bare_keyword

#: The ``script_api_type`` the converter writes on a line a keyword produced.
KEYWORD_API_TYPE = "Keyword"


class TraitKind(StrEnum):
    """The Forge trait a provenance key names, in the sidecar's spelling.

    A key keeps its ``trait_kind`` as the string the record or sidecar
    carries; the enum is what code compares it against.
    """

    KEYWORD = "keyword"
    SPELL = "spell"
    TRIGGER = "trigger"
    STATIC = "static"
    REPLACEMENT = "replacement"


class LineKind(StrEnum):
    """What a rendered line is, the prefix the converter writes before it.

    A line keeps its ``line_kind`` as the string the sidecar carries; the
    enum is what code compares it against.
    """

    ACTIVATED = "activated"
    TRIGGERED = "triggered"
    STATIC = "static"
    REPLACEMENT = "replacement"
    CHAPTER = "chapter"
    LEVEL = "level"
    ALTERNATE_COST = "alternate cost"
    COST_REDUCTION = "cost reduction"
    ADDITIONAL_COST = "additional cost"
    DRAFT = "draft"
    SPELL = "spell"
    PLANESWALKER = "planeswalker"
    OPTION = "option"
    TEXT = "text"


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


class KeyResolution(StrEnum):
    """What a provenance key resolves to at the join.

    ``DROPPED`` has no text by design (Forge's implicit permanent spell, most
    often); ``RUNTIME_ONLY`` has text the join cannot reach
    (:meth:`ProvenanceSidecar.is_runtime_only`); ``UNCONVERTED`` is a script no
    sidecar describes; ``UNCONFIGURED_TREE`` names a source tree the run was
    given no root for (a variant key without ``--variant-scripts``), so whether
    it joins is unknown. ``AMBIGUOUS`` is a keyword's bare key acting where the
    keyword prints several lines (a saga, haunt) and no line declares the key:
    it names none of them, so it has no text and is counted
    (:meth:`ProvenanceSidecar.board_keys` gives a board line its lines). A key
    inside a declared range and in neither list is not a value: it raises.
    """

    LINE = "line"
    DROPPED = "dropped"
    RUNTIME_ONLY = "runtime-only"
    UNCONVERTED = "unconverted"
    UNCONFIGURED_TREE = "unconfigured-tree"
    AMBIGUOUS = "ambiguous"


class SidecarError(ValueError):
    """A sidecar breaks its contract: two lines of a face declare one key."""


@dataclass(frozen=True, slots=True)
class BuiltTrait:
    """The trait a keyword instance built, by its kind and position.

    ``index`` is the trait's position in that instance's list of ``kind``
    (``KeywordInterface.getTriggers()``, ``getReplacements()``,
    ``getStaticAbilities()``, ``getAbilities()``): a saga's chapter III is
    ``trigger`` 2. ``kind`` is never ``keyword``, since a keyword builds the
    other four kinds of trait.
    """

    kind: str
    index: int

    def as_dict(self) -> dict:
        return {"kind": self.kind, "index": self.index}

    @classmethod
    def from_dict(cls, data: dict) -> BuiltTrait:
        return cls(kind=str(data["kind"]), index=int(data["index"]))


@dataclass(frozen=True, slots=True)
class ProvenanceKey:
    """``(script_file, face, trait_kind, index_within_kind, option?, built?)``.

    ``script_file`` includes its tree. ``index_within_kind`` indexes that
    kind's slice of the face's raw trait list, not a global ordinal.
    ``option`` is a charm mode's 0-based position in the root's ``Choices$`` —
    Forge gives a mode no identity of its own, so the position is the one both
    converter and collector can compute. A key without it names the root.
    ``built`` is set only on a ``keyword`` key, and names which of the traits
    that keyword instance built the key is about; a key without it names the
    keyword itself.
    """

    script_file: str
    face: int
    trait_kind: str
    index_within_kind: int
    option: int | None = None
    built: BuiltTrait | None = None

    def __post_init__(self) -> None:
        if self.built is not None:
            _check_built(self.trait_kind, self.built)

    @property
    def tree(self) -> str:
        """The source tree this key's script file belongs to."""
        return split_script_file(self.script_file)[0]

    @property
    def label(self) -> str:
        """The key as a report names it."""
        suffix = f" option {self.option}" if self.option is not None else ""
        if self.built is not None:
            suffix += f" built {self.built.kind}[{self.built.index}]"
        return (
            f"{self.script_file} face {self.face} {self.trait_kind}"
            f"[{self.index_within_kind}]{suffix}"
        )

    @property
    def root(self) -> ProvenanceKey:
        """The charm's own key for a mode key; the key itself otherwise."""
        if self.option is None:
            return self
        return replace(self, option=None)

    def without_built(self) -> ProvenanceKey:
        """The keyword's own key for a built key; the key itself otherwise."""
        if self.built is None:
            return self
        return replace(self, built=None)

    def as_dict(self) -> dict:
        out = {
            "script_file": self.script_file,
            "face": self.face,
            "trait_kind": self.trait_kind,
            "index_within_kind": self.index_within_kind,
        }
        if self.option is not None:
            out["option"] = self.option
        if self.built is not None:
            out["built"] = self.built.as_dict()
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
            built=None if data.get("built") is None else BuiltTrait.from_dict(data["built"]),
        )


def _check_built(trait_kind: str, built: BuiltTrait) -> None:
    """Refuse a ``built`` member outside its contract."""
    if trait_kind != TraitKind.KEYWORD:
        raise ValueError(f"only a keyword key has a built trait, not a {trait_kind} key")
    if built.kind == TraitKind.KEYWORD:
        raise ValueError("a built trait is never itself a keyword")
    if built.index < 0:
        raise ValueError(f"a built trait's index is a position: {built.index}")


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

    def prose_in(self, rendered: Sequence[str]) -> str | None:
        """This line's converted prose, without its ``kind[n]: `` prefix, out
        of its converted file's ``rendered`` lines; None where they hold none.

        Role spans are offsets into the prose without the prefix. The split is
        ``prose_offset``, since a continuation line has no prefix but may hold
        a ``{T}: ``; an older sidecar without it is cut at the first ``": "``,
        right for every line such sidecars hold.
        """
        if not (0 <= self.line_index < len(rendered)):
            return None
        line = rendered[self.line_index]
        if self.prose_offset is not None:
            return line[self.prose_offset:]
        separator = line.find(": ")
        return line if separator < 0 else line[separator + 2:]

    # Two answers to "which keyword is this line", which part company on a
    # ``Keyword`` line whose script carries parameters (``Equip:2:::ReduceCost$
    # Y``, a Class level, a conditional ``etbCounter``): the first names its
    # keyword, the second does not.

    @property
    def keyword_display_name(self) -> str | None:
        """The keyword a ``Keyword`` line names, as Forge displays it
        (``Ward``, ``Protection from red``), whatever parameters follow the
        name; None on a line no keyword produced."""
        if self.script_api_type != KEYWORD_API_TYPE or not self.script_text:
            return None
        return display_name_of(self.script_text)

    @property
    def bare_keyword(self) -> str | None:
        """The keyword this line *is* where its script is the keyword alone,
        no ``$`` parameter and no ``|`` alternative (White Knight's ``First
        Strike``), in the package's spelling (``first_strike``); None
        otherwise, a ``Keyword`` line carrying parameters included."""
        if not is_bare_keyword(self.script_text):
            return None
        return normalize_keyword(self.script_text)


def _note_declared(
    declared: dict[tuple[int, str], int], key: ProvenanceKey,
) -> None:
    slot = (key.face, key.trait_kind)
    highest = declared.get(slot)
    if highest is None or key.index_within_kind > highest:
        declared[slot] = key.index_within_kind


def _first_built_key_per_keyword(
    line: SidecarLine,
) -> Iterator[tuple[ProvenanceKey, ProvenanceKey]]:
    """``(bare key, first built key)`` for each keyword ``line`` declares a
    ``built`` key of."""
    seen: set[ProvenanceKey] = set()
    for key in line.provenance:
        if key.built is not None and key.without_built() not in seen:
            seen.add(key.without_built())
            yield key.without_built(), key


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
    _built_lines: dict[ProvenanceKey, tuple[ProvenanceKey, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        declared: dict[tuple[int, str], int] = {}
        for key in (*(k for line in self.lines for k in line.provenance), *self.dropped_keys):
            _note_declared(declared, key)
        object.__setattr__(self, "_row_by_key", self._key_index())
        object.__setattr__(self, "_built_lines", self._built_line_keys())
        object.__setattr__(self, "_declared_max", declared)
        object.__setattr__(self, "_dropped", frozenset(self.dropped_keys))

    def _key_index(self) -> dict[ProvenanceKey, int]:
        """``key -> row``; raises where two lines declare one key, since the
        join would have to pick one of them."""
        index: dict[ProvenanceKey, int] = {}
        for row, line in enumerate(self.lines):
            for key in dict.fromkeys(line.provenance):
                if index.setdefault(key, row) != row:
                    raise SidecarError(
                        f"{self.script_file}: key {key.label} is declared by lines "
                        f"{self.lines[index[key]].line_index} and {line.line_index}; "
                        "a key names at most one line (reconvert the card)"
                    )
        return index

    def _built_line_keys(self) -> dict[ProvenanceKey, tuple[ProvenanceKey, ...]]:
        """``bare keyword key -> the first built key of each line declaring a
        built key of it``, in line order: a keyword that prints several lines."""
        lines: dict[ProvenanceKey, list[ProvenanceKey]] = {}
        for line in self.lines:
            for bare, built in _first_built_key_per_keyword(line):
                lines.setdefault(bare, []).append(built)
        return {bare: tuple(keys) for bare, keys in lines.items()}

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

    def script_texts(self) -> list[str]:
        """The script text of every line that has one, in line order."""
        return [line.script_text for line in self.lines if line.script_text]

    def row_for(self, key: ProvenanceKey) -> int | None:
        """The ability-cache row a key resolves to, or None where it names no
        line (dropped, runtime-only, ambiguous)."""
        return self._resolve(key)[1]

    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """Which of the join's answers this key gets; raises on a mismatch.

        Unlike ``row_for``, tells a dropped key from a runtime-only one.
        """
        return self._resolve(key)[0]

    def board_keys(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The keys a board line of ``key`` reads as slots, one per line.

        A keyword's bare key that no line declares, where the keyword prints
        several lines declaring ``built`` keys (a saga's chapters), is all of
        those lines in line order, each read through its first ``built`` key;
        any other key is itself.
        """
        if key in self._row_by_key:
            return (key,)
        return self._built_lines.get(key, (key,))

    def _resolve(self, key: ProvenanceKey) -> tuple[KeyResolution, int | None]:
        """The join's answer for a key and its row, the one place it is decided.

        Exactly the line declaring the key; else, for a ``built`` key, as the
        keyword's bare key, whose one line holds its text; a bare key naming a
        keyword of several lines is ``AMBIGUOUS``.
        """
        row = self._row_by_key.get(key)
        if row is not None:
            return KeyResolution.LINE, row
        if key.built is not None:
            return self._resolve(key.without_built())
        if key in self._built_lines:
            return KeyResolution.AMBIGUOUS, None
        if key in self._dropped:
            return KeyResolution.DROPPED, None
        if self.is_runtime_only(key):
            return KeyResolution.RUNTIME_ONLY, None
        raise KeyError(self._mismatch_message(key))

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The rendered line a key resolves to, or None where it names none
        (dropped, runtime-only, ambiguous).

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
            if line.line_kind != LineKind.OPTION or not any(
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


class Joined(NamedTuple):
    """The join's answer for one key: its resolution, and the line it names,
    None wherever it names none (every resolution but ``LINE``)."""

    resolution: KeyResolution
    line: SidecarLine | None


class SidecarLookup(Protocol):
    """The sidecar join as its readers use it, by provenance key.

    ``infrastructure.sidecar_io.SidecarCache`` implements it over the converted
    trees, every lookup below reading through :meth:`join`; a test double need
    implement only what its caller reads.
    """

    def get(self, script_file: str) -> ProvenanceSidecar:
        """The sidecar of one script file; ``KeyError`` where none was written."""
        ...

    def join(self, key: ProvenanceKey) -> Joined:
        """The key's resolution and line, every no-line answer counted;
        ``KeyError`` on a key the sidecar does not describe."""
        ...

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """:meth:`join`'s line: None where there is none; ``KeyError`` on a key
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


def first_text[T](
    keys: Iterable[ProvenanceKey], text_of: Callable[[ProvenanceKey], T | None],
) -> T | None:
    """The text of the first key ``text_of`` names one for, or None.

    The acting line is the first acting key that resolves, so every reader of
    "the text this record acts through" folds a key tuple this way, whatever
    ``text_of`` resolves to — ``[ACT]``'s ``e`` included, which is why the
    check is against None: row 0 of the encoded matrix is a valid hit.
    """
    for key in keys:
        text = text_of(key)
        if text is not None:
            return text
    return None
