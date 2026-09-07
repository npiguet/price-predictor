"""The join between runtime Forge traits and converted ability lines.

Every effect record names an ability through a :class:`ProvenanceKey`, and every
ability-cache row is aligned by the sidecar that maps those keys to rendered
lines. This is the most load-bearing contract in the feature, so the shapes live
here as pure dataclasses; reading and writing the JSON is
``infrastructure/sidecar_io.py``'s job (the same split
``draft/domain/draft_geometry.py`` makes against ``draft_record_io``).

Converted-line ordinals cannot be the key. The converter runs one per-face
bracket counter across five mixed line kinds, emits keyword-derived lines first
in the file though Forge appends them last in its runtime lists, and merges,
deduplicates and splits abilities relative to the runtime objects. An ordinal
therefore names different things on the two sides. Printed provenance is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The three converted source trees a key can name. They differ in layout as well
# as in content: ``cardsfolder`` is letter-keyed and ``tokenscripts`` is flat, so
# Ajani's Pridemate is ``cardsfolder/a/ajanis_pridemate.txt`` on one side and
# ``tokenscripts/ajanis_pridemate.txt`` on the other.
SOURCE_TREES: tuple[str, ...] = ("cardsfolder", "tokenscripts", "variant-scripts")

# Prose role tags spanning the converted text of one line.
ROLES: tuple[str, ...] = ("cost", "effect", "trigger-condition", "target-spec")


@dataclass(frozen=True, slots=True)
class ProvenanceKey:
    """``(script_file, face, trait_kind, index_within_kind)``.

    ``script_file`` is the path as written, including its tree, and is produced
    identically by the Java writer and the Python reader — a disagreement is the
    fail-loudly case at the join. ``index_within_kind`` indexes that kind's slice
    of the face's raw trait list, not a global ordinal.
    """

    script_file: str
    face: int
    trait_kind: str
    index_within_kind: int

    @property
    def tree(self) -> str:
        """The source tree this key's script file belongs to."""
        return self.script_file.split("/", 1)[0]

    def as_dict(self) -> dict:
        return {
            "script_file": self.script_file,
            "face": self.face,
            "trait_kind": self.trait_kind,
            "index_within_kind": self.index_within_kind,
        }

    @classmethod
    def from_dict(cls, data: dict, *, script_file: str | None = None) -> ProvenanceKey:
        """Parse a key, taking ``script_file`` from the sidecar's header.

        Sidecar entries omit the script file because it is the same for every key
        in one file; a key carried on a record spells it out.
        """
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
        )


@dataclass(frozen=True, slots=True)
class SubAbilityLink:
    """One sub-ability below a trait, addressed by its index path."""

    path: tuple[int, ...]
    label: str


@dataclass(frozen=True, slots=True)
class RoleSpan:
    """A character range over the converted prose, tagged with its role."""

    start: int
    end: int
    role: str


@dataclass(frozen=True, slots=True)
class SidecarLine:
    """One rendered line of a converted card, with everything that produced it.

    ``provenance`` holds several keys where the converter merged several runtime
    traits into one line, so the join is many-to-one and never assumed
    one-to-one. ``script_text`` is the stage-four primary encoding surface, which
    is why every command that encodes reads it here and needs no Forge
    cardsfolder path of its own.

    ``line_index`` points into the converted ``.txt``'s rendered lines. It is not
    the ability-cache row: that is this entry's position in the sidecar's
    ``lines`` array, which is what the cache layout aligns to.
    """

    line_index: int
    line_kind: str
    provenance: tuple[ProvenanceKey, ...]
    sub_ability_links: tuple[SubAbilityLink, ...] = ()
    script_api_type: str | None = None
    script_param_keys: tuple[str, ...] = ()
    script_text: str | None = None
    role_spans: tuple[RoleSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvenanceSidecar:
    """``<name>.provenance.json``, read beside the converted ``<name>.txt``.

    ``dropped_keys`` lists traits that map to no rendered line. It exists
    because the collector computes keys from runtime trait accessors, never from
    the sidecar: a trait the converter deduplicated away is still live at
    runtime and still produces a key, so without this list an expected dedup and
    a genuine corpus/sidecar mismatch would look identical at the join — and the
    mismatch is the one condition that must fail loudly.
    """

    card: str
    script_file: str
    lines: tuple[SidecarLine, ...]
    dropped_keys: tuple[ProvenanceKey, ...] = ()
    # key -> position in ``lines``, built once per card and reused across the
    # millions of records that join through it.
    _row_by_key: dict[ProvenanceKey, int] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        index: dict[ProvenanceKey, int] = {}
        for row, line in enumerate(self.lines):
            for key in line.provenance:
                index.setdefault(key, row)
        object.__setattr__(self, "_row_by_key", index)

    @property
    def tree(self) -> str:
        return self.script_file.split("/", 1)[0]

    def row_for(self, key: ProvenanceKey) -> int | None:
        """The ability-cache row a key resolves to, or None where it was dropped."""
        row = self._row_by_key.get(key)
        if row is not None:
            return row
        if key in self.dropped_keys:
            return None
        raise KeyError(self._mismatch_message(key))

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The rendered line a key resolves to, or None where it was dropped.

        Raises when the key is in neither ``lines`` nor ``dropped_keys``: the
        sidecar does not describe the card the record was collected against,
        which means the corpus was reconverted between collection and training.
        """
        row = self.row_for(key)
        return None if row is None else self.lines[row]

    def _mismatch_message(self, key: ProvenanceKey) -> str:
        return (
            f"provenance key {key} appears in neither the lines nor the "
            f"dropped_keys of {self.script_file}; the sidecar does not describe "
            "the card this record was collected against (reconversion between "
            "collection and training?)"
        )
