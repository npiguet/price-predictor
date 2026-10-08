"""Build a fixed training corpus and two fixed validation strata (FR-135).

Two passes. The survey reads every shard and reports what the whole corpus
holds; the main process folds provenance keys to ability texts, decides the
split and the write targets, and the write pass re-reads each shard and keeps
what the decisions admit.

The survey keys on the provenance key rather than the ability text on purpose.
Resolving a key to its text needs a ``SidecarCache`` over tens of thousands of
sidecar files, and building one per worker process would cost more than the
scan it serves. Keys are in the record; the fold happens once, in the process
that already has a cache for the holdout.

Both passes read their shards through ``read_shard_remapped``, which repairs
the old collector's token keys in the raw JSON before a record is built
(FR-151), so the survey's rarity table and caps and the written shards all
name the same ability.
"""

from __future__ import annotations

import logging
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.corpus_curation import (
    SIGNATURE_CLASSES,
    CapHeap,
    CopyPlan,
    outcome_signature,
    record_hash,
    signature_label,
)
from effects.domain.corpus_manifest import SourceShard
from effects.domain.provenance import ProvenanceKey
from effects.domain.record_quality import (
    NO_ACTING_TEXT,
    acting_text_defect,
    has_unattributed_events,
    quality_defect,
)
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.token_key_remap import RemapCounts, TokenKeyRemapper, remap_record_dict

if TYPE_CHECKING:
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)

_KEY_FIELD_SEPARATOR = "|"
_KEY_SEPARATOR = ";"


class BuildCorpusError(ValueError):
    """A build this corpus, these flags or this working directory cannot do.

    Separated from a bare ``ValueError`` so the CLI can report an operator
    condition as one logged line and a non-zero exit while letting an
    internal-invariant violation — ``decide``'s cap mismatch, ``CapHeap.merge``'s
    — keep its traceback. The two arrive at the same ``except`` clause
    otherwise, and a bug in the builder then reads exactly like a misconfigured
    run.

    A ``ValueError`` subclass rather than a fresh exception type, so a caller
    that already handles the broader class keeps working.
    """


def ability_key(record: EffectRecord) -> str | None:
    """A record's whole acting-ability key tuple, as one hashable string.

    The whole tuple rather than its first element: ``ability_text_of`` resolves
    the first key that the sidecars can read, so a survey keyed on the first
    key alone would group records the trainer separates whenever the first key
    names a script with no sidecar.
    """
    if not record.ability:
        return None
    return _KEY_SEPARATOR.join(_render_key(key) for key in record.ability)


def _render_key(key: ProvenanceKey) -> str:
    fields = [key.script_file, str(key.face), key.trait_kind, str(key.index_within_kind)]
    if key.option is not None:
        fields.append(str(key.option))
    return _KEY_FIELD_SEPARATOR.join(fields)


def parse_ability_key(rendered: str) -> tuple[ProvenanceKey, ...]:
    """The inverse of :func:`ability_key`."""
    keys = []
    for part in rendered.split(_KEY_SEPARATOR):
        script_file, face, trait_kind, index, *option = part.split(_KEY_FIELD_SEPARATOR)
        keys.append(ProvenanceKey(
            script_file, int(face), trait_kind, int(index),
            int(option[0]) if option else None,
        ))
    return tuple(keys)


def game_hash(game_id: str) -> int:
    """A stable 32-bit id for a game, for counting distinct games cheaply.

    Rarity needs the *number* of distinct games per text, not their names, and
    holding tens of millions of game-id strings across worker results costs
    hundreds of megabytes for a figure that is a cardinality. ``crc32`` rather
    than the built-in ``hash()``, which is salted per process (FR-088a).
    """
    return zlib.crc32(game_id.encode("utf-8"))


def read_shard_remapped(
    path: Path, remapper: TokenKeyRemapper | None, counts: RemapCounts,
) -> Iterator[EffectRecord]:
    """Every complete record of one shard, its old token keys remapped first.

    The remap works on the raw JSON before the record is built, so both
    passes see the corrected keys and nothing downstream has to know they
    were ever wrong: the survey's rarity entry, the per-text cap it sizes and
    the written shard all name the token script. Reading through the record
    and rewriting it afterwards would not do -- ``ProvenanceKey`` is frozen
    and the remap needs the entity dicts the record drops.

    ``remapper is None`` (``--no-remap-token-keys``) is a plain read, and
    ``counts`` is then left alone.
    """

    from effects.infrastructure.record_io import (
        iter_shard_lines,
        parse_record_line,
        record_from_dict,
    )

    for line in iter_shard_lines(Path(path)):
        stripped = line.strip()
        if not stripped:
            continue
        data = parse_record_line(stripped)
        if remapper is not None:
            remap_record_dict(data, remapper, counts)
        yield record_from_dict(data)


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once through the pool initializer."""

    records_dir: str
    held_out_names: frozenset[str]
    held_out_script_files: frozenset[str]
    text_cap: int
    seed: int
    #: ``--max-events-per-record``, passed through to ``quality_defect``. No
    #: default: the survey counts what the write pass will keep, so a survey
    #: run under a different rule than the write pass reports availability for
    #: records that are about to be refused.
    max_events: int
    #: Tree name -> converted root, so a worker can build its own
    #: SidecarCache: the refusal rule needs the sidecar's answer per key, and
    #: a cache does not pickle.
    sidecar_roots: dict[str, str] = field(default_factory=dict)
    #: The old-token-key remap, or None when it is off (FR-151). A plain
    #: object of dicts and frozensets, so it pickles into every worker with
    #: the rest of this config rather than being rebuilt per shard.
    remapper: TokenKeyRemapper | None = None
    #: ``--game-disjoint-keywords`` in gate 2's spelling (``first_strike``):
    #: a game holding a combat record gate 2 qualifies for one of these is
    #: placed at the keyword threshold (FR-045).
    game_disjoint_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    """One selection cell of FR-049: class, legality half, family, signature.

    ``half`` is set on ``playability-legality`` only (``real`` or
    ``what-if``), ``signature`` on the four classes FR-049 step 2 splits by
    outcome. A frozen dataclass rather than a joined string, so it pickles
    between processes and sorts the same way in all of them.
    """

    klass: str
    half: str = ""
    family: str = ""
    signature: str = ""


#: The two halves of the legality class (FR-049 step 3). A gen-1 record, whose
#: ``what_if`` is unknown, is never a real decision (FR-033).
REAL, WHAT_IF = "real", "what-if"


def selection_half(record: EffectRecord, klass: str) -> str:
    """``real`` / ``what-if`` on a legality record, empty on every other."""
    from effects.domain.effect_model import CLASS_PLAYABILITY_LEGALITY

    if klass != CLASS_PLAYABILITY_LEGALITY:
        return ""
    return REAL if record.what_if is False else WHAT_IF


def cell_of(record: EffectRecord, sidecars, resolver) -> Cell:
    """The cell a record is selected in. Both passes call this one function.

    ``random_seat`` is deliberately not a component (FR-050): off-policy
    records compete for the same share as on-policy ones, and the manifest
    reports how many of each were written instead.
    """
    from effects.application.train_effect_model import sampling_class
    from effects.domain.rule_families import rule_family

    klass = sampling_class(record)
    return Cell(
        klass=klass,
        half=selection_half(record, klass),
        family=rule_family(record, sidecars, resolver=resolver),
        signature=(
            signature_label(outcome_signature(record))
            if klass in SIGNATURE_CLASSES else ""
        ),
    )


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution. Plain data, so it pickles back cheaply."""

    name: str
    size: int
    records: int
    key_games: dict[str, set[int]] = field(default_factory=dict)
    key_records: Counter[str] = field(default_factory=Counter)
    class_records: Counter[str] = field(default_factory=Counter)
    #: ``(cell, rendered key) -> records``: what one text can supply to one
    #: cell, once the main process has folded keys to texts (FR-049).
    cell_key_records: Counter[tuple[Cell, str]] = field(default_factory=Counter)
    #: ``(cell, rendered key) ->`` its ``--text-cap`` smallest record hashes.
    #: Enough to place any quota a text can be given, since no quota exceeds
    #: the cap, and bounded where every hash would not be.
    cell_key_hashes: dict[tuple[Cell, str], tuple[int, ...]] = field(default_factory=dict)
    #: Records with no acting key, per cell -- combat and legality, which have
    #: no text to cap and are sampled by hash rate instead.
    cell_keyless: Counter[Cell] = field(default_factory=Counter)
    #: Games holding a combat gate 2 qualifies for a listed keyword (FR-045).
    keyword_combat_games: set[str] = field(default_factory=set)
    #: Each key's game ids as strings, for keys carried by a held-out game.
    #: The card-disjoint cap admits whole games (FR-142) and so needs the ids
    #: themselves, not the hashed cardinality rarity counts with.
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)
    held_out_games: set[str] = field(default_factory=set)
    games: set[str] = field(default_factory=set)
    #: Records this shard's survey refused, by reason (FR-148). Counted
    #: rather than silently skipped: a corpus that is a third junk should say
    #: so in the manifest rather than in nothing.
    quality_dropped: Counter[str] = field(default_factory=Counter)
    #: The ``no-acting-text`` refusals alone, by the acting key's script file.
    #: Per script, so a converter regression that drops a whole card family's
    #: keys shows up in the build output rather than as one larger number.
    textless_scripts: Counter[str] = field(default_factory=Counter)
    #: Old token keys this shard's read rewrote, and the ones it refused to
    #: guess at (FR-151).
    remap: RemapCounts = field(default_factory=RemapCounts)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds."""

    shards: tuple[SourceShard, ...]
    records: int
    key_games: dict[str, set[int]]
    key_records: Counter[str]
    class_records: Counter[str]
    cell_key_records: Counter[tuple[Cell, str]]
    cell_key_heaps: dict[tuple[Cell, str], CapHeap]
    cell_keyless: Counter[Cell]
    keyword_combat_games: frozenset[str]
    held_out_text_games: dict[str, set[str]]
    held_out_games: frozenset[str]
    games: frozenset[str]
    quality_dropped: Counter[str] = field(default_factory=Counter)
    #: The survey's ``no-acting-text`` refusals by acting script file. The
    #: write pass keeps its own, and the manifest records that one.
    textless_scripts: Counter[str] = field(default_factory=Counter)
    #: Distinct games per top-level source directory, and how many of those
    #: name a held-out card. What FR-149's report is computed from: a
    #: directory of depleted shards whose held-out count is not zero is a leak
    #: in collection, and nothing else in the build would notice.
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: The whole corpus's remap counts, summed over the shards and over
    #: **every record read** -- including the ones the write pass goes on to
    #: refuse or drop. The write pass counts written records only, so its
    #: figures are a subset of these; one exceeding the survey's means the
    #: corpus changed between the two passes.
    remap: RemapCounts = field(default_factory=RemapCounts)


_CONFIG: SurveyConfig | None = None

#: One ``SidecarCache`` per worker process, built on first use. A cache holds
#: open-ended per-card state and does not pickle, so it cannot ride in the
#: config; and building one per shard would re-read tens of thousands of
#: sidecars per shard instead of once per process.
_SIDECARS: SidecarCache | None = None


#: The combat keyword resolver over that cache, for the same reason: it
#: memoizes per provenance key, and a corpus repeats the same cards millions
#: of times.
_RESOLVER = None


def _worker_sidecars(roots: dict[str, str]) -> SidecarCache:
    """One cache per worker process, built on first use from the config's roots."""
    global _SIDECARS, _RESOLVER
    if _SIDECARS is None:
        from effects.infrastructure.sidecar_io import SidecarCache

        _SIDECARS = SidecarCache({name: Path(root) for name, root in roots.items()})
        _RESOLVER = None
    return _SIDECARS


def _worker_resolver(roots: dict[str, str]):
    """The per-process keyword resolver over :func:`_worker_sidecars`."""
    global _RESOLVER
    sidecars = _worker_sidecars(roots)
    if _RESOLVER is None:
        from effects.domain.damage_step_keywords import KeywordResolver

        _RESOLVER = KeywordResolver(sidecars)
    return _RESOLVER


def _heap_cap(text_cap: int) -> int:
    """How many smallest hashes the survey keeps per cell and key.

    ``--text-cap 0`` is no cap, so a text's quota is bounded by its repeats
    alone and every hash may be needed.
    """
    import sys

    return text_cap if text_cap > 0 else sys.maxsize


def _refusal(record: EffectRecord, *, max_events: int, roots: dict[str, str]) -> str | None:
    """The one refusal rule both passes apply, so they cannot drift apart.

    ``build()`` warns when the survey and the write pass disagree about which
    records are refused, which only reads as "the corpus changed between the
    passes" while the rule itself is the same object in both.
    """
    defect = quality_defect(record, max_events=max_events)
    if defect is not None:
        return defect
    return acting_text_defect(record, _worker_sidecars(roots).resolution_of)


def init_survey_worker(config: SurveyConfig) -> None:
    """Pool initializer: hand every worker the config once, not per shard.

    Clears the sidecar cache too: a pool process runs this once, but a
    ``workers=1`` run calls it in the main process for every build, and a
    cache held over from a previous build would answer for another corpus's
    converted tree.
    """
    global _CONFIG, _SIDECARS, _RESOLVER
    _CONFIG = config
    _SIDECARS = None
    _RESOLVER = None


def survey_shard(relative: str) -> ShardSurvey:
    """Survey one shard. Module-level so a process pool can pickle it."""
    from effects.application.train_effect_model import (
        HeldOutCards,
        record_names_held_out_card,
    )
    from effects.domain.damage_step_keywords import (
        KEYWORDS_BY_NAME,
        qualifying_observations,
    )

    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    held_out = HeldOutCards(
        names=config.held_out_names, script_files=config.held_out_script_files,
    )
    path = Path(config.records_dir) / relative
    out = ShardSurvey(name=relative, size=path.stat().st_size, records=0)
    heaps: dict[tuple[Cell, str], CapHeap] = {}
    heap_cap = _heap_cap(config.text_cap)
    rows = tuple(
        KEYWORDS_BY_NAME[name] for name in config.game_disjoint_keywords
        if name in KEYWORDS_BY_NAME
    )
    sidecars = _worker_sidecars(config.sidecar_roots)
    resolver = _worker_resolver(config.sidecar_roots)

    for record in read_shard_remapped(path, config.remapper, out.remap):
        out.records += 1
        # The keyword threshold is a property of the game (FR-046), so it is
        # read above the quality check, like the held-out test below: a game
        # does not stop holding a combat because one record was refused.
        if (
            rows and record.kind is RecordKind.COMBAT
            and record.game_id not in out.keyword_combat_games
            and qualifying_observations(record, rows, resolver)
        ):
            out.keyword_combat_games.add(record.game_id)
        # Above the quality check on purpose: a game that played a held-out
        # card played it whether or not the record saying so survives the
        # rules. Routed on the surviving records alone, a game whose only
        # held-out mention sits on a refused record becomes a training game
        # and the card-disjoint split stops being card-disjoint. ``games`` is
        # counted below the check instead, because that one is a count of the
        # records the build will actually write.
        held = record_names_held_out_card(record, held_out)
        if held:
            out.held_out_games.add(record.game_id)
        defect = _refusal(
            record, max_events=config.max_events, roots=config.sidecar_roots,
        )
        if defect is not None:
            out.quality_dropped[defect] += 1
            if defect == NO_ACTING_TEXT:
                out.textless_scripts[record.ability[0].script_file] += 1
            continue
        out.games.add(record.game_id)
        cell = cell_of(record, sidecars, resolver)
        out.class_records[cell.klass] += 1
        key = ability_key(record)
        if key is None:
            out.cell_keyless[cell] += 1
            continue
        out.key_records[key] += 1
        out.cell_key_records[cell, key] += 1
        out.key_games.setdefault(key, set()).add(game_hash(record.game_id))
        if held:
            out.held_out_text_games.setdefault(key, set()).add(record.game_id)
        heap = heaps.get((cell, key))
        if heap is None:
            heap = heaps[cell, key] = CapHeap(heap_cap)
        heap.offer(record_hash(record.record_id, seed=config.seed))

    out.cell_key_hashes = {slot: heap.values() for slot, heap in heaps.items()}
    return out


def source_of(relative: str) -> str:
    """The top-level directory a raw shard sits in: ``depleted``, ``full-strength``…

    ``"."`` for a shard at the root. What FR-149 reports held-out games
    against: a collection run is a directory, and a leak is a directory that
    should hold none.
    """
    return relative.split("/", 1)[0] if "/" in relative else "."


def merge_surveys(parts: Iterable[ShardSurvey]) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    cap = _heap_cap(config.text_cap)
    shards: list[SourceShard] = []
    records = 0
    key_games: dict[str, set[int]] = defaultdict(set)
    key_records: Counter[str] = Counter()
    class_records: Counter[str] = Counter()
    cell_key_records: Counter[tuple[Cell, str]] = Counter()
    cell_key_heaps: dict[tuple[Cell, str], CapHeap] = {}
    cell_keyless: Counter[Cell] = Counter()
    keyword_combat_games: set[str] = set()
    held_out_games: set[str] = set()
    games: set[str] = set()

    held_out_text_games: dict[str, set[str]] = defaultdict(set)
    quality_dropped: Counter[str] = Counter()
    textless_scripts: Counter[str] = Counter()
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)
    remap = RemapCounts()

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        records += part.records
        key_records.update(part.key_records)
        class_records.update(part.class_records)
        cell_key_records.update(part.cell_key_records)
        cell_keyless.update(part.cell_keyless)
        keyword_combat_games |= part.keyword_combat_games
        held_out_games |= part.held_out_games
        games |= part.games
        quality_dropped.update(part.quality_dropped)
        textless_scripts.update(part.textless_scripts)
        remap.merge(part.remap)
        source = source_of(part.name)
        games_by_source[source] |= part.games
        held_by_source[source] |= part.held_out_games
        for key, ids in part.held_out_text_games.items():
            held_out_text_games[key] |= ids
        for key, hashed in part.key_games.items():
            key_games[key] |= hashed
        for slot, values in part.cell_key_hashes.items():
            heap = cell_key_heaps.get(slot)
            if heap is None:
                heap = cell_key_heaps[slot] = CapHeap(cap)
            for value in values:
                heap.offer(value)

    return Survey(
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        records=records,
        key_games=dict(key_games),
        key_records=key_records,
        class_records=class_records,
        cell_key_records=cell_key_records,
        cell_key_heaps=cell_key_heaps,
        cell_keyless=cell_keyless,
        keyword_combat_games=frozenset(keyword_combat_games),
        held_out_text_games=dict(held_out_text_games),
        held_out_games=frozenset(held_out_games),
        games=frozenset(games),
        quality_dropped=quality_dropped,
        textless_scripts=textless_scripts,
        games_by_source={s: len(g) for s, g in games_by_source.items()},
        held_out_games_by_source={s: len(g) for s, g in held_by_source.items()},
        remap=remap,
    )


def progress_line(*, done: int, total: int, elapsed: float) -> str:
    """``done of total`` with the rate and the time left.

    A bare count says the run is alive. It does not say whether the rest is ten
    minutes away or two hours, and on a pass over thousands of shards that is
    the question actually being asked.
    """
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate if rate > 0 else 0.0
    if rate <= 0:
        return f"{done} of {total} shard(s)"
    return (
        f"{done} of {total} shard(s), {rate * 60:.0f}/min, "
        f"~{left / 60:.0f} min left"
    )


def shard_names(records_dir: Path) -> list[str]:
    """Every shard under the corpus root, as posix paths relative to it."""
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return [shard.relative_to(root).as_posix() for shard in iter_shards(root)]


def run_survey(
    records_dir: Path,
    *,
    config: SurveyConfig,
    workers: int | None = None,
    progress_every: int = 50,
) -> Survey:
    """Survey every shard, in parallel, reporting progress as it goes."""
    root = Path(records_dir)
    names = shard_names(root)
    if not names:
        raise BuildCorpusError(f"{root}: no shards to build a corpus from")
    logger.info("Surveying %d shard(s) under %s", len(names), root)

    started = time.monotonic()
    init_survey_worker(config)          # so a workers=1 run needs no pool
    if workers is not None and workers <= 1:
        parts = []
        for index, name in enumerate(names, start=1):
            parts.append(survey_shard(name))
            if index % progress_every == 0:
                logger.info("Surveyed %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
        return merge_surveys(parts)

    parts = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_survey_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(pool.map(survey_shard, names, chunksize=1), 1):
            parts.append(part)
            if index % progress_every == 0:
                logger.info("Surveyed %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
    return merge_surveys(parts)


#: FR-045's default keyword list, in the spelling an operator types.
DEFAULT_GAME_DISJOINT_KEYWORDS: tuple[str, ...] = (
    "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
)
_DAMAGE_STEP_NAMES = frozenset({
    "first_strike", "double_strike", "deathtouch", "lifelink", "trample",
    "indestructible", "wither", "infect",
})


def gate_keyword(name: str) -> str:
    """An operator's keyword spelling in gate 2's (``first strike`` → ``first_strike``)."""
    return name.strip().lower().replace(" ", "_")


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus)."""

    records_dir: Path
    output: Path = Path("output/effects/corpus")
    cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")
    #: Forge's own raw token scripts, read to resolve the old collector's
    #: token keys (FR-151). The *raw* tree, not the converted one: the remap
    #: tells same-name scripts apart by their ``Colors:``, ``Types:`` and
    #: ``PT:`` lines, which conversion does not preserve.
    forge_tokenscripts: Path | None = Path("../forge/forge-gui/res/tokenscripts")
    remap_token_keys: bool = True
    vocab_path: str = "models/effects/vocab.txt"
    variant_scripts: str | None = None
    holdout_permille: int = 20
    holdout_max_carriers: int = 8
    #: ``template`` (gen-2) or ``text`` (feature 023's rule, FR-041).
    holdout_unit: str = "template"
    #: Most written records per ability text **per cell**, repeats included.
    text_cap: int = 200
    #: Most copies of one record toward a short family's share (FR-049).
    reuse_cap: int = 4
    class_mix: dict[str, float] | None = None
    training_records: int = 0
    #: The per-game hash threshold of the game-disjoint stratum (FR-044).
    game_disjoint_share: float = 0.01
    #: The threshold for a game holding a qualifying combat for one of
    #: ``game_disjoint_keywords`` (FR-045).
    game_disjoint_keyword_share: float = 0.15
    game_disjoint_keywords: tuple[str, ...] = DEFAULT_GAME_DISJOINT_KEYWORDS
    #: The keyword-definitions table the tokenizer expands with; FR-015's check
    #: reads it. None, or a path that does not exist, skips the check: nothing
    #: can expand without a table.
    keyword_definitions: Path | None = Path("output/effects/keyword-definitions.json")
    card_disjoint_text_cap: int = 50
    shard_records: int = 2000
    max_events: int = 64
    validation_sample: int = 2048
    seed: int = 42
    workers: int | None = None
    verify: bool = False

    def __post_init__(self) -> None:
        """Refuse a negative count, and say what zero means for each.

        Zero is a real setting for all seven and means something different in
        each, which is why it is spelled out rather than rejected along with
        the negatives:

        - ``--text-cap 0`` — no per-text cap; a text's quota in a cell is bounded
          by ``--reuse-cap`` repeats alone.
        - ``--card-disjoint-text-cap 0`` — no cap on the card-disjoint
          stratum; every held-out game carrying a held-out text is admitted.
        - ``--game-disjoint-share 0`` — no game-disjoint stratum, unless the
          keyword share places some games.
        - ``--training-records 0`` — no ceiling beyond ``--text-cap``.
        - ``--shard-records 0`` — no repacking; one output shard per source
          shard, which is what the write pass did before repacking existed
          (``repack_shards`` reads a non-positive size this way).
        - ``--max-events-per-record 0`` — no event-flood rule; every record's
          event count is accepted (``quality_defect`` reads a non-positive
          maximum this way, the same reading ``--text-cap`` gives zero). The
          other two quality rules have no number and stay on.
        - ``--validation-sample 0`` — no fixed validation sample; nothing is
          written under ``validation/samples/``.

        A negative is none of those and has no reading at all: it would make
        ``min(cap, total)`` negative and ``random.sample`` raise several
        thousand records into the build.
        """
        for flag, value in (
            ("--text-cap", self.text_cap),
            ("--card-disjoint-text-cap", self.card_disjoint_text_cap),
            ("--training-records", self.training_records),
            ("--shard-records", self.shard_records),
            ("--max-events-per-record", self.max_events),
            ("--validation-sample", self.validation_sample),
        ):
            if value < 0:
                raise BuildCorpusError(
                    f"{flag} is {value}; it counts things and cannot be "
                    "negative. Zero is allowed and means no cap, no ceiling, "
                    "or an empty stratum depending on the flag — see --help."
                )
        if self.reuse_cap < 1:
            raise BuildCorpusError(
                f"--reuse-cap is {self.reuse_cap}; it is the most copies of one "
                "record a short family may write, and 1 already means no repeats."
            )
        for flag, share in (
            ("--game-disjoint-share", self.game_disjoint_share),
            ("--game-disjoint-keyword-share", self.game_disjoint_keyword_share),
        ):
            if not 0.0 <= share <= 1.0:
                raise BuildCorpusError(f"{flag} is {share}; a share is between 0 and 1.")
        if self.holdout_unit not in ("template", "text"):
            raise BuildCorpusError(
                f"--holdout-unit is {self.holdout_unit!r}; expected template or text."
            )
        unknown = [
            name for name in self.game_disjoint_keywords
            if gate_keyword(name) not in _DAMAGE_STEP_NAMES
        ]
        if unknown:
            raise BuildCorpusError(
                f"--game-disjoint-keywords names {', '.join(unknown)}, which gate 2 "
                f"does not score; it scores {', '.join(sorted(_DAMAGE_STEP_NAMES))}."
            )

    def mix(self) -> dict[str, float]:
        from effects.application.train_effect_model import DEFAULT_KIND_MIX

        return dict(self.class_mix or DEFAULT_KIND_MIX)


@dataclass(frozen=True, slots=True)
class FamilyShare:
    """One family's place in its class budget, as decided (FR-053).

    ``share`` is the family's equal split of its class (or legality half)
    budget, before redistribution — the share it would get if every family
    could fill it — so ``max(0, share - written)`` is its shortfall, the number
    the operator sends back to the coverage and variant collectors.
    """

    available: int
    share: int
    planned: int


@dataclass(frozen=True, slots=True)
class Decisions:
    """Everything the write pass needs, and nothing it has to look up.

    ``plans`` is keyed by cell and rendered provenance key rather than by
    ability text, so a worker applies it without folding keys; the fold from
    key to text happened here, once. A record whose key folds to no text, and
    a record with no key at all, takes its cell's ``keyless_plans`` entry.
    """

    plans: dict[tuple[Cell, str], CopyPlan]
    keyless_plans: dict[Cell, CopyPlan]
    class_targets: dict[str, int]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    #: Game-disjoint games placed only because they hold a qualifying combat
    #: for a listed keyword (FR-053).
    keyword_threshold_games: frozenset[str]
    rarity: dict[str, int]
    shortfall: dict[str, int]
    #: Distinct records each class can supply under the per-cell text cap,
    #: with no repeats: what the class mixture is divided over.
    capped_class_records: dict[str, int]
    key_text: dict[str, str]
    #: Rendered survey keys whose ability text is held out. What routes the
    #: gate-one slice: a card-disjoint resolution record acting on one of
    #: these is the unique-text stratum gate 1's three margins are measured
    #: on, and picking it out at write time is what saves the evaluator a
    #: second pass over the whole stratum.
    gate_one_keys: frozenset[str]
    #: ``class -> family label -> FamilyShare``.
    families: dict[str, dict[str, FamilyShare]] = field(default_factory=dict)
    #: Rendered key -> the held-out text it resolves to.
    held_out_text_of_key: dict[str, str] = field(default_factory=dict)


def family_label(cell: Cell) -> str:
    """A family's name in the manifest: prefixed by its half on a legality cell."""
    return f"{cell.half}:{cell.family}" if cell.half else cell.family


def _text_of_rendered_key(
    rendered: str, sidecars: SidecarCache, surface: str,
) -> str | None:
    """The text a survey key folds to, through the trainer's own definition."""
    from effects.application.train_effect_model import text_for_key

    for key in parse_ability_key(rendered):
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


def _held_out_text_of_key(
    survey: Survey, sidecars: SidecarCache, held_out_texts: frozenset[str],
) -> dict[str, str]:
    """Each survey key that resolves to a held-out ability text, and the text.

    Resolved on the **script** surface whatever ``--vocab-path`` says, because
    that is the surface the holdout itself is keyed on: ``select_holdout``
    hashes a sidecar line's ``script_text``, so a prose-surface fold would
    compare a prose string against a set of script strings and match nothing —
    silently, since an empty match looks exactly like a corpus with no
    held-out ability in it. Normalized the same way for the same reason.
    """
    from effects.domain.ability_encoder import SURFACE_SCRIPT
    from effects.domain.text_holdout import normalize_script_text

    out: dict[str, str] = {}
    for key in survey.held_out_text_games:
        script = _text_of_rendered_key(key, sidecars, SURFACE_SCRIPT)
        if script is None:
            continue
        normalized = normalize_script_text(script)
        if normalized in held_out_texts:
            out[key] = normalized
    return out


def _card_disjoint_games(
    survey: Survey, held_out_text_of_key: dict[str, str], *, cap: int, seed: int,
) -> frozenset[str]:
    """Held-out games admitted while every held-out text they carry is under cap.

    Games rather than records (FR-142, FR-136): a stratum built by dropping
    records inside a game would separate a probe from the combat record its
    ``mirror_of`` names, and the evaluator would score that keyword on nothing
    while reporting it as merely under-sampled.

    Two rules make the cap bind, and both are FR-142's "per **held-out**
    ability text" read literally.

    Only a held-out text is tallied. A real game carries tens of distinct
    ability texts, nearly all of them ordinary, so a tally over every text a
    game carries counted mostly things the cap is not about — and the
    held-out texts are the only ones gate 1 averages over.

    And a game is admitted only while **every** held-out text it carries is
    still under the cap, not merely one of them. Admitting on one was the
    defect: a game's own private texts sit at zero forever, so there was
    always some text under cap and the cap never declined a game. The cost of
    the strict rule is that a game pairing a saturated text with an unsaturated
    one is declined, which can leave the second short; the cap is a ceiling on
    how far any one held-out text may dominate the stratum, and a rule that
    can be pushed past its ceiling is not one.

    A game carrying no held-out text is not admitted at all. It holds nothing
    gate 1 measures, and it is withheld from training either way — a held-out
    game the cap declines is dropped, never trained on.

    Walked in a seeded order so two builds of one corpus admit the same games.
    ``cap <= 0`` means no cap, matching ``--text-cap``'s own zero.
    """
    import random

    text_games: dict[str, set[str]] = defaultdict(set)
    for key, text in held_out_text_of_key.items():
        text_games[text] |= survey.held_out_text_games[key]

    order = sorted(survey.held_out_games)
    random.Random(seed).shuffle(order)
    games_text: dict[str, set[str]] = defaultdict(set)
    for text, games in text_games.items():
        for game in games:
            games_text[game].add(text)

    taken: Counter[str] = Counter()
    admitted: set[str] = set()
    for game in order:
        texts = games_text.get(game, set())
        if not texts:
            continue
        if cap > 0 and any(taken[text] >= cap for text in texts):
            continue
        admitted.add(game)
        for text in texts:
            taken[text] += 1
    return frozenset(admitted)


_KEYLESS = ""


def _cell_members(
    survey: Survey, key_text: dict[str, str], *, heap_cap: int,
) -> tuple[dict[Cell, dict[str, list]], dict[tuple[Cell, str], list[str]]]:
    """``cell -> member -> [distinct records, CapHeap]`` and each member's keys.

    A member is an ability text, or ``_KEYLESS`` for the records whose key folds
    to no text or that carry no key at all.
    """
    members: dict[Cell, dict[str, list]] = defaultdict(dict)
    keys_of: dict[tuple[Cell, str], list[str]] = defaultdict(list)
    for (cell, key), count in survey.cell_key_records.items():
        text = key_text.get(key)
        if text is None:
            entry = members[cell].setdefault(_KEYLESS, [0, None])
            entry[0] += count
            continue
        entry = members[cell].setdefault(text, [0, CapHeap(heap_cap)])
        entry[0] += count
        entry[1].merge(survey.cell_key_heaps[cell, key])
        keys_of[cell, text].append(key)
    for cell, count in survey.cell_keyless.items():
        entry = members[cell].setdefault(_KEYLESS, [0, None])
        entry[0] += count
    return members, keys_of


def _distinct_capacity(member: str, distinct: int, text_cap: int) -> int:
    """Records a member can supply once each: the text cap, or all of them."""
    if member == _KEYLESS or text_cap <= 0:
        return distinct
    return min(distinct, text_cap)


def _capacity(member: str, distinct: int, *, text_cap: int, reuse_cap: int) -> int:
    """Records a member can supply with repeats (FR-049)."""
    from effects.domain.corpus_curation import text_capacity

    return text_capacity(
        distinct, text_cap=0 if member == _KEYLESS else text_cap, reuse_cap=reuse_cap,
    )


def _split_cell(
    budget: int, entries: dict[str, list], *, text_cap: int, reuse_cap: int,
) -> dict[str, int]:
    """A cell's budget over its members: distinct records first, then repeats.

    The same equal split as every level above, in two rounds: first over what
    each member supplies once, so no record is repeated while another goes
    unused, then over the repeats the remainder still needs — which is the
    only case FR-049 repeats at all, a family short of its share.
    """
    from effects.domain.budget_allocation import allocate

    once = {
        member: _distinct_capacity(member, entry[0], text_cap) for member, entry in entries.items()
    }
    first = allocate(budget, once)
    left = budget - sum(first.values())
    if left <= 0:
        return first
    more = {
        member: _capacity(member, entry[0], text_cap=text_cap, reuse_cap=reuse_cap) - first[member]
        for member, entry in entries.items()
    }
    second = allocate(left, more)
    return {member: first[member] + second.get(member, 0) for member in entries}


def _plan_classes(
    members: dict[Cell, dict[str, list]],
    keys_of: dict[tuple[Cell, str], list[str]],
    targets: dict[str, int],
    config: BuildCorpusConfig,
) -> tuple[
    dict[tuple[Cell, str], CopyPlan], dict[Cell, CopyPlan], dict[str, dict[str, FamilyShare]],
]:
    """FR-049 steps 1–4: class budget → half → family → signature → member.

    Every level is the same equal split with redistribution (``allocate``),
    each member capped by what it can supply with repeats, so a short family
    is written at capacity and what it leaves goes to the families that still
    have room. Within a cell the budget reaches each ability text through
    :func:`_split_cell`, and the text's quota becomes a copy plan over its
    smallest record hashes.
    """
    from effects.domain.budget_allocation import allocate
    from effects.domain.corpus_curation import copy_plan, rate_plan

    def capacity(cell: Cell) -> int:
        return sum(
            _capacity(member, entry[0], text_cap=config.text_cap, reuse_cap=config.reuse_cap)
            for member, entry in members[cell].items()
        )

    def distinct(cell: Cell) -> int:
        return sum(entry[0] for entry in members[cell].values())

    tree: dict[str, dict[str, dict[str, dict[Cell, int]]]] = {}
    for cell in members:
        tree.setdefault(cell.klass, {}).setdefault(cell.half, {}).setdefault(
            cell.family, {},
        )[cell] = capacity(cell)

    plans: dict[tuple[Cell, str], CopyPlan] = {}
    keyless: dict[Cell, CopyPlan] = {}
    families: dict[str, dict[str, FamilyShare]] = {}
    for klass, halves in sorted(tree.items()):
        budget = targets.get(klass, 0)
        half_budgets = allocate(budget, {
            half: sum(sum(cells.values()) for cells in fams.values())
            for half, fams in halves.items()
        })
        for half, fams in sorted(halves.items()):
            fam_budgets = allocate(half_budgets[half], {
                family: sum(cells.values()) for family, cells in fams.items()
            })
            # The share a family would get if every member at every level could
            # fill its equal split: the class budget halved for the legality
            # halves, then split over the half's families. A family written
            # below it is short, whatever the redistribution then did.
            fair = budget // len(halves) // len(fams) if fams else 0
            for family, cells in sorted(fams.items()):
                cell_budgets = allocate(fam_budgets[family], cells)
                label = family_label(Cell(klass, half, family))
                families.setdefault(klass, {})[label] = FamilyShare(
                    available=sum(distinct(cell) for cell in cells),
                    share=fair,
                    planned=sum(cell_budgets.values()),
                )
                for cell, cell_budget in sorted(cell_budgets.items()):
                    quotas = _split_cell(
                        cell_budget, members[cell],
                        text_cap=config.text_cap, reuse_cap=config.reuse_cap,
                    )
                    for member, quota in quotas.items():
                        count, heap = members[cell][member]
                        if member == _KEYLESS:
                            keyless[cell] = rate_plan(count, quota)
                            continue
                        plan = copy_plan(heap.values(), count, quota)
                        for key in keys_of[cell, member]:
                            plans[cell, key] = plan
    return plans, keyless, families


def _keyword_misfires(
    key_text: dict[str, str], sidecars: SidecarCache, definitions: Path | None,
) -> list[str]:
    """Texts a display-name keyword match would expand on a non-keyword line.

    FR-015: the tokenizer recognises a keyword line by the text before its
    first colon matching a definition, and the match has to land on a line
    whose ``script_api_type`` is ``Keyword``. One that lands anywhere else
    would replace a script with a reminder template the line does not carry,
    and train the encoder on it; the build refuses instead.
    """
    if definitions is None or not Path(definitions).exists():
        return []
    from effects.application.extract_keyword_definitions import load_keyword_definitions
    from effects.domain.ability_tokenizer import display_name_of
    from effects.domain.rule_families import KEYWORD_API_TYPE

    names = {name.lower() for name in load_keyword_definitions(Path(definitions))}
    found: list[str] = []
    for key in key_text:
        for parsed in parse_ability_key(key):
            try:
                line = sidecars.line_for(parsed)
            except KeyError:
                continue
            if line is None or not line.script_text:
                continue
            if line.script_api_type == KEYWORD_API_TYPE:
                continue
            if display_name_of(line.script_text).lower() in names:
                found.append(f"{parsed.script_file}: {line.script_text!r}")
    return sorted(set(found))


def decide(
    survey: Survey,
    *,
    sidecars: SidecarCache,
    surface: str,
    held_out_texts: frozenset[str],
    config: BuildCorpusConfig,
) -> Decisions:
    """Fold keys to texts, split the games, and plan every record's copies.

    Args:
        held_out_texts: the normalized script texts the holdout names, which
            is the unit ``--card-disjoint-text-cap`` counts (FR-142).
    """
    from effects.domain.corpus_curation import (
        class_targets,
        game_disjoint_draw,
        in_game_disjoint_stratum,
    )

    heap_cap = _heap_cap(config.text_cap)
    for slot, heap in survey.cell_key_heaps.items():
        if heap.cap != heap_cap:
            raise ValueError(
                f"survey cell {slot!r} was surveyed keeping {heap.cap} hashes, "
                f"but config.text_cap {config.text_cap} keeps {heap_cap}. "
                "decide() must be run against the Survey that this same "
                "--text-cap produced, or the copy plans it writes would disagree "
                "with what the manifest is about to record."
            )

    key_text: dict[str, str] = {}
    text_games: dict[str, set[int]] = defaultdict(set)
    for key in survey.key_records:
        text = _text_of_rendered_key(key, sidecars, surface)
        if text is None:
            # No sidecar can read this key, so it is not a text and cannot be
            # capped against one. Its records stay, sampled with the cell's
            # keyless records: dropping them would remove a card the converted
            # corpus never held rather than trimming a head.
            continue
        key_text[key] = text
        text_games[text] |= survey.key_games.get(key, set())

    misfires = _keyword_misfires(key_text, sidecars, config.keyword_definitions)
    if misfires:
        raise BuildCorpusError(
            f"{len(misfires)} non-keyword line(s) would expand as a keyword by "
            "display name (FR-015): the text before the first colon matches a "
            "keyword definition while the line's script_api_type is not "
            "Keyword. First: " + "; ".join(misfires[:5])
        )

    rarity = {text: len(games) for text, games in text_games.items()}

    held_out_text_of_key = _held_out_text_of_key(survey, sidecars, held_out_texts)
    # Only ``survey.held_out_text_games`` is walked, which is keyed on records
    # of held-out games -- exactly the population the gate-one slice draws
    # from, since the slice is a subset of the card-disjoint stratum.
    gate_one_keys = frozenset(held_out_text_of_key)
    card_disjoint = _card_disjoint_games(
        survey,
        held_out_text_of_key,
        cap=config.card_disjoint_text_cap,
        seed=config.seed,
    )
    # Per game, never a sample of the corpus (FR-044–046): a rebuild over more
    # shards places every game it placed before.
    game_disjoint = frozenset(
        game for game in survey.games - survey.held_out_games
        if in_game_disjoint_stratum(
            game, False, game in survey.keyword_combat_games,
            config.game_disjoint_share, config.game_disjoint_keyword_share,
        )
    )
    keyword_threshold_games = frozenset(
        game for game in game_disjoint if game_disjoint_draw(game) >= config.game_disjoint_share
    )

    members, keys_of = _cell_members(survey, key_text, heap_cap=heap_cap)
    available: Counter[str] = Counter()
    for cell, entries in members.items():
        for member, entry in entries.items():
            available[cell.klass] += _distinct_capacity(member, entry[0], config.text_cap)
    try:
        targets, shortfall = class_targets(
            dict(available), config.mix(), ceiling=config.training_records,
        )
    except ValueError as exc:
        # `--class-mix` names classes this corpus has no records of, which is
        # the operator's flag against the operator's corpus, not a broken
        # invariant.
        raise BuildCorpusError(str(exc)) from exc
    plans, keyless_plans, families = _plan_classes(members, keys_of, targets, config)
    return Decisions(
        plans=plans,
        keyless_plans=keyless_plans,
        class_targets=targets,
        card_disjoint=card_disjoint,
        game_disjoint=game_disjoint,
        keyword_threshold_games=keyword_threshold_games,
        rarity=rarity,
        shortfall=shortfall,
        capped_class_records=dict(available),
        key_text=key_text,
        gate_one_keys=gate_one_keys,
        families=families,
        held_out_text_of_key=held_out_text_of_key,
    )


#: How much of one sampling class the quality rules may refuse before the
#: build stops instead of writing the dataset. A class refused past this is
#: not a dirty corpus: it is a rule that does not fit the kind, or a corpus
#: collected against an unpatched Forge, and either way the dataset silently
#: loses a whole class the trainer then reports as one the corpus happens not
#: to hold. The withdrawn unattributed rule refused 98.9% of combat records
#: exactly this way and nothing said so.
MAX_REFUSED_SHARE = 0.5

#: Every write-pass stratum, in report order. All but the last are the real
#: outputs -- shard directories a downstream reader loads, and what
#: FR-143/FR-145 mean by "each output" -- and ``OUTPUTS`` is exactly that
#: prefix, so "every output in OUTPUTS" is what the reports iterate;
#: "dropped-held-out" is accounting only, since nothing is written for it, so
#: it is counted in ``per_stratum`` but never carries a unique-text figure.
#: "gate-one" is a *slice* of "card-disjoint" rather than a fourth
#: destination: its records are written twice on purpose, so
#: the evaluator reads gate 1's unique-text stratum without re-filtering the
#: whole card-disjoint output, and its per-stratum count therefore overlaps
#: card-disjoint's rather than partitioning with it.
STRATA: tuple[str, ...] = (
    "training", "card-disjoint", "game-disjoint", "gate-one", "dropped-held-out",
)
OUTPUTS: tuple[str, ...] = STRATA[:-1]


@dataclass(frozen=True, slots=True)
class WriteConfig:
    """Where one shard's records go, and by what rule.

    The four output directories name **parts** directories rather than the
    strata themselves: a worker writes one part per source shard, and
    ``build()`` repacks the parts into uniform shards afterwards.
    """

    records_dir: str
    training_dir: str
    card_disjoint_dir: str
    game_disjoint_dir: str
    gate_one_dir: str
    gate_one_keys: frozenset[str]
    max_events: int
    plans: dict[tuple[Cell, str], CopyPlan]
    keyless_plans: dict[Cell, CopyPlan]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    held_out_games: frozenset[str]
    seed: int
    #: Tree name -> converted root, so a worker can build its own
    #: SidecarCache: the refusal rule needs the sidecar's answer per key, and
    #: a cache does not pickle.
    sidecar_roots: dict[str, str] = field(default_factory=dict)
    #: The same remapper the survey read through, so both passes see one
    #: corpus (FR-151); None when ``--no-remap-token-keys``.
    remapper: TokenKeyRemapper | None = None


@dataclass(slots=True)
class WriteResult:
    #: Records written to the training output per class, copies included.
    kept: Counter[str] = field(default_factory=Counter)
    #: Records a copy plan wrote zero times, per class.
    dropped_by_cap: Counter[str] = field(default_factory=Counter)
    read: Counter[str] = field(default_factory=Counter)
    kept_keys: dict[str, set[str]] = field(default_factory=dict)
    #: Records written to each output, keyed "training" / "card-disjoint" /
    #: "game-disjoint", plus "dropped-held-out" for held-out games the
    #: card-disjoint cap declined. FR-145 wants the report per stratum as well
    #: as per class, and a stratum nobody counted is a stratum nobody notices
    #: is empty.
    stratum: Counter[str] = field(default_factory=Counter)
    #: Ability keys admitted to each output in ``OUTPUTS``, never
    #: "dropped-held-out", which writes nothing -- the same role
    #: ``kept_keys`` plays per class, but per
    #: stratum instead: ``build()`` folds these through ``decisions.key_text``
    #: to get each output's unique-ability-text count (FR-143, FR-145).
    stratum_keys: dict[str, set[str]] = field(default_factory=dict)
    #: Records refused by ``effects.domain.record_quality``, by reason
    #: (FR-148). Counted before every routing decision, so a defective record
    #: reaches no output at all and belongs to no class's read count.
    quality_dropped: Counter[str] = field(default_factory=Counter)
    #: The same refusals by sampling class rather than by reason, which is the
    #: axis ``MAX_REFUSED_SHARE`` is judged on: a corpus can be a few percent
    #: junk overall and still have lost one whole class, and
    #: ``quality_dropped`` cannot tell the two apart.
    refused_by_class: Counter[str] = field(default_factory=Counter)
    #: The ``no-acting-text`` refusals alone, by the acting key's script file
    #: (FR-148). Per script, so a converter regression that drops a whole card
    #: family's keys shows up in the build output rather than as one larger
    #: number. This is the tally the manifest records.
    textless_scripts: Counter[str] = field(default_factory=Counter)
    #: Records the pass **kept** whose events name no producing clause
    #: (FR-148). Counted after the quality check, so it is a share of what the
    #: dataset actually holds rather than of what the shards held. Watched
    #: rather than refused: a clause the collector could not find on the
    #: acting chain does not make the outcome another ability's.
    unattributed: int = 0
    #: Old token keys rewritten, and the ones refused, **on written records**
    #: (FR-151) -- a record this pass refuses or drops contributes neither.
    #: The manifest records these rather than the survey's, since these are
    #: the keys the written dataset actually carries; the survey's own counts
    #: are over every record read and are kept for the cross-check.
    remap: RemapCounts = field(default_factory=RemapCounts)
    #: Training records written per ``(class, family label)``, copies
    #: included, and the distinct records behind them: the difference is the
    #: family's repeats (FR-053).
    family_written: Counter[tuple[str, str]] = field(default_factory=Counter)
    family_distinct: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: Training records per ``(class, family label, signature)``.
    signature_written: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    #: Training records per ``(class, on_policy | off_policy)``.
    policy: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: Training legality records per ``(subkind, real | what_if | unknown)``.
    legality: Counter[tuple[str, str]] = field(default_factory=Counter)


_WRITE: WriteConfig | None = None

#: The plan of a cell the decision gave nothing: written zero times.
_NO_COPIES = CopyPlan()


def init_write_worker(config: WriteConfig) -> None:
    """Pool initializer, and the same cache reset ``init_survey_worker`` makes.

    A ``workers=1`` run calls both in the one process, so the write pass would
    otherwise inherit whatever tree the survey — or an earlier build — read.
    """
    global _WRITE, _SIDECARS, _RESOLVER
    _WRITE = config
    _SIDECARS = None
    _RESOLVER = None


def _output_name(relative: str) -> str:
    """A flat, **injective** shard name for a source that may sit in a subdirectory.

    ``depleted/run.0-a.jsonl.gz`` and ``full-strength/run.0-a.jsonl.gz`` are
    different shards and must not write to one file — and neither may any other
    pair of distinct relative paths, because ``write_shard`` opens in truncate
    mode, so a collision is a silent overwrite in every output with no
    stable answer, under ``workers > 1``, as to which source survives.

    A plain ``"/" -> "__"`` substitution is not injective: ``"a_/b"`` and
    ``"a/_b"`` both become ``"a___b"``, because an original ``_`` and a
    doubled-up ``/`` separator are indistinguishable in the output. The fix is
    the standard one for building an injective string encoding out of a small
    alphabet of specials: escape the escape character *first*. Every ``_`` in
    the result is then the first character of a two-character escape --
    ``_u`` for an original ``_``, ``_s`` for an original ``/`` -- and never an
    untouched original character (every real ``_`` was already rewritten to
    ``_u`` before ``/`` is touched), so the two cases can never be confused
    and the mapping is one-to-one.
    """
    return relative.replace("_", "_u").replace("/", "_s")


def write_shard_pass(relative: str) -> WriteResult:
    """Filter one shard into its output parts. Module-level so it pickles."""
    from effects.application.train_effect_model import sampling_class
    from effects.infrastructure.record_io import write_shard

    config = _WRITE
    assert config is not None, "init_write_worker was not run"
    out = WriteResult()
    sidecars = _worker_sidecars(config.sidecar_roots)
    resolver = _worker_resolver(config.sidecar_roots)
    training: list = []
    card_disjoint: list = []
    game_disjoint: list = []
    gate_one: list = []

    # The remap's counts land here first and are folded into ``out.remap``
    # only for a record that reaches an output, so the manifest's figure is
    # what the written dataset carries rather than what the shards held
    # (FR-151). A record the quality check refuses, the per-text cap drops
    # or the held-out rule discards takes its remaps with it.
    pending = RemapCounts()

    for record in read_shard_remapped(
        Path(config.records_dir) / relative, config.remapper, pending,
    ):
        written = False
        try:
            name = sampling_class(record)
            # Computed once, up front: every branch below -- including the two
            # validation strata, which never used to look at it at all -- needs
            # it to track which ability texts that output actually holds.
            key = ability_key(record)
            # Before the read count, so a refused record is not counted as read
            # against a class whose availability the survey computed without it.
            defect = _refusal(
                record, max_events=config.max_events, roots=config.sidecar_roots,
            )
            if defect is not None:
                out.quality_dropped[defect] += 1
                out.refused_by_class[name] += 1
                if defect == NO_ACTING_TEXT:
                    out.textless_scripts[record.ability[0].script_file] += 1
                continue
            out.read[name] += 1
            if has_unattributed_events(record):
                out.unattributed += 1
            if record.game_id in config.card_disjoint:
                card_disjoint.append(record)
                written = True
                out.stratum["card-disjoint"] += 1
                if key is not None:
                    out.stratum_keys.setdefault("card-disjoint", set()).add(key)
                if (
                    record.kind is RecordKind.RESOLUTION
                    and key is not None and key in config.gate_one_keys
                ):
                    gate_one.append(record)
                    out.stratum["gate-one"] += 1
                    out.stratum_keys.setdefault("gate-one", set()).add(key)
                continue
            if record.game_id in config.game_disjoint:
                game_disjoint.append(record)
                written = True
                out.stratum["game-disjoint"] += 1
                if key is not None:
                    out.stratum_keys.setdefault("game-disjoint", set()).add(key)
                continue
            if record.game_id in config.held_out_games:
                # Held out but not admitted to the stratum: dropped, never trained
                # on (FR-088). Nothing is written for it, so it earns no entry in
                # stratum_keys -- only an output in OUTPUTS does.
                out.stratum["dropped-held-out"] += 1
                continue
            cell = cell_of(record, sidecars, resolver)
            plan = config.plans.get((cell, key)) if key is not None else None
            if plan is None:
                plan = config.keyless_plans.get(cell, _NO_COPIES)
            copies = plan.copies(record_hash(record.record_id, seed=config.seed))
            if copies <= 0:
                out.dropped_by_cap[name] += 1
                continue
            training.extend([record] * copies)
            written = True
            out.kept[name] += copies
            out.stratum["training"] += copies
            label = family_label(cell)
            out.family_written[name, label] += copies
            out.family_distinct[name, label] += 1
            out.signature_written[name, label, cell.signature] += copies
            out.policy[name, "off_policy" if record.random_seat else "on_policy"] += copies
            if cell.half:
                decision = (
                    "unknown" if record.what_if is None
                    else "what_if" if record.what_if else "real"
                )
                out.legality[record.subkind.value, decision] += copies
            if key is not None:
                out.kept_keys.setdefault(name, set()).add(key)
                out.stratum_keys.setdefault("training", set()).add(key)
        finally:
            if written:
                out.remap.merge(pending)
            pending.remapped = 0
            pending.ambiguous.clear()

    shard_name = _output_name(relative)
    # Only where there is something to write. A source shard contributes to one
    # stratum far more often than to three, so writing all three unconditionally
    # produced two empty files per source shard -- 2,103 files from 701 sources.
    # An empty validation shard is read and logged before the first training
    # step; an empty training shard drawn into an epoch forfeits its share of
    # the step budget, because `_train_on_shard` returns early on one.
    for directory, records in (
        (config.training_dir, training),
        (config.card_disjoint_dir, card_disjoint),
        (config.game_disjoint_dir, game_disjoint),
        (config.gate_one_dir, gate_one),
    ):
        if records:
            write_shard(Path(directory) / shard_name, records)
    return out


def run_write_pass(
    names: list[str], *, config: WriteConfig, workers: int | None,
    progress_every: int = 50,
) -> WriteResult:
    """Write every shard's output parts, in parallel, reporting progress."""
    started = time.monotonic()
    total = WriteResult()

    def absorb(part: WriteResult) -> None:
        total.kept.update(part.kept)
        total.dropped_by_cap.update(part.dropped_by_cap)
        total.read.update(part.read)
        total.stratum.update(part.stratum)
        total.quality_dropped.update(part.quality_dropped)
        total.refused_by_class.update(part.refused_by_class)
        total.textless_scripts.update(part.textless_scripts)
        total.unattributed += part.unattributed
        total.remap.merge(part.remap)
        total.family_written.update(part.family_written)
        total.family_distinct.update(part.family_distinct)
        total.signature_written.update(part.signature_written)
        total.policy.update(part.policy)
        total.legality.update(part.legality)
        for name, keys in part.kept_keys.items():
            total.kept_keys.setdefault(name, set()).update(keys)
        for name, keys in part.stratum_keys.items():
            total.stratum_keys.setdefault(name, set()).update(keys)

    if workers is not None and workers <= 1:
        init_write_worker(config)
        for index, name in enumerate(names, start=1):
            absorb(write_shard_pass(name))
            if index % progress_every == 0:
                logger.info("Wrote %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
        return total

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_write_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(
            pool.map(write_shard_pass, names, chunksize=1), start=1,
        ):
            absorb(part)
            if index % progress_every == 0:
                logger.info("Wrote %s", progress_line(
                    done=index, total=len(names),
                    elapsed=time.monotonic() - started,
                ))
    return total


def _repack_outputs(store: CorpusStore, *, shard_records: int) -> None:
    """Stream each stratum's parts into uniform shards, then drop the parts.

    The write pass writes one part per source shard, so an output's shard
    sizes mirror the raw corpus's -- hundreds of tiny files beside a handful
    of huge ones. The trainer reads a shard at a time and takes that shard's
    share of an epoch's steps, so uneven shards make the step budget uneven.
    """
    import shutil

    from effects.infrastructure.record_io import iter_shards, repack_shards

    for stratum, target in (
        ("training", store.training_dir),
        ("card-disjoint", store.card_disjoint_dir),
        ("game-disjoint", store.game_disjoint_dir),
        ("gate-one", store.gate_one_dir),
    ):
        parts = iter_shards(store.parts_dir_for(stratum))
        paths = repack_shards(parts, target, shard_records=shard_records)
        logger.info(
            "%-22s %d part(s) repacked into %d shard(s)",
            stratum, len(parts), len(paths),
        )
    shutil.rmtree(store.parts_dir, ignore_errors=True)


def _check_refusals(written: WriteResult) -> None:
    """Stop when the quality rules refused most of any one sampling class.

    A few percent refused per class is what the rules are for. A majority is
    one of three things, and the message names all three because the counts
    alone do not separate them: a corpus collected against an unpatched
    (degraded) Forge, which resolves no attribution and stamps every event
    ``unresolved``; a rule that does not fit the kind it is being applied to;
    or a corpus rebuilt against sidecars written before the converter claimed
    the traits whose text its lines carry, which leaves the records acting
    through keys the sidecar still drops. The third shows itself as
    ``no-acting-text`` dominating ``quality_dropped``, and a reconversion is
    the fix. Written anyway, the dataset is simply missing a sampling class,
    and the trainer reports that as a class this corpus happens not to hold.
    """
    for name in sorted(written.refused_by_class):
        refused = written.refused_by_class[name]
        kept = written.read.get(name, 0)
        total = refused + kept
        if not total or refused <= MAX_REFUSED_SHARE * total:
            continue
        raise BuildCorpusError(
            f"{name}: {refused} of the {total} record(s) read were refused by "
            f"the record-quality rules ({100.0 * refused / total:.1f}%), past "
            f"the {100.0 * MAX_REFUSED_SHARE:.0f}% a class may lose. A whole "
            "class refused is usually one of three things rather than a "
            "dirty corpus: an unpatched (degraded) checkout, which resolves "
            "no attribution and stamps every event `unresolved`; a quality "
            "rule that is wrong for this kind of record; or a corpus rebuilt "
            "before the converter's claims landed -- `no-acting-text` "
            "dominating `quality_dropped` means the sidecars still drop the "
            "keys the records act through, so reconvert and try again. Read "
            "the quality_dropped breakdown and the mode the shards were "
            "collected in before rebuilding."
        )


def _check_samples(counts: dict[str, int], *, size: int) -> None:
    """Stop when a stratum's validation sample came out empty.

    ``--validation-sample`` is what the trainer validates on and what selects
    the best checkpoint, so an empty one makes every epoch validate ``nan`` on
    that stratum -- reported, if at all, as a stratum the corpus happens not
    to hold. ``size <= 0`` asked for no sample at all, so nothing is wrong.
    """
    if size <= 0:
        return
    for stratum in sorted(counts):
        if counts[stratum]:
            continue
        raise BuildCorpusError(
            f"the {stratum} validation sample is empty: --validation-sample "
            f"{size} was asked for and no record of any sampled class was "
            f"found in the {stratum} stratum. The trainer validates on this "
            "sample, so a build that shipped it would validate nan every "
            "epoch and select no checkpoint. Check that the stratum holds "
            "records of the classes --class-mix names."
        )


def _token_key_remapper(
    config: BuildCorpusConfig, folders: dict[str, Path], card_files: dict[str, str],
) -> TokenKeyRemapper | None:
    """The remap both passes read through, or None when it is turned off.

    Built once in the main process and pickled into every worker with its
    config: it is dicts and frozensets, and re-reading Forge's token scripts
    per worker would cost more than the scan they serve.
    """
    if not config.remap_token_keys:
        return None

    from effects.domain.token_key_remap import load_token_script_facts

    forge_tokens = Path(config.forge_tokenscripts) if config.forge_tokenscripts else None
    if forge_tokens is None or not forge_tokens.is_dir():
        raise BuildCorpusError(
            f"--forge-tokenscripts {forge_tokens} is not a directory. The old "
            "collector keyed forked tokens to cardsfolder paths; remapping them "
            "needs Forge's raw token scripts (their Colors/PT lines) to tell "
            "same-name scripts apart. Point the flag at "
            "<forge checkout>/forge-gui/res/tokenscripts/, or pass "
            "--no-remap-token-keys to build without the remap."
        )
    token_dir = folders.get("tokenscripts")
    # The *converted* token tree: the remap rewrites a key only to a script
    # this corpus holds a sidecar for, so a token Forge scripts and the
    # converter never wrote is left alone rather than pointed at nothing.
    token_sidecars = (
        frozenset(path.stem for path in Path(token_dir).glob("*.txt"))
        if token_dir is not None and Path(token_dir).is_dir()
        else frozenset()
    )
    remapper = TokenKeyRemapper(
        load_token_script_facts(forge_tokens),
        converted_card_files=frozenset(card_files.values()),
        token_sidecars=token_sidecars,
    )
    logger.info(
        "Token key remap: %d token name(s) over %d script(s); %d converted token "
        "sidecar(s).",
        len(remapper.facts_by_name),
        sum(len(facts) for facts in remapper.facts_by_name.values()),
        len(token_sidecars),
    )
    if not token_sidecars:
        logger.warning(
            "No converted token sidecar under %s, so the remap can resolve "
            "nothing: every old token key will stay as it was collected. Pass "
            "--cards-folder output/tokenscripts/ beside the card tree.",
            # The folders actually looked in, not the `tokenscripts` entry that
            # is missing from them: `None` names nothing an operator can check.
            ", ".join(str(folder) for folder in config.cards_folders) or "(none)",
        )
    return remapper


def _report_remap(survey: Survey, written: WriteResult) -> None:
    """Log what the remap did, and say so when the two passes disagree."""
    remap = written.remap
    top = remap.ambiguous.most_common(5)
    logger.info(
        "%-22s %9d remapped to tokenscripts/; %d left ambiguous over %d stem(s), "
        "on written records%s",
        "token keys", remap.remapped, sum(remap.ambiguous.values()),
        len(remap.ambiguous),
        (" — most: " + ", ".join(f"{stem} ({n})" for stem, n in top)) if top else "",
    )
    # The survey counts every record it read; the write pass counts only the
    # records it wrote, so its figures are a subset of the survey's. Either one
    # exceeding the survey's means the two passes did not read the same corpus
    # -- and then the survey sized the caps for keys the write pass never saw.
    ambiguous, survey_ambiguous = sum(remap.ambiguous.values()), sum(
        survey.remap.ambiguous.values()
    )
    logger.info(
        "%-22s %9d remapped; %d left ambiguous — over every record read.",
        "token keys (survey)", survey.remap.remapped, survey_ambiguous,
    )
    if remap.remapped > survey.remap.remapped or ambiguous > survey_ambiguous:
        logger.warning(
            "The survey remapped %d key(s) and left %d ambiguous over every "
            "record it read, but the write pass counted %d and %d over the "
            "records it wrote — a subset cannot be larger, so the corpus "
            "changed between the passes.",
            survey.remap.remapped, survey_ambiguous, remap.remapped, ambiguous,
        )


def _nested(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (outer, inner), count in sorted(counts.items()):
        out.setdefault(outer, {})[inner] = count
    return out


def _family_tables(
    decisions: Decisions, written: WriteResult,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]]]:
    """The manifest's per-family and per-signature tables (FR-053).

    ``share`` is the family's equal split of its class budget, ``written`` and
    ``repeats`` what the write pass actually wrote, and ``shortfall`` the gap
    between the two that the coverage and variant collectors exist to close.
    """
    families: dict[str, dict[str, dict[str, int]]] = {}
    for klass, per_family in decisions.families.items():
        for label, row in per_family.items():
            wrote = written.family_written.get((klass, label), 0)
            distinct = written.family_distinct.get((klass, label), 0)
            families.setdefault(klass, {})[label] = {
                "available": row.available,
                "share": row.share,
                "written": wrote,
                "repeats": wrote - distinct,
                "shortfall": max(0, row.share - wrote),
            }
    signatures: dict[str, dict[str, dict[str, int]]] = {}
    for (klass, label, signature), count in sorted(written.signature_written.items()):
        if signature:
            signatures.setdefault(klass, {}).setdefault(label, {})[signature] = count
    return families, signatures


#: A held-out text recorded in fewer games than this is listed in the manifest
#: (FR-053), the floor the held-out coverage round collects to (FR-036).
HELD_OUT_GAME_FLOOR = 5


def _held_out_text_report(
    survey: Survey, decisions: Decisions, written: WriteResult,
    held_out_texts: frozenset[str],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Held-out texts with no gate-one record, and those under five games."""
    from effects.domain.text_holdout import normalize_script_text

    covered = {
        decisions.held_out_text_of_key[key]
        for key in written.stratum_keys.get("gate-one", ())
        if key in decisions.held_out_text_of_key
    }
    games: dict[str, set[str]] = defaultdict(set)
    for key, text in decisions.held_out_text_of_key.items():
        games[text] |= survey.held_out_text_games.get(key, set())
    texts = {normalize_script_text(text) for text in held_out_texts}
    without = tuple(sorted(texts - covered))
    under = {
        text: len(games.get(text, ()))
        for text in sorted(texts)
        if len(games.get(text, ())) < HELD_OUT_GAME_FLOOR
    }
    return without, under


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.application.train_effect_model import (
        load_card_files,
        load_card_texts,
        text_keyed_holdout,
    )
    from effects.domain.ability_encoder import surface_of
    from effects.domain.corpus_manifest import ClassCounts, CorpusManifest
    from effects.infrastructure.corpus_store import CorpusStore, current_sources
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    store = CorpusStore(config.output)
    records_dir = Path(config.records_dir)

    if config.verify:
        manifest = store.load()
        added, removed, resized = manifest.drift(current_sources(records_dir))
        for label, names in (("added", added), ("removed", removed), ("resized", resized)):
            if names:
                logger.warning(
                    "%d shard(s) %s since this dataset was built: %s%s",
                    len(names), label, ", ".join(names[:5]),
                    ", …" if len(names) > 5 else "",
                )
        # Drift stays shard-based, but which Forge token scripts were on disk
        # decides which old token keys resolved (FR-151), so a dataset built
        # against another token tree is a different dataset even when every
        # shard matches.
        built_against = manifest.forge_tokenscripts
        if built_against and built_against != str(Path(config.forge_tokenscripts or "")):
            logger.warning(
                "This dataset's token keys were remapped against %s, and "
                "--forge-tokenscripts now names %s. Which token scripts are on "
                "disk decides which keys resolved, so the two trees build "
                "different datasets from the same shards.",
                built_against, config.forge_tokenscripts,
            )
        if added or removed or resized:
            logger.warning("Rebuild with `python -m effects build-corpus`.")
            return 1
        logger.info("Dataset is current against %s.", records_dir)
        return 0

    folders = sidecar_roots(config.cards_folders)
    cards_folder = folders.get("cardsfolder", Path(config.cards_folders[0]))
    roots = dict(folders)
    if config.variant_scripts:
        # Keyed "variant-scripts" because that is the tree a variant line's
        # own provenance names, and the trainer registers it under the same
        # name. Without it every variant key resolves to no text: no rarity
        # entry and no per-text cap, while every real ability has both.
        roots["variant-scripts"] = Path(config.variant_scripts)
    sidecars = SidecarCache(roots)
    card_files = load_card_files(cards_folder)
    held_out = text_keyed_holdout(
        card_files,
        load_card_texts(card_files, sidecars),
        permille=config.holdout_permille,
        max_carriers=config.holdout_max_carriers,
        unit=config.holdout_unit,
    )
    surface = surface_of(config.vocab_path)
    logger.info(
        "Holdout (%s unit): %d ability text(s) on %d card(s); encoding surface %r.",
        config.holdout_unit, len(held_out.texts), len(held_out.names), surface,
    )
    if not held_out.names:
        tried = ", ".join(
            f"{folder} ({'found' if Path(folder).is_dir() else 'missing'})"
            for folder in config.cards_folders
        )
        raise BuildCorpusError(
            "Nothing is held out, so the card-disjoint stratum would be empty "
            "and gate 1 would have nothing to measure at all. Cards folders "
            f"tried: {tried}; {len(card_files)} converted card(s) read, "
            f"{len(held_out.texts)} held-out ability text(s), no held-out card. "
            "Run from the repository root so the relative --cards-folder paths "
            "resolve, raise --holdout-permille, or check that the converted "
            "tree has sidecars with script text."
        )

    remapper = _token_key_remapper(config, folders, card_files)

    from effects.infrastructure.record_io import (
        MixedGenerationsError,
        iter_shards,
        refuse_mixed_generations,
    )

    try:
        # Before any record is read (FR-033).
        refuse_mixed_generations(iter_shards(records_dir))
    except MixedGenerationsError as exc:
        raise BuildCorpusError(str(exc)) from exc

    survey = run_survey(
        records_dir,
        config=SurveyConfig(
            records_dir=str(records_dir),
            held_out_names=held_out.names,
            held_out_script_files=held_out.script_files,
            text_cap=config.text_cap,
            seed=config.seed,
            max_events=config.max_events,
            sidecar_roots={name: str(path) for name, path in roots.items()},
            remapper=remapper,
            game_disjoint_keywords=tuple(
                gate_keyword(name) for name in config.game_disjoint_keywords
            ),
        ),
        workers=config.workers,
    )
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card.",
        survey.records, len(survey.games), len(survey.held_out_games),
    )

    decisions = decide(
        survey, sidecars=sidecars, surface=surface,
        held_out_texts=held_out.texts, config=config,
    )
    if not decisions.card_disjoint:
        raise BuildCorpusError(
            "No game is admitted to the card-disjoint stratum, so the dataset "
            "cannot score gate 1 and the trainer would refuse it in its first "
            f"minute. {records_dir} holds {len(survey.games)} game(s), "
            f"{len(survey.held_out_games)} of which name one of the "
            f"{len(held_out.names)} held-out card(s). Depleted and "
            "full-strength shards live in sibling subdirectories of the raw "
            "corpus root: point --records-dir at the parent of both, and check "
            "that a full-strength run — pools generated without "
            "--exclude-cards — has actually been collected. A held-out game is "
            "admitted only while every held-out ability text it carries is "
            "under --card-disjoint-text-cap, and one that carries none is "
            "never admitted."
        )
    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(
            records_dir=str(records_dir),
            training_dir=str(store.parts_dir_for("training")),
            card_disjoint_dir=str(store.parts_dir_for("card-disjoint")),
            game_disjoint_dir=str(store.parts_dir_for("game-disjoint")),
            gate_one_dir=str(store.parts_dir_for("gate-one")),
            gate_one_keys=decisions.gate_one_keys,
            max_events=config.max_events,
            plans=decisions.plans,
            keyless_plans=decisions.keyless_plans,
            card_disjoint=decisions.card_disjoint,
            game_disjoint=decisions.game_disjoint,
            held_out_games=survey.held_out_games,
            seed=config.seed,
            sidecar_roots={name: str(path) for name, path in roots.items()},
            remapper=remapper,
        ),
        workers=config.workers,
    )
    _check_refusals(written)
    if remapper is not None:
        _report_remap(survey, written)
    else:
        logger.info(
            "%-22s %9s — every key was written exactly as collected "
            "(--no-remap-token-keys)", "token keys", "not remapped",
        )
    kept_records = sum(written.read.values())
    logger.info(
        "%-22s %9d record(s) kept (%.1f%%) name no producing clause for at "
        "least one event — watched, not refused; a rising share is the "
        "collector's attribution, not the corpus",
        "unattributed", written.unattributed,
        100.0 * written.unattributed / kept_records if kept_records else 0.0,
    )
    if survey.quality_dropped != written.quality_dropped:
        # Both passes run the same rule over the same shards, so they must
        # agree. A disagreement means the two passes read different records --
        # a corpus appended to between them, or a rule that is not pure over
        # the record -- and every count the manifest carries is then about a
        # corpus that no longer exists.
        logger.warning(
            "The survey and the write pass disagree about which records are "
            "refused: survey %s, write pass %s. Both read the same shards "
            "under the same rule, so this means the corpus changed between "
            "the passes.",
            dict(sorted(survey.quality_dropped.items())),
            dict(sorted(written.quality_dropped.items())),
        )
    _repack_outputs(store, shard_records=config.shard_records)

    if config.validation_sample > 0:
        from effects.application.validation_samples import draw_samples, write_samples

        samples = draw_samples(
            card_disjoint=store.card_disjoint_dir, game_disjoint=store.game_disjoint_dir,
            gate_one=store.gate_one_dir, mix=config.mix(), size=config.validation_sample,
            seed=config.seed, held_out_text_of_key=decisions.held_out_text_of_key,
        )
        sampled = write_samples(store, samples)
        for stratum, count in sampled.items():
            logger.info("%-22s %9d record(s) sampled for validation", stratum, count)
        _check_samples(sampled, size=config.validation_sample)
    else:
        logger.info("no validation sample written (--validation-sample 0)")

    per_class = {
        name: ClassCounts(
            read=written.read.get(name, 0),
            kept=written.kept.get(name, 0),
            dropped_by_cap=written.dropped_by_cap.get(name, 0),
            unique_texts=len({
                decisions.key_text.get(key, key)
                for key in written.kept_keys.get(name, ())
            }),
        )
        for name in sorted(written.read)
    }
    # The same fold as per_class's unique_texts, one axis over: by output
    # (stratum) rather than by sampling class (FR-143, FR-145). per_stratum
    # covers every name in STRATA; unique_texts covers every output in
    # OUTPUTS -- dropped-held-out writes nothing, so it has no ability texts
    # of its own to count.
    per_stratum = {name: written.stratum.get(name, 0) for name in STRATA}
    trained = sum(written.kept.values())
    delivered_mix = {
        name: count / trained for name, count in sorted(written.kept.items())
    } if trained else {}
    unique_texts = {
        name: len({
            decisions.key_text.get(key, key)
            for key in written.stratum_keys.get(name, ())
        })
        for name in OUTPUTS
    }
    families, signatures = _family_tables(decisions, written)
    held_out_report = _held_out_text_report(survey, decisions, written, held_out.texts)
    manifest = CorpusManifest(
        seed=config.seed,
        surface=surface,
        vocab_path=config.vocab_path,
        variant_scripts=config.variant_scripts or "",
        holdout_permille=config.holdout_permille,
        holdout_max_carriers=config.holdout_max_carriers,
        text_cap=config.text_cap,
        card_disjoint_text_cap=config.card_disjoint_text_cap,
        training_records=config.training_records,
        class_mix=config.mix(),
        delivered_mix=delivered_mix,
        held_out_cards=tuple(sorted(held_out.names)),
        held_out_texts=tuple(sorted(held_out.texts)),
        card_disjoint_games=tuple(sorted(decisions.card_disjoint)),
        game_disjoint_games=tuple(sorted(decisions.game_disjoint)),
        rarity=decisions.rarity,
        sources=survey.shards,
        per_class=per_class,
        per_stratum=per_stratum,
        unique_texts=unique_texts,
        shortfall=decisions.shortfall,
        quality_dropped=dict(written.quality_dropped),
        # The hundred worst scripts, not all of them: every permanent carries
        # the implicit spell this refusal is mostly about, and the manifest is
        # read by people.
        no_acting_text_scripts=dict(written.textless_scripts.most_common(100)),
        games_by_source=survey.games_by_source,
        held_out_games_by_source=survey.held_out_games_by_source,
        shard_records=config.shard_records,
        validation_sample=config.validation_sample,
        max_events_per_record=config.max_events,
        unattributed_records=written.unattributed,
        token_keys_remapped=written.remap.remapped,
        # The hundred worst stems, not all of them: a corpus can leave a long
        # tail of one-off ambiguities, and the manifest is read by people.
        token_keys_ambiguous=dict(written.remap.ambiguous.most_common(100)),
        forge_tokenscripts=(
            str(config.forge_tokenscripts) if remapper is not None else ""
        ),
        holdout_unit=config.holdout_unit,
        families=families,
        signatures=signatures,
        policy_counts=_nested(written.policy),
        legality_counts=_nested(written.legality),
        keyword_threshold_games=tuple(sorted(decisions.keyword_threshold_games)),
        held_out_texts_without_resolution=held_out_report[0],
        held_out_texts_under_five_games=held_out_report[1],
        game_disjoint_share=config.game_disjoint_share,
        game_disjoint_keyword_share=config.game_disjoint_keyword_share,
        game_disjoint_keywords=tuple(config.game_disjoint_keywords),
        reuse_cap=config.reuse_cap,
    )
    store.save(manifest)

    for name, counts in per_class.items():
        logger.info(
            "%-22s read %9d  kept %9d  cap dropped %8d  unique texts %7d",
            name, counts.read, counts.kept, counts.dropped_by_cap, counts.unique_texts,
        )
    for name in OUTPUTS:
        logger.info(
            "%-22s %9d record(s)  %7d unique text(s)",
            name, per_stratum[name], unique_texts[name],
        )
    logger.info("%-22s %9d record(s)", "dropped-held-out", per_stratum["dropped-held-out"])
    for source in sorted(survey.games_by_source):
        games = survey.games_by_source[source]
        held = survey.held_out_games_by_source.get(source, 0)
        share = held / games if games else 0.0
        report = logger.warning if 0 < share < 0.05 else logger.info
        report(
            "%-22s %7d game(s), %6d name a held-out card (%.1f%%)%s",
            source, games, held, 100.0 * share,
            " — a few held-out games in a directory that should hold none is "
            "a leak in collection, not a full-strength source" if 0 < share < 0.05 else "",
        )
    for reason, count in sorted(written.quality_dropped.items()):
        logger.info("%-22s %9d record(s) refused", reason, count)
    if written.textless_scripts:
        worst = sorted(written.textless_scripts.items(), key=lambda kv: -kv[1])[:5]
        logger.info(
            "no-acting-text refusals by script, most first: %s",
            ", ".join(f"{script} ({count})" for script, count in worst),
        )
    # Requested against delivered, side by side: the mixture in `class_mix` is
    # what the build was asked for, and only this says whether it got it.
    requested = config.mix()
    for name in sorted(set(requested) | set(delivered_mix)):
        asked, got = requested.get(name, 0.0), delivered_mix.get(name, 0.0)
        logger.info(
            "%-22s requested %6.2f%%  delivered %6.2f%%  (%.2fx)",
            name, 100.0 * asked, 100.0 * got, (got / asked) if asked else 0.0,
        )
    for name, missing in sorted(decisions.shortfall.items()):
        logger.warning(
            "%s is short %d record(s) of its share: --training-records asks for "
            "more than the corpus can supply at this mixture.", name, missing,
        )
    for klass, per_family in sorted(families.items()):
        for family, row in sorted(per_family.items()):
            if row["shortfall"]:
                logger.warning(
                    "%-22s family %-28s short %7d of its share %d (%d available, "
                    "%d written, %d repeats) — collect more",
                    klass, family, row["shortfall"], row["share"], row["available"],
                    row["written"], row["repeats"],
                )
    for subkind, counts in sorted(manifest.legality_counts.items()):
        logger.info(
            "%-22s real %9d  what-if %9d  unknown %9d",
            f"legality {subkind}", counts.get("real", 0), counts.get("what_if", 0),
            counts.get("unknown", 0),
        )
    logger.info(
        "game-disjoint %d game(s), %d of them under the keyword threshold; "
        "%d held-out text(s) with no gate-one resolution record, %d under five games",
        len(decisions.game_disjoint), len(decisions.keyword_threshold_games),
        len(held_out_report[0]), len(held_out_report[1]),
    )
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.kept.values()), store.directory, manifest.digest(),
    )
    return 0
