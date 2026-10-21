"""Build a fixed training corpus and two fixed validation strata (FR-135).

Two passes. The survey (:mod:`effects.application.corpus_survey`) reads every
shard and reports what the whole corpus holds and what its training-bound
records can supply; this module folds provenance keys to ability texts,
decides the split and the plan (:mod:`effects.domain.corpus_plan`), and the
write pass (:mod:`effects.application.corpus_write`) re-reads each shard and
keeps what the decisions admit. :mod:`effects.application.corpus_report`
assembles the manifest and the operator's report.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.corpus_report import (
    build_manifest,
    check_samples,
    log_report,
    report_write_pass,
)
from effects.application.corpus_survey import (
    ReadConfig,
    Survey,
    SurveyConfig,
    WorkerContext,
    run_survey,
)
from effects.application.corpus_write import WriteConfig, WritePlan, WriteResult, run_write_pass
from effects.domain.corpus_plan import (
    BuildCorpusError,
    GameRouting,
    GameStrata,
    Keys,
    SelectionPlan,
    admit_card_disjoint,
    members_of,
    plan_selection,
)
from effects.domain.keyword_names import normalize_keyword
from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
from effects.domain.records import acting_keys
from effects.domain.sampling import DEFAULT_KIND_MIX
from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE, HeldOutCards
from effects.domain.strata import OUTPUT_STRATA

if TYPE_CHECKING:
    from effects.domain.token_key_remap import TokenKeyRemapper
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

__all__ = [
    "DEFAULT_GAME_DISJOINT_KEYWORDS", "BuildCorpusConfig", "BuildCorpusError",
    "Decisions", "build", "decide",
]

logger = logging.getLogger(__name__)


#: FR-045's default keyword list, in the spelling an operator types.
DEFAULT_GAME_DISJOINT_KEYWORDS: tuple[str, ...] = (
    "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
)


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
    holdout_permille: int = HOLDOUT_PERMILLE
    holdout_max_carriers: int = HOLDOUT_MAX_CARRIERS
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
    max_events: int = MAX_EVENTS_PER_RECORD
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
        from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME

        unknown = [
            name for name in self.game_disjoint_keywords
            if normalize_keyword(name) not in KEYWORDS_BY_NAME
        ]
        if unknown:
            raise BuildCorpusError(
                f"--game-disjoint-keywords names {', '.join(unknown)}, which gate 2 "
                f"does not score; it scores {', '.join(sorted(KEYWORDS_BY_NAME))}."
            )

    def mix(self) -> dict[str, float]:
        return dict(self.class_mix or DEFAULT_KIND_MIX)

    @property
    def heap_cap(self) -> int:
        """How many smallest hashes the survey keeps per cell and key.

        ``--text-cap 0`` is no cap, so a text's quota is bounded by its repeats
        alone and every hash may be needed.
        """
        return self.text_cap if self.text_cap > 0 else sys.maxsize

    @property
    def routing(self) -> GameRouting:
        return GameRouting(
            game_disjoint_share=self.game_disjoint_share,
            keyword_share=self.game_disjoint_keyword_share,
        )


# ── the decisions ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Decisions:
    """Everything the write pass needs and the manifest records about the plan.

    The fold from acting keys to ability texts happened here, once, so the
    plans are keyed by cell and key tuple and a worker applies them without
    folding anything.
    """

    selection: SelectionPlan
    strata: GameStrata
    #: Game-disjoint games placed only because they hold a qualifying combat
    #: for a listed keyword (FR-053).
    keyword_threshold_games: frozenset[str]
    rarity: dict[str, int]
    key_text: dict[Keys, str]
    #: Kept records per legality mode (FR-053's ``legality_mode_counts``).
    mode_counts: dict[str, int]


def _text_of_keys(keys: Keys, sidecars: SidecarCache, surface: str) -> str | None:
    """The text a key tuple folds to, through the trainer's own definition."""
    from effects.application.ability_text import text_for_key

    for key in keys:
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


def _keyword_misfires(
    key_text: dict[Keys, str], sidecars: SidecarCache, definitions: Path | None,
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
    for keys in key_text:
        for key in keys:
            try:
                line = sidecars.line_for(key)
            except KeyError:
                continue
            if line is None or not line.script_text:
                continue
            if line.script_api_type == KEYWORD_API_TYPE:
                continue
            if display_name_of(line.script_text).lower() in names:
                found.append(f"{key.script_file}: {line.script_text!r}")
    return sorted(set(found))


def decide(
    survey: Survey,
    *,
    sidecars: SidecarCache,
    surface: str,
    config: BuildCorpusConfig,
) -> Decisions:
    """Fold keys to texts, split the games, and plan every record's copies.

    ``survey.supply`` must already be legality-resolved, as :func:`_survey`
    leaves it.
    """
    if survey.supply.heap_cap != config.heap_cap:
        raise ValueError(
            f"the survey kept {survey.supply.heap_cap} hashes per cell and key, "
            f"but config.text_cap {config.text_cap} keeps {config.heap_cap}. "
            "decide() must be run against the Survey that this same "
            "--text-cap produced, or the copy plans it writes would disagree "
            "with what the manifest is about to record."
        )

    key_text: dict[Keys, str] = {}
    text_games: dict[str, set[int]] = {}
    for keys in survey.rarity.key_records:
        text = _text_of_keys(keys, sidecars, surface)
        if text is None:
            # No sidecar line carries these keys: a runtime-only trait (level
            # up, bestow, a trigger another card's static granted) or, outside
            # the resolution class -- whose records acting through no text the
            # quality rules already refused -- a script the converted corpus
            # never held. Not a text, so it cannot be capped against one: its
            # records are sampled with the cell's keyless records.
            continue
        key_text[keys] = text
        text_games.setdefault(text, set()).update(survey.rarity.key_games.get(keys, ()))

    misfires = _keyword_misfires(key_text, sidecars, config.keyword_definitions)
    if misfires:
        raise BuildCorpusError(
            f"{len(misfires)} non-keyword line(s) would expand as a keyword by "
            "display name (FR-015): the text before the first colon matches a "
            "keyword definition while the line's script_api_type is not "
            "Keyword. First: " + "; ".join(misfires[:5])
        )

    held_out = frozenset(survey.games.held_out)
    game_disjoint = frozenset(survey.games.game_disjoint) - held_out
    card_disjoint = admit_card_disjoint(
        held_out, survey.games.held_out_text_games,
        cap=config.card_disjoint_text_cap, seed=config.seed,
    )
    members, keys_of = members_of(survey.supply, key_text)
    return Decisions(
        selection=plan_selection(
            members, keys_of, mix=config.mix(), ceiling=config.training_records,
            text_cap=config.text_cap, reuse_cap=config.reuse_cap,
        ),
        strata=GameStrata(
            card_disjoint=card_disjoint, game_disjoint=game_disjoint, held_out=held_out,
        ),
        keyword_threshold_games=frozenset(
            game for game in game_disjoint if config.routing.under_keyword_threshold(game)
        ),
        rarity={text: len(games) for text, games in text_games.items()},
        key_text=key_text,
        mode_counts=dict(sorted(survey.mode_counts.items())),
    )


# ── the build ───────────────────────────────────────────────────────────


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.domain.ability_encoder import surface_of
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(config.output)
    if config.verify:
        return _verify(config, store)

    sources = _Sources.of(config)
    held_out = _select_holdout(config, sources)
    surface = surface_of(config.vocab_path)
    logger.info(
        "Holdout (%s unit): %d ability text(s) on %d card(s) and %d token "
        "script(s); encoding surface %r.",
        config.holdout_unit, len(held_out.texts), len(held_out.names),
        sum(1 for script in held_out.script_files if script.startswith("tokenscripts/")),
        surface,
    )
    remapper = _token_key_remapper(config, sources.folders, sources.card_files)
    _refuse_mixed_generations(Path(config.records_dir))
    read = ReadConfig(
        records_dir=str(config.records_dir), sidecar_roots=sources.root_paths,
        max_events=config.max_events, held_out=held_out, seed=config.seed,
        remapper=remapper,
    )
    survey = _survey(config, read)
    decisions = _decide_or_refuse(config, survey, sources, surface, held_out)

    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(read=read, plan=WritePlan(
            parts_dirs={
                stratum: str(store.parts_dir_for(stratum)) for stratum in OUTPUT_STRATA
            },
            strata=decisions.strata,
            plans=decisions.selection.plans,
            keyless_plans=decisions.selection.keyless,
            mode_counts=decisions.mode_counts,
        )),
        workers=config.workers,
    )
    report_write_pass(survey, written, remapped=remapper is not None)
    _repack_outputs(store, written, shard_records=config.shard_records, seed=config.seed)
    _write_validation_samples(config, store, WorkerContext(read))

    manifest = build_manifest(
        config, surface=surface, held_out=held_out, survey=survey,
        decisions=decisions, written=written, remapped=remapper is not None,
    )
    store.save(manifest)
    log_report(manifest, survey, decisions, written)
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.training.kept.values()), store.directory, manifest.digest(),
    )
    return 0


def _verify(config: BuildCorpusConfig, store: CorpusStore) -> int:
    """``--verify``: the manifest's drift against the corpus on disk; writes nothing."""
    from effects.infrastructure.corpus_store import current_sources

    records_dir = Path(config.records_dir)
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


@dataclass(frozen=True)
class _Sources:
    """The converted trees a build reads its ability texts from."""

    #: Tree name -> converted root (``cardsfolder``, ``tokenscripts``).
    folders: dict[str, Path]
    #: ``folders`` plus the variant tree, when ``--variant-scripts`` names one.
    roots: dict[str, Path]
    sidecars: SidecarCache
    #: ``card name -> script file`` over the converted card tree.
    card_files: dict[str, str]

    @classmethod
    def of(cls, config: BuildCorpusConfig) -> _Sources:
        from effects.infrastructure.card_sources import load_card_files
        from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

        folders = sidecar_roots(config.cards_folders)
        roots = dict(folders)
        if config.variant_scripts:
            # Keyed "variant-scripts" because that is the tree a variant line's
            # own provenance names, and the trainer registers it under the same
            # name. Without it every variant key resolves to no text, and a
            # resolution acting through one is refused as textless.
            roots["variant-scripts"] = Path(config.variant_scripts)
        cards_folder = folders.get("cardsfolder", Path(config.cards_folders[0]))
        return cls(
            folders=folders, roots=roots, sidecars=SidecarCache(roots),
            card_files=load_card_files(cards_folder),
        )

    @property
    def root_paths(self) -> dict[str, str]:
        """``roots`` as strings, which is how a worker process receives them."""
        return {name: str(path) for name, path in self.roots.items()}


def _select_holdout(config: BuildCorpusConfig, sources: _Sources) -> HeldOutCards:
    """The trainer's own holdout rule over the converted trees; refuses an empty one.

    Token scripts are carriers like cards: a token printing a held-out text
    puts its script in the holdout, so the games it appears in are held out.
    """
    from effects.domain.split import text_keyed_holdout
    from effects.infrastructure.card_sources import load_card_texts
    from effects.infrastructure.sidecar_io import load_tree_texts

    held_out = text_keyed_holdout(
        sources.card_files,
        load_card_texts(sources.card_files, sources.sidecars),
        permille=config.holdout_permille,
        max_carriers=config.holdout_max_carriers,
        unit=config.holdout_unit,
        token_texts=load_tree_texts(sources.folders.get("tokenscripts")),
    )
    if not held_out.names:
        tried = ", ".join(
            f"{folder} ({'found' if Path(folder).is_dir() else 'missing'})"
            for folder in config.cards_folders
        )
        raise BuildCorpusError(
            "Nothing is held out, so the card-disjoint stratum would be empty "
            "and gate 1 would have nothing to measure at all. Cards folders "
            f"tried: {tried}; {len(sources.card_files)} converted card(s) read, "
            f"{len(held_out.texts)} held-out ability text(s), no held-out card. "
            "Run from the repository root so the relative --cards-folder paths "
            "resolve, raise --holdout-permille, or check that the converted "
            "tree has sidecars with script text."
        )
    return held_out


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

    from effects.domain.token_key_remap import TokenKeyRemapper
    from effects.infrastructure.token_script_io import load_token_script_facts

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


def _refuse_mixed_generations(records_dir: Path) -> None:
    """Before any record is read (FR-033)."""
    from effects.infrastructure.record_io import (
        MixedGenerationsError,
        iter_shards,
        refuse_mixed_generations,
    )

    try:
        refuse_mixed_generations(iter_shards(records_dir))
    except MixedGenerationsError as exc:
        raise BuildCorpusError(str(exc)) from exc


def _survey(config: BuildCorpusConfig, read: ReadConfig) -> Survey:
    """The parallel survey, its supply's legality cells folded into families."""
    survey = run_survey(
        Path(config.records_dir),
        config=SurveyConfig(
            read=read, heap_cap=config.heap_cap, routing=config.routing,
            game_disjoint_keywords=tuple(
                normalize_keyword(name) for name in config.game_disjoint_keywords
            ),
        ),
        workers=config.workers,
    )
    supply = survey.supply
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card, "
        "%d are game-disjoint; %d record(s) are training-bound, and %d more act "
        "through a held-out text.",
        survey.records, len(survey.games.games), len(survey.games.held_out),
        len(survey.games.game_disjoint),
        sum(supply.keyed.values()) + sum(supply.keyless.values()),
        sum(survey.held_out_text_dropped.values()),
    )
    return dataclasses.replace(survey, supply=supply.resolved(survey.mode_counts))


def _decide_or_refuse(
    config: BuildCorpusConfig, survey: Survey, sources: _Sources, surface: str,
    held_out: HeldOutCards,
) -> Decisions:
    """:func:`decide`, refusing a dataset gate 1 could not be scored on."""
    decisions = decide(survey, sidecars=sources.sidecars, surface=surface, config=config)
    if not decisions.strata.card_disjoint:
        raise BuildCorpusError(
            "No game is admitted to the card-disjoint stratum, so the dataset "
            "cannot score gate 1 and the trainer would refuse it in its first "
            f"minute. {config.records_dir} holds {len(survey.games.games)} game(s), "
            f"{len(survey.games.held_out)} of which name one of the "
            f"{len(held_out.names)} held-out card(s). Depleted and "
            "full-strength shards live in sibling subdirectories of the raw "
            "corpus root: point --records-dir at the parent of both, and check "
            "that a full-strength run — pools generated without "
            "--exclude-cards — has actually been collected. A held-out game is "
            "admitted only while every held-out ability text it resolves is "
            "under --card-disjoint-text-cap, and one that resolves none is "
            "never admitted."
        )
    return decisions


def _write_validation_samples(
    config: BuildCorpusConfig, store: CorpusStore, context: WorkerContext,
) -> None:
    if config.validation_sample <= 0:
        logger.info("no validation sample written (--validation-sample 0)")
        return
    from effects.application.validation_samples import draw_samples, write_samples

    samples = draw_samples(
        card_disjoint=store.card_disjoint_dir, game_disjoint=store.game_disjoint_dir,
        gate_one=store.gate_one_dir, mix=config.mix(), size=config.validation_sample,
        seed=config.seed,
        held_out_text_of=lambda record: context.held_out_text(acting_keys(record) or None),
    )
    sampled = write_samples(store, samples)
    for stratum, count in sampled.items():
        logger.info("%-22s %9d record(s) sampled for validation", stratum, count)
    check_samples(sampled, size=config.validation_sample)


def _repack_outputs(
    store: CorpusStore, written: WriteResult, *, shard_records: int, seed: int,
) -> None:
    """Deal each stratum's parts into uniform, mixed shards, then drop the parts.

    The write pass writes one part per source shard, so an output's shard
    sizes mirror the raw corpus's -- hundreds of tiny files beside a handful
    of huge ones -- and each part is one collection run's lifetime. The
    trainer reads a shard at a time and takes that shard's share of an
    epoch's steps, so uneven shards make the step budget uneven, and a shard
    of one run's games makes its batches correlated by run.
    """
    import shutil

    from effects.infrastructure.record_io import iter_shards, repack_shards

    for stratum in OUTPUT_STRATA:
        parts = iter_shards(store.parts_dir_for(stratum))
        paths = repack_shards(
            parts, store.output_dir(stratum), shard_records=shard_records,
            seed=seed, records=written.strata.records.get(stratum, 0),
        )
        logger.info(
            "%-22s %d part(s) repacked into %d shard(s)",
            stratum, len(parts), len(paths),
        )
    shutil.rmtree(store.parts_dir, ignore_errors=True)
