"""The ``build-corpus`` flags as one value (root spec § Curated corpus).

Each field's ``help`` metadata is its flag's help, so a default lives here
alone; ``cli.add_config_flags`` turns the fields into flags.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain import default_paths
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from effects.domain.keyword_names import normalize_keyword
from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
from effects.domain.sampling import DEFAULT_KIND_MIX
from effects.domain.split import HoldoutRule
from effects.domain.stratum_placement import (
    MAX_COMPONENT_ERROR,
    MAX_ITEM_ERROR,
    MIN_GAMES_PER_HALF,
    PlacementRule,
)
from price_predictor.infrastructure.forge_worker_pool import MAX_WORKERS, worker_count

if TYPE_CHECKING:
    from effects.infrastructure.converted_trees import SourceTrees

__all__ = ["DEFAULT_GAME_DISJOINT_KEYWORDS", "BuildCorpusConfig", "parse_keyword_list"]


#: FR-045's default keyword list, in the spelling an operator types.
DEFAULT_GAME_DISJOINT_KEYWORDS: tuple[str, ...] = (
    "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
)


def parse_keyword_list(text: str) -> tuple[str, ...]:
    """``--game-disjoint-keywords``: a comma list, blanks dropped."""
    return tuple(part.strip() for part in text.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus).

    ``help`` metadata is the flag's help, so a default lives here alone. A
    directory's help names its default as an operator types it, trailing
    slash and all, which the ``Path`` the flag parses to cannot spell.
    """

    records_dir: Path = field(kw_only=True, metadata={"help": (
        "Raw corpus root, a records run; discovery recurses"
    )})
    output: Path = field(kw_only=True, metadata={"help": (
        "Curated dataset directory, a corpus run (output/effects/<gen>/corpus/<run>/)"
    )})
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    #: A string rather than a path: the manifest records it, and a Path would
    #: rewrite its separators on Windows.
    vocab_path: str = field(kw_only=True, metadata={"help": (
        "Decides the encoding surface the rarity table's text keys are built on"
    )})
    variant_scripts: str | None = None
    holdout: HoldoutRule = HoldoutRule()
    text_cap: int = field(default=200, metadata={"help": (
        "Max written training records per ability text per selection cell, "
        "repeats included; 0 means no cap (default: %(default)s)"
    )})
    reuse_cap: int = field(default=4, metadata={"help": (
        "Max copies of one record toward a short effect family's share (FR-049); "
        "1 means no repeats (default: %(default)s)"
    )})
    class_mix: dict[str, float] | None = None
    training_records: int = field(default=0, metadata={"help": (
        "Total training records after capping, subsampled per class; 0 means "
        "no ceiling (default: %(default)s)"
    )})
    game_disjoint_share: float = field(default=0.01, metadata={"help": (
        "A game naming no held-out card enters the game-disjoint stratum when "
        "crc32(game_id) mod 1e6 / 1e6 is below this: the draw depends on the "
        "game id alone, so a rebuild over a grown corpus draws the same games "
        "again, and placement then promotes further games from just above the "
        "share where a rule cell is short, which a grown corpus can change "
        "(default: %(default)s)"
    )})
    game_disjoint_keyword_share: float = field(default=0.15, metadata={"help": (
        "The threshold instead for a game holding a qualifying combat for one "
        "of --game-disjoint-keywords (default: %(default)s)"
    )})
    game_disjoint_keywords: tuple[str, ...] = field(
        default=DEFAULT_GAME_DISJOINT_KEYWORDS, metadata={
            "type": parse_keyword_list,
            "help": (
                "Comma list of damage-step keywords raising a game's threshold "
                f"(default: {','.join(DEFAULT_GAME_DISJOINT_KEYWORDS)})"
            ),
        },
    )
    #: Read by FR-015's check; None or a missing file skips it, since nothing
    #: can expand without a table.
    keyword_definitions: Path | None = field(kw_only=True, metadata={"help": (
        "The keyword table the tokenizer expands with; the build fails when "
        "a display-name match would expand a non-keyword line; empty skips the check"
    )})
    card_disjoint_text_cap: int = field(default=50, metadata={"help": (
        "Max card-disjoint games admitted per held-out ability text, so each "
        "weighs comparably in gate 1; 0 means no cap (default: %(default)s)"
    )})
    shard_records: int = field(default=2000, metadata={"help": (
        "Records per output shard: whole games are dealt to shards by a seeded "
        "hash of their game id; 0 keeps one output shard per source shard "
        "(default: %(default)s)"
    )})
    max_events_per_record: int = field(default=MAX_EVENTS_PER_RECORD, metadata={"help": (
        "A record carrying more events than this is refused as a game dump "
        "rather than an ability; 0 disables the event-flood rule "
        "(default: %(default)s)"
    )})
    validation_sample: int = field(default=2048, metadata={"help": (
        "Records per stratum in validation/samples/, mixture-matched; 0 writes "
        "no sample (default: %(default)s)"
    )})
    seed: int = field(default=42, metadata={"help": (
        "Seeds the selection and the repacking (default: %(default)s)"
    )})
    workers: int = field(default=MAX_WORKERS, metadata={
        "type": worker_count,
        "help": f"Parallel shard readers, 1 to {MAX_WORKERS} (default: %(default)s)",
    })
    verify: bool = field(default=False, metadata={"help": (
        "Report drift against an existing manifest and write nothing"
    )})
    #: The rules catalogue (``scenarios/`` and ``heads/``), read once. No
    #: flag (contracts/cli.md): a field so a test can point at its own.
    rules_dir: Path = Path("rules")
    #: The precision thresholds of research E5. No operator flag overrides
    #: them; fields so a test can lower them.
    min_games_per_half: int = MIN_GAMES_PER_HALF
    max_item_error: float = MAX_ITEM_ERROR
    max_component_error: float = MAX_COMPONENT_ERROR

    def __post_init__(self) -> None:
        self._refuse_negative_counts()
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
        self._refuse_unscored_keywords()

    def _refuse_negative_counts(self) -> None:
        """Refuse a negative count; zero is a real setting for each.

        - ``--text-cap 0`` — no per-text cap; ``--reuse-cap`` alone bounds a
          text's quota in a cell.
        - ``--card-disjoint-text-cap 0`` — every held-out game carrying a
          held-out text is admitted.
        - ``--game-disjoint-share 0`` — no game-disjoint stratum, unless the
          keyword share places some games.
        - ``--training-records 0`` — no ceiling beyond ``--text-cap``.
        - ``--shard-records 0`` — no repacking; one output shard per source
          shard.
        - ``--max-events-per-record 0`` — no event-flood rule; the other two
          quality rules stay on.
        - ``--validation-sample 0`` — nothing written under
          ``validation/samples/``.
        """
        for flag, value in (
            ("--text-cap", self.text_cap),
            ("--card-disjoint-text-cap", self.card_disjoint_text_cap),
            ("--training-records", self.training_records),
            ("--shard-records", self.shard_records),
            ("--max-events-per-record", self.max_events_per_record),
            ("--validation-sample", self.validation_sample),
        ):
            if value < 0:
                raise BuildCorpusError(
                    f"{flag} is {value}; it counts things and cannot be "
                    "negative. Zero is allowed and means no cap, no ceiling, "
                    "or an empty stratum depending on the flag — see --help."
                )

    def _refuse_unscored_keywords(self) -> None:
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
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        from effects.infrastructure.converted_trees import SourceTrees

        return SourceTrees.of(self.cards_folders, self.variant_scripts)

    @property
    def heap_cap(self) -> int:
        """How many smallest hashes the survey keeps per cell and key; every
        one under ``--text-cap 0``."""
        return self.text_cap if self.text_cap > 0 else sys.maxsize

    @property
    def placement(self) -> PlacementRule:
        """Placement's flags and precision thresholds (research R3, E5)."""
        return PlacementRule(
            game_disjoint_share=self.game_disjoint_share,
            keyword_share=self.game_disjoint_keyword_share,
            card_disjoint_text_cap=self.card_disjoint_text_cap,
            seed=self.seed,
            min_games_per_half=self.min_games_per_half,
            max_item_error=self.max_item_error,
            max_component_error=self.max_component_error,
        )
