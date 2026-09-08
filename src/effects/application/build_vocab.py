"""``build-vocab``: the effects-side tokenizer vocabulary (FR-064, FR-065).

A thin wrapper over ``price_predictor.application.build_vocabulary``, mirroring
``sealed``'s, with three differences that matter:

- the scan covers **three** sources — converted cards, converted token scripts,
  and the keyword-definition file — because a token a card creates and a
  keyword's reminder text are both text the encoder has to read;
- ``[CLS]`` is seeded alongside ``[MASK]``, since the ability encoder pools
  through it to the bottleneck ``e``;
- the prose and script surfaces write **separate files**, so a stage-four
  rebuild never overwrites the vocabulary a stage-one-to-three checkpoint
  recorded and can still be loaded against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from price_predictor.application.build_vocabulary import (
    build_vocabulary,
    truncate_to_target_size,
)
from price_predictor.infrastructure.tokenizer_store import save_vocabulary

logger = logging.getLogger(__name__)

SURFACE_PROSE = "prose"
SURFACE_SCRIPT = "script"

DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"),
    Path("output/tokenscripts/"),
)
DEFAULT_KEYWORD_DEFINITIONS = Path("output/effects/keyword-definitions.json")
DEFAULT_VOCAB_PATH = Path("models/effects/vocab.txt")
DEFAULT_SCRIPT_VOCAB_PATH = Path("models/effects/vocab-script.txt")
DEFAULT_TARGET_SIZE = 5000

#: Seeded into every effects vocabulary regardless of corpus frequency.
#: ``[PAD]``, ``[UNK]`` and ``cardname`` come from the shared builder;
#: ``[MASK]`` is the MLM auxiliary's, and ``[CLS]`` is the pooling token the
#: bottleneck ``e`` is read from.
SEEDED_SPECIALS: tuple[str, ...] = ("[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]")


@dataclass
class BuildVocabConfig:
    surface: str = SURFACE_PROSE
    cards_folders: tuple[Path, ...] = DEFAULT_CARDS_FOLDERS
    vocab_path: Path | None = None
    keyword_definitions: Path = field(
        default_factory=lambda: DEFAULT_KEYWORD_DEFINITIONS,
    )
    target_size: int = DEFAULT_TARGET_SIZE
    printings_path: Path = field(
        default_factory=lambda: Path("resources/AllPrintings.json"),
    )

    def resolved_vocab_path(self) -> Path:
        """``--vocab-path``, defaulting per surface so the two never collide."""
        if self.vocab_path is not None:
            return Path(self.vocab_path)
        return (
            DEFAULT_SCRIPT_VOCAB_PATH
            if self.surface == SURFACE_SCRIPT
            else DEFAULT_VOCAB_PATH
        )


class EmptyCardsFolderError(RuntimeError):
    """No ``*.txt`` under any ``--cards-folder``. Maps to exit code 1."""


def _scan_sources(config: BuildVocabConfig) -> list[Path]:
    """Existing directories to scan, in the order they were given."""
    present = [Path(folder) for folder in config.cards_folders if Path(folder).is_dir()]
    if not present or not any(any(p.rglob("*.txt")) for p in present):
        raise EmptyCardsFolderError(
            "No converted cards under "
            + ", ".join(str(f) for f in config.cards_folders)
            + ". Run python -m price_predictor convert first."
        )
    return present


def _keyword_corpus_text(path: Path, surface: str = SURFACE_PROSE) -> str:
    """What a keyword expands to on this surface, as a blob for the scan.

    Keyword definitions are what expansion substitutes in, so every word in
    them has to be in the vocabulary — otherwise expanding a keyword would
    replace one known token with a sentence of ``[UNK]``.

    **Which field, though, follows the surface.** ``expand_keywords``
    substitutes the reminder template on the prose surface and the captured
    script from stage four (FR-060, FR-062); scanning both put Forge script
    syntax — ``activezones``, ``9999``, bare ``$`` and ``%`` — into a prose
    vocabulary that can never encode it. On the script surface the template is
    still scanned, because the engine-coded keywords generate no script and
    keep their template, and the encoder falls back to it for exactly those.
    """
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )

    lines: list[str] = []
    for definition in load_keyword_definitions(path).values():
        lines.append(definition.keyword)
        if definition.reminder_template:
            lines.append(definition.reminder_template)
        if surface == SURFACE_SCRIPT and definition.generated_script:
            lines.append(definition.generated_script)
    return "\n".join(lines)


def _script_corpus_text(folders: list[Path]) -> str:
    """Every sidecar's script lines, for the ``script`` surface's scan."""
    from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX, read_sidecar

    lines: list[str] = []
    for folder in folders:
        for sidecar_path in sorted(folder.rglob(f"*{SIDECAR_SUFFIX}")):
            for line in read_sidecar(sidecar_path).lines:
                if line.script_text:
                    lines.append(line.script_text)
                if line.script_api_type:
                    lines.append(line.script_api_type)
    return "\n".join(lines)


def run(config: BuildVocabConfig) -> int:
    """Build and write the vocabulary. Returns the resulting size."""
    if config.surface not in (SURFACE_PROSE, SURFACE_SCRIPT):
        raise ValueError(
            f"--surface must be {SURFACE_PROSE!r} or {SURFACE_SCRIPT!r}, "
            f"got {config.surface!r}"
        )
    folders = _scan_sources(config)
    printings = (
        Path(config.printings_path)
        if Path(config.printings_path).exists()
        else None
    )

    # The shared builder scans one directory tree, so the extra sources are
    # staged as text files in a scratch tree it can walk alongside the corpus.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="effects-vocab-") as scratch:
        extra = Path(scratch)
        keyword_path = Path(config.keyword_definitions)
        if keyword_path.exists():
            (extra / "keyword_definitions.txt").write_text(
                _keyword_corpus_text(keyword_path, config.surface),
                encoding="utf-8",
            )
        else:
            logger.warning(
                "No keyword-definition file at %s; the vocabulary will not "
                "cover reminder text, so expanding a keyword would produce "
                "[UNK] tokens. Run extract-keyword-definitions first.",
                keyword_path,
            )
        if config.surface == SURFACE_SCRIPT:
            (extra / "script_lines.txt").write_text(
                _script_corpus_text(folders), encoding="utf-8",
            )

        # Specials first, so they hold the lowest ids and survive a truncation
        # that drops from the tail. The shared builder seeds four of the five
        # itself; [CLS] is the effects side's own, and seeding it here rather
        # than appending it later keeps its id stable across rebuilds.
        vocab: dict[str, int] = {}
        _seed_specials(vocab)
        domain_token_count = len(SEEDED_SPECIALS)
        for source in [*folders, extra]:
            result = build_vocabulary(
                cards_path=source,
                freq_threshold=2,
                printings_path=printings,
            )
            # Every source seeds the same fixed domain terms, so the prefix
            # length is the largest of them plus the specials already in place.
            domain_token_count = max(
                domain_token_count,
                result.domain_token_count + 1,  # +1: [CLS], absent from theirs
            )
            for token in result.vocab:
                if token not in vocab:
                    vocab[token] = len(vocab)

        # Counted inside the scratch tree's lifetime: the staged sources are
        # real sources, and reporting only the card folders understates the
        # scan the docs describe as covering three.
        scanned = len(folders) + len(list(extra.glob("*.txt")))

    truncated = truncate_to_target_size(vocab, domain_token_count, config.target_size)

    vocab_path = config.resolved_vocab_path()
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(truncated, vocab_path)
    logger.info(
        "Wrote %d tokens to %s (surface=%s, %d sources)",
        len(truncated), vocab_path, config.surface, scanned,
    )
    return len(truncated)


def _seed_specials(vocab: dict[str, int]) -> None:
    """Ensure every seeded special is present, without disturbing existing ids."""
    for token in SEEDED_SPECIALS:
        if token not in vocab:
            vocab[token] = len(vocab)
