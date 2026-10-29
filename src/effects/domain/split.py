"""The held-out split: which ability texts, cards and games a model never trains on.

The split excludes whole games, not rows: a held-out card in the context of a
training record would leak through its context role.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from effects.domain.card_names import fold_card_name
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import EffectRecord, key_sites
from effects.domain.text_holdout import HoldoutUnit, holdout_key, select_holdout

#: An eligible ability text is held out when crc32(text) % 1000 falls below
#: this, stable per text so a depleted corpus stays valid (FR-088).
HOLDOUT_PERMILLE = 20
#: A text more cards than this carry is never eligible; depleting every
#: carrier of `Flying` would empty the training corpus.
HOLDOUT_MAX_CARRIERS = 8


@dataclass(frozen=True, slots=True)
class HoldoutRule:
    """How the held-out texts are chosen: one value, so every command computing
    the holdout is handed the same rule rather than three flags to keep in step.

    The defaults are the commands' own; a corpus manifest or checkpoint that
    predates the unit records ``text``.
    """

    permille: int = HOLDOUT_PERMILLE
    max_carriers: int = HOLDOUT_MAX_CARRIERS
    unit: HoldoutUnit = HoldoutUnit.TEMPLATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", HoldoutUnit(self.unit))


@dataclass(frozen=True, slots=True)
class HeldOutCards:
    """The card-disjoint holdout, addressable both ways a record names a card.

    By snapshot entity name and by provenance-key script file. A token script
    carrying a held-out text is in ``script_files`` but not ``names``: a token
    is never in a deck, and its printed name is shared by scripts that hold
    nothing back. ``names`` is folded on construction (:func:`fold_card_name`).
    """

    names: frozenset[str]
    script_files: frozenset[str]
    #: The held-out ability texts; under depletion also gate 1's unique-text
    #: slice.
    texts: frozenset[str] = frozenset()
    #: The held-out units (masked templates under ``template``, the texts under
    #: ``text``) that :meth:`holds_text` tests against.
    keys: frozenset[str] = frozenset()
    unit: HoldoutUnit = HoldoutUnit.TEXT

    def __post_init__(self) -> None:
        folded = frozenset(fold_card_name(name) for name in self.names)
        if folded != self.names:
            object.__setattr__(self, "names", folded)

    def __len__(self) -> int:
        return len(self.script_files or self.names)

    def holds_text(self, script_text: str) -> bool:
        """Whether a script text is held out, carried by a card or not.

        By unit rather than ``texts`` membership: a variant or a token can
        print a text no card carries.
        """
        return holdout_key(script_text, self.unit) in self.keys

    def text_acted_through(
        self, keys: Iterable[ProvenanceKey],
        script_text_of: Callable[[ProvenanceKey], str | None],
    ) -> str | None:
        """The held-out script text a record's acting ``keys`` act through, or None.

        The first acting key whose script text :meth:`holds_text`, so under
        ``template`` an earlier key printing a held-out template takes the
        record even where no card carries that exact text. ``build-corpus``'s
        survey and ``collect-coverage``'s held-out round both credit a game to
        a held-out text through here, so the two count its games alike
        (FR-053). ``script_text_of`` answers with the normalized script text.
        """
        def held_out(key: ProvenanceKey) -> str | None:
            text = script_text_of(key)
            return text if text is not None and self.holds_text(text) else None

        return first_text(keys, held_out)


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """The split a checkpoint records and the evaluator reads back.

    Both strata are explicit ``game_id`` sets, never a rule: the corpus grows
    between runs, so a recomputed split would not be the trained-against one.
    """

    held_out_cards: tuple[str, ...] = ()
    card_disjoint_games: frozenset[str] = frozenset()
    game_disjoint_games: frozenset[str] = frozenset()

    @property
    def validation_games(self) -> frozenset[str]:
        return self.card_disjoint_games | self.game_disjoint_games


def text_keyed_holdout(
    card_files: dict[str, str],
    texts_by_card: Mapping[str, Iterable[str]],
    rule: HoldoutRule,
    *,
    token_texts: Mapping[str, Iterable[str]] | None = None,
) -> HeldOutCards:
    """Hold out ability texts, and with them every card carrying one (FR-088).

    ``rule.unit`` selects what is counted and hashed (FR-040, FR-041).
    ``holdout-cards``, ``build-corpus`` and ``train-effect-model`` all go
    through here, so the three agree by construction.

    Args:
        card_files: ``card name -> script file`` per converted card; variant
            scripts are not in this denominator.
        texts_by_card: ``card name -> its sidecar's script texts``.
        token_texts: ``token script file -> its sidecar's script texts``. A
            token counts as a carrier exactly as a card does.
    """
    tokens = dict(token_texts or {})
    chosen = select_holdout(
        {**texts_by_card, **tokens}, permille=rule.permille,
        max_carriers=rule.max_carriers, unit=rule.unit,
    )
    names = frozenset(chosen.cards & card_files.keys())
    return HeldOutCards(
        names=names,
        script_files=frozenset(card_files[name] for name in names)
        | (chosen.cards & tokens.keys()),
        texts=chosen.texts,
        keys=chosen.keys,
        unit=rule.unit,
    )


def record_names_held_out_card(
    record: EffectRecord, held_out: HeldOutCards,
) -> bool:
    """Whether this record names a held-out card or token, acting or in context.

    By entity name, and by every key the record carries (``key_sites``).
    """
    for entity in record.state.entities:
        if fold_card_name(entity.name) in held_out.names:
            return True
    return any(key.script_file in held_out.script_files for key in key_sites(record))
