"""The held-out split: which ability texts, cards and games a model never trains on.

**The split excludes whole games, not held-out rows.** Records from one game
share a board, so a held-out card sitting in the *context* of a training record
leaks through its context role even when it is not the acting ability — and the
context role is exactly how the design trains each embedding from both
directions. Excluding the row and keeping the game would make gate 1, the
shipping gate, score partly on cards the encoder had already seen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from effects.domain.card_names import fold_card_name
from effects.domain.records import EffectRecord, key_sites

#: An eligible ability text is held out when crc32(text) % 1000 falls below
#: this. Keyed on the text rather than the card because the model never reads a
#: card's name, and stable per text so a depleted corpus stays valid (FR-088).
HOLDOUT_PERMILLE = 20
#: A text more cards than this carry is never eligible. `Flying` is on thousands
#: of cards and depleting all of them would empty the training corpus.
HOLDOUT_MAX_CARRIERS = 8


@dataclass(frozen=True, slots=True)
class HeldOutCards:
    """The card-disjoint holdout, addressable both ways a record names a card.

    A record names a card by the entity names in its snapshot and by the script
    files its provenance keys point at; matching on either catches both. A
    token script carrying a held-out text is in ``script_files`` too, though
    not in ``names``: a token is never dealt into a deck, so it has no name the
    depletion list could remove, and its printed name is shared by scripts
    that hold nothing back.

    ``names`` is folded on construction (:func:`fold_card_name`), because the
    two sides of this comparison spell a card differently and a caller that
    passed printed case would produce a holdout matching nothing.
    """

    names: frozenset[str]
    script_files: frozenset[str]
    #: The held-out ability texts themselves. Under depletion no training card
    #: carries one, so this is also gate 1's unique-text slice — sizing the
    #: stratum needs no scan of the training corpus.
    texts: frozenset[str] = frozenset()
    #: The held-out units the texts were chosen by (masked templates under
    #: ``template``, the texts again under ``text``), and that unit: what
    #: :meth:`holds_text` tests a text no carrier prints against.
    keys: frozenset[str] = frozenset()
    unit: str = "text"

    def __post_init__(self) -> None:
        folded = frozenset(fold_card_name(name) for name in self.names)
        if folded != self.names:
            object.__setattr__(self, "names", folded)

    def __len__(self) -> int:
        return len(self.script_files or self.names)

    def holds_text(self, script_text: str) -> bool:
        """Whether a script text is held out, carried by a card or not.

        By its unit rather than by membership in ``texts``: a variant script
        or a token can print a text no card carries, and under ``template``
        it is held out whenever its masked template is.
        """
        from effects.domain.text_holdout import holdout_key

        return holdout_key(script_text, self.unit) in self.keys


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """The split a checkpoint records and the evaluator reads back.

    Both strata are stored as explicit ``game_id`` sets rather than as a rule,
    because the corpus is append-only and grows between runs: a recomputed
    split would not be the trained-against one, and the gates would score partly
    on games the model had trained on.
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
    *,
    permille: int = HOLDOUT_PERMILLE,
    max_carriers: int = HOLDOUT_MAX_CARRIERS,
    unit: str = "text",
    token_texts: Mapping[str, Iterable[str]] | None = None,
) -> HeldOutCards:
    """Hold out ability texts, and with them every card carrying one (FR-088).

    ``unit`` selects what is counted and hashed (FR-040, FR-041): ``template``
    holds out every text sharing a masked template together, ``text`` is
    feature 023's rule. ``holdout-cards``, ``build-corpus`` and
    ``train-effect-model`` all go through here, so the three agree by
    construction rather than by three implementations kept in step.

    Args:
        card_files: ``card name -> script file``, one entry per converted card
            under ``output/cardsfolder/``. Variant scripts are not in this
            denominator: one exists only because a card was perturbed.
        texts_by_card: ``card name -> its sidecar's script texts``, from
            ``load_card_texts``.
        token_texts: ``token script file -> its sidecar's script texts``, from
            ``load_tree_texts``. A token carries a text exactly as a card
            does: it counts toward the carrier ceiling, a text only tokens
            print can be held out, and a token carrying a held-out text puts
            its script file in ``script_files``. Left out, a token printing a
            held-out text trained on it, and every game it sat in did too.

    The text rather than the card, because the model never reads a card's name
    and functional reprints compile to the same script: a name-keyed holdout
    trains on Lightning Strike while holding out Searing Spear, and gate 1 then
    drops those records for having text a training card carries.
    """
    from effects.domain.text_holdout import select_holdout

    tokens = dict(token_texts or {})
    chosen = select_holdout(
        {**texts_by_card, **tokens}, permille=permille, max_carriers=max_carriers,
        unit=unit,
    )
    names = frozenset(chosen.cards & card_files.keys())
    return HeldOutCards(
        names=names,
        script_files=frozenset(card_files[name] for name in names)
        | (chosen.cards & tokens.keys()),
        texts=chosen.texts,
        keys=chosen.keys,
        unit=unit,
    )


def record_names_held_out_card(
    record: EffectRecord, held_out: HeldOutCards,
) -> bool:
    """Whether this record names a held-out card or token, acting or in context.

    By entity name, and by every key the record carries (``key_sites``): the
    acting line, each entity's printed and granted lines, and the payload's
    keys -- a decision's candidates, a static keeping a creature out of
    combat, a rewrite's replacement.
    """
    for entity in record.state.entities:
        if fold_card_name(entity.name) in held_out.names:
            return True
    return any(key.script_file in held_out.script_files for key in key_sites(record))
