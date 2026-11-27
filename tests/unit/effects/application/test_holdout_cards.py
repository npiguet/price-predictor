"""``holdout-cards``: the depletion list `generate-pools` reads.

The list and the trainer's split must agree exactly. They are two programs
reading two different things — one the converted tree, the other its own
config — and a disagreement is silent: pools would be depleted of one set of
cards while the split held out another, so held-out cards would reach training
games and the card-disjoint stratum would hold games that are not disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.holdout_cards import (
    HoldoutCardsConfig,
    HoldoutCardsError,
    HoldoutReport,
    holdout_report,
    run,
)
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.split import HoldoutRule, text_keyed_holdout
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.builders import write_sidecar


@dataclass(frozen=True)
class _Printings:
    """What MTGJSON says about the cards: printed spellings and split faces."""

    canonical_names: dict | None = None
    split_faces: dict | None = None


def _report(card_files, texts_by_card, *, printings=_Printings(), **rule):
    """The report over ``text_keyed_holdout``, as ``run`` builds it from the trees."""
    return holdout_report(
        text_keyed_holdout(card_files, texts_by_card, HoldoutRule(**rule)),
        converted_cards=len(card_files), canonical_names=printings.canonical_names,
        **({} if printings.split_faces is None else {"split_faces": printings.split_faces}),
    )


def _convert(root: Path, script_file: str, name: str, texts: list[str]) -> None:
    """A converted ``.txt`` named ``name`` and a sidecar whose lines print ``texts``."""
    tree, relative = script_file.split("/", 1)
    txt = root / tree / relative
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(f"name: {name}\ntypes: instant\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card=name, script_file=script_file,
            lines=tuple(
                SidecarLine(
                    line_index=i, line_kind="spell",
                    provenance=(ProvenanceKey(script_file, 0, "spell", i),),
                    script_text=text, prose_offset=0,
                )
                for i, text in enumerate(texts)
            ),
        ),
        sidecar_path_for(txt),
    )


def _run(root: Path, out: Path, **rule) -> HoldoutReport:
    return run(HoldoutCardsConfig(
        out=out,
        cards_folders=(root / "cardsfolder", root / "tokenscripts"),
        printings_path=root / "AllPrintings.json",
        holdout=HoldoutRule(**{"max_carriers": 8, **rule}),
    ))


_SPEAR = "SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any"


def test_the_list_is_the_cards_the_trainer_holds_out(tmp_path) -> None:
    """One rule, exercised through both entry points."""
    _convert(tmp_path, "cardsfolder/s/searing_spear.txt", "Searing Spear", [_SPEAR])
    _convert(tmp_path, "cardsfolder/l/lightning_strike.txt", "Lightning Strike", [_SPEAR])
    _convert(tmp_path, "cardsfolder/g/grizzly_bears.txt", "Grizzly Bears", [])
    card_files = {
        "Searing Spear": "cardsfolder/s/searing_spear.txt",
        "Lightning Strike": "cardsfolder/l/lightning_strike.txt",
        "Grizzly Bears": "cardsfolder/g/grizzly_bears.txt",
    }
    texts_by_card = {
        "Searing Spear": [_SPEAR], "Lightning Strike": [_SPEAR], "Grizzly Bears": [],
    }
    out = tmp_path / "holdout-cards.txt"

    for permille in (0, 250, 500, 750, 1000):
        listed = _run(tmp_path, out, permille=permille).names
        held = text_keyed_holdout(
            card_files, texts_by_card, HoldoutRule(permille, 8),
        )
        assert set(listed) == held.names, f"disagreed at permille={permille}"
        assert out.read_text(encoding="utf-8").splitlines() == listed


def test_token_carriers_move_the_list_as_they_move_the_split(tmp_path) -> None:
    """FR-041: holdout-cards, build-corpus and the trainer share one rule,
    token carriers included, or the list depletes cards the split trains on."""
    _convert(tmp_path, "cardsfolder/s/searing_spear.txt", "Searing Spear", [_SPEAR])
    out = tmp_path / "holdout-cards.txt"
    assert _run(tmp_path, out, permille=1000, max_carriers=1).names == [
        "searing spear",
    ]

    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", [_SPEAR])

    assert _run(tmp_path, out, permille=1000, max_carriers=1).names == [], (
        "a token printing the text is a second carrier, over the ceiling of one"
    )


def test_no_converted_card_is_refused(tmp_path) -> None:
    with pytest.raises(HoldoutCardsError, match="no converted cards"):
        _run(tmp_path, tmp_path / "holdout-cards.txt")


def test_the_list_is_sorted_so_two_runs_produce_the_same_file() -> None:
    """A file that reordered between runs would look like a changed holdout."""
    card_files = {
        "Zed": "cardsfolder/z/zed.txt",
        "Alice": "cardsfolder/a/alice.txt",
        "Mallory": "cardsfolder/m/mallory.txt",
    }
    texts_by_card = {
        name: [f"SP$ Bespoke | Who$ {name}"] for name in card_files
    }

    listed = _report(card_files, texts_by_card, permille=1000, max_carriers=8).names

    assert listed == sorted(listed)
    assert listed == ["alice", "mallory", "zed"], (
        "with no printings file to resolve against, the list keeps the "
        "converted tree's spelling"
    )


class TestTheFileSpeaksForgesSpelling:
    """`holdout-cards.txt` crosses into Forge, so it should carry Forge's names.

    Folding at every comparison makes the case irrelevant to correctness. The
    file is still written in printed case, because it is an interface file: an
    operator reads it, Forge's own `getName()` is what it will be matched
    against, and a future consumer that forgets to fold should not be the one
    that discovers the convention.
    """

    def test_names_are_resolved_to_their_printed_spelling(self) -> None:
        listed = _report(
            {"soul echo": "cardsfolder/s/soul_echo.txt"},
            {"soul echo": ["SP$ Bespoke | Weird$ True"]},
            permille=1000, max_carriers=8,
            printings=_Printings(canonical_names={"Soul Echo": "2004-10-01"}),
        ).names

        assert listed == ["Soul Echo"]

    def test_an_unresolvable_name_keeps_the_converted_spelling(self) -> None:
        """A token or a card MTGJSON has never heard of still gets depleted."""
        listed = _report(
            {"nonesuch": "cardsfolder/n/nonesuch.txt"},
            {"nonesuch": ["SP$ Bespoke | Weird$ True"]},
            permille=1000, max_carriers=8,
            printings=_Printings(canonical_names={"Soul Echo": "2004-10-01"}),
        ).names

        assert listed == ["nonesuch"]


class TestTheTrainingCorpusFlag:
    """One flag for a corpus and the holdout list that depleted it.

    They belong together: the list is what the corpus was built against, and
    naming them separately invites the mismatch that silently produces an
    undepleted corpus. `--training-corpus DIR` says both live in DIR.
    """

    def _config(self, *argv):
        from effects.infrastructure.cli import build_parser, collector_config

        return collector_config(
            CollectCoverageConfig,
            build_parser().parse_args(["collect-coverage", *argv]),
        )

    def test_it_sets_both_the_records_dir_and_the_holdout_list(self, tmp_path):
        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")

        config = self._config("--training-corpus", str(corpus))

        assert config.effect_records == corpus
        assert config.deck_holdout.exclude_cards == corpus / "holdout-cards.txt"

    def test_a_corpus_without_a_holdout_list_is_refused(self, tmp_path):
        """Silently running undepleted is the failure this whole area had."""
        from effects.infrastructure.cli import build_parser, collector_config

        corpus = tmp_path / "depleted"
        corpus.mkdir()

        with pytest.raises(ValueError, match="holdout-cards.txt"):
            collector_config(CollectCoverageConfig, build_parser().parse_args(
                ["collect-coverage", "--training-corpus", str(corpus)]
            ))

    def test_it_refuses_to_share_the_stage_with_the_flags_it_implies(self, tmp_path):
        from effects.infrastructure.cli import build_parser, collector_config

        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")

        with pytest.raises(ValueError, match="one source"):
            collector_config(CollectCoverageConfig, build_parser().parse_args([
                "collect-coverage", "--training-corpus", str(corpus),
                "--effect-records", str(tmp_path / "elsewhere"),
            ]))

    def test_without_it_the_ordinary_flags_still_work(self, tmp_path):
        config = self._config("--effect-records", str(tmp_path / "records"))

        assert config.effect_records == tmp_path / "records"
        assert config.deck_holdout.exclude_cards is None


# ── the template unit (FR-040, FR-043) ──────────────────────


def test_the_report_counts_templates_texts_cards_and_the_depleted_share():
    card_files = {f"card {i}": f"cardsfolder/c/card_{i}.txt" for i in range(6)}
    texts_by_card = {
        "card 0": ["SP$ Bespoke | Weird$ 1"],
        "card 1": ["SP$ Bespoke | Weird$ 2"],          # same template as card 0
        "card 2": ["SP$ Other | Thing$ X"],
        "card 3": ["SP$ Other | Thing$ X"],            # same text as card 2
        "card 4": [], "card 5": [],
    }
    report = _report(
        card_files, texts_by_card, permille=1000, max_carriers=8,
    )

    assert report.keys == 2
    assert report.texts == 3
    assert len(report.names) == 4
    assert report.depleted_share == 4 / 6


def test_number_only_variants_land_on_the_same_side_of_the_list():
    from effects.domain.text_holdout import masked_template

    card_files = {f"card {i}": f"cardsfolder/c/card_{i}.txt" for i in range(40)}
    texts_by_card = {
        f"card {i}": [f"SP$ DealDamage | NumDmg$ {i % 4} | ValidTgts$ Kind{i // 4}"]
        for i in range(40)
    }
    listed = set(_report(
        card_files, texts_by_card, permille=500, max_carriers=8,
    ).names)
    by_template: dict[str, set[bool]] = {}
    for card, texts in texts_by_card.items():
        by_template.setdefault(masked_template(texts[0]), set()).add(card in listed)
    assert all(len(sides) == 1 for sides in by_template.values())


def test_no_command_takes_a_holdout_unit():
    """Template is the only unit: ``--holdout-unit`` is no flag, and ``text``
    (feature 023's rule) is refused."""
    from tests.unit.effects.cli_args import parse_cli

    for command in (["holdout-cards", "--out", "x.txt"], ["build-corpus"],
                    ["train-effect-model", "--corpus", "c"]):
        args = parse_cli(command)
        assert not hasattr(args, "holdout_unit")
    with pytest.raises(SystemExit):
        parse_cli(["build-corpus", "--holdout-unit", "template"])


def test_the_cli_takes_the_repeatable_cards_folder_like_every_other_command():
    from effects.infrastructure.cli import build_parser

    args = build_parser().parse_args(["holdout-cards", "--out", "x.txt"])
    assert args.cards_folders == HoldoutCardsConfig(out=Path("x.txt")).cards_folders
    args = build_parser().parse_args([
        "holdout-cards", "--out", "x.txt", "--cards-folder", "a/", "--cards-folder", "b/",
    ])
    assert args.cards_folders == (Path("a/"), Path("b/"))


class TestSplitCardsAreListedByTheirCombinedName:
    """Forge names a split card "Fire // Ice" and depletes by exact
    name, but the list wrote the converted first face ("fire"), so a held-out
    split card stayed in every depleted pool."""

    _FIRE = "SP$ DealDamage | NumDmg$ 2 | ValidTgts$ Any | Bespoke$ Fire"

    def test_the_report_writes_the_combined_printed_name(self) -> None:
        listed = _report(
            {"fire": "cardsfolder/f/fire_ice.txt"},
            {"fire": [self._FIRE]},
            permille=1000, max_carriers=8,
            printings=_Printings(
                canonical_names={"Fire // Ice": "2001-01-01", "Fire": "2001-01-01"},
                split_faces={"fire": ("fire", "ice")},
            ),
        ).names

        assert listed == ["Fire // Ice"]

    def test_a_held_out_split_card_on_disk_is_listed_combined(self, tmp_path) -> None:
        _convert(tmp_path, "cardsfolder/f/fire_ice.txt", "fire", [self._FIRE])
        (tmp_path / "cardsfolder" / "f" / "fire_ice.txt").write_text(
            "layout: split\nname: fire\ntypes: instant\n\nALTERNATE\n\n"
            "name: ice\ntypes: instant\n",
            encoding="utf-8",
        )

        listed = _run(tmp_path, tmp_path / "holdout-cards.txt", permille=1000).names

        assert listed == ["fire // ice"]

    def test_the_collectors_read_the_combined_name_as_the_converted_card(
        self, tmp_path,
    ) -> None:
        """A collector keys cards by the converted first face."""
        from effects.application.collector_common import read_name_list

        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Fire // Ice\nShock\n", encoding="utf-8")

        names = read_name_list(listed)

        assert "fire" in names
        assert "shock" in names
        assert "ice" not in names
