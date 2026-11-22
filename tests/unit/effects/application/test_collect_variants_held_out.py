"""The variants a run keeps: only those Forge loads, and none printing a
held-out text (T138, G14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.collect_variants import (
    CollectVariantsConfig,
    GeneratedVariant,
    loadable_variants,
)
from effects.domain.split import HoldoutRule


class TestOnlyLoadableVariantsAreDecked:
    """A perturbation can produce a script Forge declines to load.

    The script is still written and still gets a sidecar, so nothing before
    the card database can tell it apart. A decks-only round refuses a deck
    naming a card Forge does not know rather than playing basics, so an
    unfiltered deck build is not a rounding error: at roughly one variant in
    nine unloadable, almost every deck of 23 nonlands holds at least one. A
    real 4,000-deck round lost 350 of its 354 workers to exactly that.
    """

    def _generated(self, tmp_path, names):
        return [
            GeneratedVariant(name=name, source_card="x",
                             path=tmp_path / "x.txt", perturbation="numeric")
            for name in names
        ]

    def test_a_variant_forge_rejected_is_not_decked(self, tmp_path):
        (tmp_path / "loadable.lst").write_text(
            "Bolt Variant 0" + chr(10) + "Shock Variant 2" + chr(10),
            encoding="utf-8",
        )
        kept = loadable_variants(
            self._generated(
                tmp_path,
                ["Bolt Variant 0", "Broken Variant 1", "Shock Variant 2"],
            ),
            tmp_path,
        )
        assert [v.name for v in kept] == ["Bolt Variant 0", "Shock Variant 2"]

    def test_the_list_is_no_card_script(self):
        """Python R1: Forge loads every ``.txt`` staged from the variant tree as
        a card, and a list of names with no ``Name:`` line stops every Forge
        startup on the machine."""
        from effects.application.collect_variants import LOADABLE_FILE

        # The property, not the spelling: the sibling below pins the spelling
        # to the Java writer, which would follow any rename to a ``.txt``.
        assert not LOADABLE_FILE.lower().endswith(".txt")

    def test_the_list_is_the_one_variant_sidecar_main_writes(self):
        import re

        from effects.application.collect_variants import LOADABLE_FILE

        java = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main"
                / "java" / "com" / "pricepredictor" / "connector" / "VariantSidecarMain.java")
        declared = re.search(r'LOADABLE_FILE\s*=\s*"([^"]+)"', java.read_text(encoding="utf-8"))
        assert declared is not None and declared.group(1) == LOADABLE_FILE

    def test_a_missing_list_is_refused_rather_than_ignored(self, tmp_path):
        """No list is no evidence, and decking anyway is the defect itself."""
        with pytest.raises(FileNotFoundError, match="loadable"):
            loadable_variants(self._generated(tmp_path, ["Bolt Variant 0"]),
                              tmp_path)


class TestNoVariantPrintsAHeldOutText:
    """A perturbation can land on a held-out text, or under template on its template.

    Its source card is not held out, so the name test passes it; nothing else
    would keep a held-out text's mechanics out of training (G14).
    """

    _HELD = "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 2"

    def _variant(self, tmp_path, name: str, text: str) -> GeneratedVariant:
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.infrastructure.sidecar_io import sidecar_path_for
        from tests.unit.effects.builders import write_sidecar

        path = tmp_path / f"{name}.txt"
        path.write_text(f"Name:{name}\n", encoding="utf-8")
        key = ProvenanceKey(f"variant-scripts/{name}.txt", 0, "spell", 0)
        write_sidecar(
            ProvenanceSidecar(
                card=name, script_file=f"variant-scripts/{name}.txt",
                lines=(SidecarLine(
                    line_index=0, line_kind="spell", provenance=(key,), script_text=text,
                    prose_offset=0,
                ),),
            ),
            sidecar_path_for(path),
        )
        return GeneratedVariant(name=name, source_card="x", path=path, perturbation="numeric")

    def _holdout(self):
        from effects.domain.split import text_keyed_holdout

        return text_keyed_holdout(
            {"Held": "cardsfolder/h/held.txt"}, {"Held": [self._HELD]},
            HoldoutRule(1000, 8),
        )

    def test_a_variant_printing_a_held_out_template_is_dropped_with_its_files(self, tmp_path):
        from effects.application.collect_variants import drop_held_out_variants
        from effects.infrastructure.sidecar_io import sidecar_path_for

        held = self._variant(
            tmp_path, "act_of_treason_variant_1",
            "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        )
        clean = self._variant(tmp_path, "shock_variant_2", "SP$ Draw | NumCards$ 2")

        kept = drop_held_out_variants([held, clean], self._holdout())

        assert kept == [clean]
        assert not held.path.exists()
        assert not sidecar_path_for(held.path).exists()

    def test_a_reused_variant_printing_a_held_out_text_stays_on_disk(self, tmp_path):
        """FR-058b: only a variant no deck has held yet is deleted. A reused
        one may have records from an earlier run, which join its file by key."""
        import dataclasses

        from effects.application.collect_variants import drop_held_out_variants
        from effects.infrastructure.sidecar_io import sidecar_path_for

        held = dataclasses.replace(self._variant(
            tmp_path, "act_of_treason_variant_1",
            "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        ), fresh=False)

        assert drop_held_out_variants([held], self._holdout()) == []
        assert held.path.exists() and sidecar_path_for(held.path).exists()

    def test_the_rule_the_collector_is_given_decides_what_is_dropped(self, tmp_path):
        """A6: under ``build-corpus --holdout-permille 1000`` this text is held
        out, under the default it is not; a collector applying the default
        would deck it into training games' boards."""
        from effects.application.collect_variants import _keep_unheld
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.domain.split import HOLDOUT_PERMILLE
        from effects.domain.text_holdout import masked_template, text_is_held_out
        from effects.infrastructure.sidecar_io import sidecar_path_for
        from tests.unit.effects.builders import write_sidecar

        assert not text_is_held_out(masked_template(self._HELD), permille=HOLDOUT_PERMILLE)
        card = tmp_path / "cardsfolder" / "h" / "held.txt"
        card.parent.mkdir(parents=True)
        card.write_text("name: Held\ntypes: instant\n", encoding="utf-8")
        write_sidecar(
            ProvenanceSidecar(
                card="Held", script_file="cardsfolder/h/held.txt",
                lines=(SidecarLine(
                    line_index=0, line_kind="spell", script_text=self._HELD,
                    provenance=(ProvenanceKey("cardsfolder/h/held.txt", 0, "spell", 0),),
                    prose_offset=0,
                ),),
            ),
            sidecar_path_for(card),
        )
        variants = tmp_path / "variant-scripts"
        variants.mkdir()

        def config(**rule) -> CollectVariantsConfig:
            return CollectVariantsConfig(
                effect_records=tmp_path / "records", variant_scripts=variants,
                cards_folders=(tmp_path / "cardsfolder",), holdout=HoldoutRule(**rule),
            )

        printed = self._variant(
            variants, "held_variant_1", "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        )
        clean = self._variant(variants, "draw_variant_2", "SP$ Draw | NumCards$ 2")
        assert _keep_unheld(config(), [printed, clean]) == [printed, clean]
        assert _keep_unheld(config(permille=1000), [printed, clean]) == [clean]
