"""Synthetic script variants (T138).

Four properties: a variant record carries ``synthetic`` and ``variant_of``, a
variant of a held-out card is held out **with it**, variants contribute no
pairing loss, and no variant is ever converted to prose.

The held-out rule is the one that would be silently wrong: a variant of a
held-out card puts that card's mechanics in front of the model under a different
name, and the card-disjoint split would stop meaning "deployment to an unseen
set" without anything looking broken.
"""

from __future__ import annotations

import random

import pytest

from effects.application.collect_variants import (
    DEFAULT_VARIANT_VOLUME,
    CollectVariantsConfig,
    GeneratedVariant,
    generate_variants,
    perturb_script,
    variant_budget,
)
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.script_variants import (
    MAX_NUMERIC_SHIFT,
    NUMERIC_PARAM_KEYS,
    SELECTOR_WHITELIST,
    PerturbationKind,
    double,
    numeric_params,
    perturb,
    selector_params,
    shift,
    variant_name,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)

_BOLT_SCRIPT = (
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target."
)


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "r.0.1", "run_id": "r", "timestamp": "t", "game_id": "r.0.1",
        "kind": RecordKind.RESOLUTION, "moment": Moment.RESOLUTION,
        "actor_player": "P0", "state": _SNAPSHOT,
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestVariantRecordFlags:
    def test_a_variant_record_is_synthetic_and_names_its_source(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        assert record.synthetic
        assert record.variant_of == "Lightning Bolt"

    def test_variant_of_requires_the_synthetic_flag(self):
        with pytest.raises(ValueError, match="variant_of"):
            _record(variant_of="Lightning Bolt")

    def test_the_synthetic_flag_never_reaches_the_model(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        fields = record.model_input_fields()
        assert "synthetic" not in fields
        assert "variant_of" not in fields

    def test_a_variant_is_otherwise_an_ordinary_record(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        assert record.kind is RecordKind.RESOLUTION
        assert set(record.model_input_fields()) == {
            "kind", "moment", "subkind", "actor_player", "ability", "state",
            "payload",
        }


class TestNumericPerturbation:
    def test_a_shift_moves_the_value_by_at_most_three(self):
        rng = random.Random(0)
        for _ in range(50):
            assert abs(shift(10, rng) - 10) <= MAX_NUMERIC_SHIFT

    def test_a_shift_is_floored_at_zero(self):
        """A negative count is not a smaller effect, it is a script Forge
        rejects — and a rejected script never reaches a game."""
        rng = random.Random(0)
        for _ in range(50):
            assert shift(1, rng) >= 0

    def test_doubling_keeps_the_card_recognisably_the_same(self):
        assert double(3) == 6
        assert double(0) == 0

    def test_the_numeric_keys_are_a_whitelist(self):
        """A shifted CounterType would produce a script that fails to load
        rather than a card that plays differently."""
        assert "NumDmg" in NUMERIC_PARAM_KEYS
        assert "CounterType" not in NUMERIC_PARAM_KEYS
        assert "ValidTgts" not in NUMERIC_PARAM_KEYS

    def test_numeric_params_reads_only_whitelisted_numbers(self):
        found = numeric_params(_BOLT_SCRIPT)
        assert found == {"NumDmg": 3}

    def test_a_perturbed_script_differs_in_exactly_one_parameter(self):
        rng = random.Random(1)
        result = perturb(_BOLT_SCRIPT, rng)
        assert result is not None
        perturbed, perturbation = result
        assert perturbed != _BOLT_SCRIPT
        assert perturbation.key == "NumDmg"
        # Every other parameter survives untouched.
        assert "ValidTgts$ Any" in perturbed
        assert "Cost$ R" in perturbed

    def test_a_line_with_nothing_to_perturb_is_skipped(self):
        assert perturb("Name:Lightning Bolt", random.Random(0)) is None
        assert perturb("Types:Instant", random.Random(0)) is None


class TestSelectorPerturbation:
    def test_a_whitelisted_selector_is_swappable(self):
        script = "Mode$ Continuous | Affected$ Creature | AddPower$ 1"
        assert selector_params(script) == {"Affected": "Creature"}

    def test_a_swap_produces_a_selector_forge_recognises(self):
        script = "Mode$ Continuous | Affected$ Creature | AddPower$ 1"
        rng = random.Random(7)
        for _ in range(20):
            result = perturb(script, rng)
            if result is None:
                continue
            perturbed, perturbation = result
            if perturbation.kind == PerturbationKind.SELECTOR:
                assert perturbation.replacement in SELECTOR_WHITELIST
                assert perturbation.replacement != perturbation.original
                assert perturbation.replacement in perturbed

    def test_the_whitelist_holds_only_composable_restrictions(self):
        for selector in SELECTOR_WHITELIST:
            assert " " not in selector
            assert selector[0].isupper()


class TestVariantGeneration:
    def _write_source(self, folder, stem, name, script):
        (folder / f"{stem}.txt").write_text(
            f"Name:{name}\nManaCost:R\nTypes:Instant\n{script}\n"
            f"Oracle:{name} deals 3 damage.\n",
            encoding="utf-8",
        )

    def test_a_variant_is_written_with_its_own_name(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=5,
        )
        assert variants
        assert variants[0].source_card == "Lightning Bolt"
        assert "Lightning Bolt" in variants[0].name
        assert variants[0].name != "Lightning Bolt"
        assert f"Name:{variants[0].name}" in variants[0].path.read_text(
            encoding="utf-8"
        )

    def test_a_variant_of_a_held_out_card_is_held_out_with_it(self, tmp_path):
        """Otherwise a held-out card's mechanics reach the model under another
        name, and the card-disjoint split stops meaning what it says."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants",
            held_out=frozenset({"Lightning Bolt"}), limit=5,
        )
        assert variants == []

    def test_a_variant_is_never_converted_to_prose(self, tmp_path):
        """There is no oracle text for a card nobody printed."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        output = tmp_path / "variants"
        generate_variants(source, output, held_out=frozenset(), limit=5)
        # The variant tree holds Forge source scripts. The sidecar beside them
        # is written by the Java parser, not here.
        for path in output.glob("*.txt"):
            assert path.read_text(encoding="utf-8").startswith("Name:")

    def test_a_variant_file_is_named_for_the_key_that_resolves_it(
        self, tmp_path,
    ):
        """The runtime keys a variant as ``variant-scripts/{sanitized name}.txt``
        and the reader looks for a sidecar at exactly that path, so a filename
        derived from anything else makes every variant record unjoinable."""
        from price_predictor.infrastructure.card_filenames import (
            sanitize_card_name,
        )

        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=1,
        )
        assert variants
        assert variants[0].path.name == (
            f"{sanitize_card_name(variants[0].name)}.txt"
        )

    def test_a_variant_name_survives_sanitizing_without_punctuation(self):
        """A parenthesised suffix reads as a set code on a Forge deck-list line
        and survives both filename sanitizers verbatim."""
        from price_predictor.infrastructure.card_filenames import (
            sanitize_card_name,
        )

        stem = sanitize_card_name(variant_name("Lightning Bolt", 0))
        assert stem == "lightning_bolt_variant_0"
        assert "(" not in stem and ")" not in stem

    def test_the_limit_is_respected(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(10):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=3,
        )
        assert len(variants) == 3

    def test_generation_is_reproducible_under_a_seed(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(5):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        first = generate_variants(
            source, tmp_path / "a", held_out=frozenset(), limit=3, seed=5,
        )
        second = generate_variants(
            source, tmp_path / "b", held_out=frozenset(), limit=3, seed=5,
        )
        assert [v.name for v in first] == [v.name for v in second]

    def test_a_card_with_no_perturbable_line_is_skipped(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        (source / "bear.txt").write_text(
            "Name:Grizzly Bears\nManaCost:1 G\nTypes:Creature Bear\nPT:2/2\n",
            encoding="utf-8",
        )
        assert generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=5,
        ) == []

    def test_perturb_script_finds_the_first_perturbable_line(self):
        lines = ["Name:Bolt", "Types:Instant", _BOLT_SCRIPT]
        result = perturb_script(lines, random.Random(3))
        assert result is not None
        perturbed, _description = result
        assert perturbed[0] == "Name:Bolt"
        assert perturbed[2] != _BOLT_SCRIPT


class TestVolumeCap:
    def test_the_budget_is_a_fraction_of_the_real_corpus(self):
        """Expressed against the corpus so it scales with it, rather than
        needing a new absolute every time the corpus grows."""
        assert variant_budget(1000, 0.2) == 200
        assert variant_budget(0, 0.2) == 0

    def test_the_default_volume_is_the_contract(self):
        assert DEFAULT_VARIANT_VOLUME == 0.2
        assert CollectVariantsConfig().variant_volume == 0.2

    def test_a_larger_volume_allows_more(self):
        assert variant_budget(1000, 0.5) > variant_budget(1000, 0.2)


class TestPairingLoss:
    def test_a_variant_has_only_the_script_surface(self):
        """So it contributes no pairing term: there is no prose to pair with."""
        from effects.domain.ability_encoder import (
            SURFACE_PROSE,
            SURFACE_SCRIPT,
            encoding_text,
        )

        class _Line:
            script_text = "NumDmg$ 7"

        line = _Line()
        assert encoding_text(line, None, SURFACE_SCRIPT) == "NumDmg$ 7"
        # No prose exists, so the prose surface falls back to the script rather
        # than pairing two different readings of one line.
        assert encoding_text(line, None, SURFACE_PROSE) == "NumDmg$ 7"

    def test_a_real_card_has_both_surfaces_and_can_pair(self):
        from effects.domain.ability_encoder import (
            SURFACE_PROSE,
            SURFACE_SCRIPT,
            encoding_text,
        )

        class _Line:
            script_text = "NumDmg$ 3"

        line = _Line()
        prose = "deal 3 damage to any target"
        assert encoding_text(line, prose, SURFACE_SCRIPT) == "NumDmg$ 3"
        assert encoding_text(line, prose, SURFACE_PROSE) == prose


def test_variant_name_derives_from_its_source():
    assert variant_name("Lightning Bolt", 0) == "Lightning Bolt Variant 0"
    assert variant_name("Lightning Bolt", 1) != variant_name("Lightning Bolt", 0)


class TestRunClosesTheSupervisor:
    """final-fix-3.md item 5: same finding, same fix shape as
    ``collect_coverage.run`` (see ``TestRunClosesTheSupervisor`` in
    ``test_coverage.py`` for the full rationale) -- ``run()`` built a
    ``CollectorSupervisor`` and played a round through it, but never called
    ``.stop()``, so the worker log handles opened for F4's latched
    effect-record failure reporters stayed open until the interpreter exited
    on its own.
    """

    def _config(self, tmp_path, **overrides):
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        variant_scripts = tmp_path / "variant-scripts"
        return CollectVariantsConfig(
            effect_records=tmp_path / "records",
            forge_cards_path=forge_cards,
            variant_scripts=variant_scripts,
            **overrides,
        )

    def _patch_prerequisites(self, monkeypatch, variant):
        """Everything ``run()`` needs before it ever reaches the supervisor,
        stood in for rather than exercised for real: the corpus size, the
        variant generator (which otherwise reads real Forge card scripts) and
        the sidecar writer (which otherwise needs a real JVM)."""
        from unittest.mock import MagicMock

        import effects.application.collect_variants as collect_variants
        import effects.infrastructure.record_io as record_io
        import effects.infrastructure.variant_sidecar_connector as sidecar_module

        monkeypatch.setattr(record_io, "count_records", lambda path: 1000)
        monkeypatch.setattr(
            collect_variants, "generate_variants",
            lambda *a, **k: [variant],
        )
        sidecar = MagicMock()
        sidecar.run.return_value = 0
        monkeypatch.setattr(
            sidecar_module, "VariantSidecarConnector",
            MagicMock(return_value=sidecar),
        )
        return collect_variants

    def test_stop_runs_after_an_ordinary_round(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import effects.infrastructure.collector_connector as collector_connector

        variant = GeneratedVariant(
            name="Lightning Bolt Variant 0", source_card="Lightning Bolt",
            path=tmp_path / "variant-scripts" / "lightning_bolt_variant_0.txt",
            perturbation="numeric",
        )
        collect_variants = self._patch_prerequisites(monkeypatch, variant)
        supervisor = MagicMock()
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        code = collect_variants.run(self._config(tmp_path))

        assert code == 0
        supervisor.play_round.assert_called_once()
        supervisor.stop.assert_called_once()

    def test_stop_runs_even_if_the_round_raises(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import effects.infrastructure.collector_connector as collector_connector

        variant = GeneratedVariant(
            name="Lightning Bolt Variant 0", source_card="Lightning Bolt",
            path=tmp_path / "variant-scripts" / "lightning_bolt_variant_0.txt",
            perturbation="numeric",
        )
        collect_variants = self._patch_prerequisites(monkeypatch, variant)
        supervisor = MagicMock()
        supervisor.play_round.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        with pytest.raises(RuntimeError, match="boom"):
            collect_variants.run(self._config(tmp_path))

        supervisor.stop.assert_called_once()
