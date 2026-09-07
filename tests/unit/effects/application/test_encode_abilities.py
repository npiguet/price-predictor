"""The ability cache as a command (T085).

Row alignment against the sidecar, one subtree per source tree, variant files
written beside the shipping cache rather than over it, and idempotence — a
second run of the same checkpoint must produce the same bytes, because every
downstream consumer reads the cache rather than the encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from effects.application.encode_abilities import (
    EncodeAbilitiesConfig,
    encode_sidecar,
    iter_sidecars,
    run,
    taxonomy_vector,
    tree_of,
)
from effects.domain.ability_cache_layout import ARRAY_KEY, cache_path_for
from effects.domain.ability_encoder import AbilityEncoderConfig
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.infrastructure.ability_cache_store import AbilityCacheStore
from effects.infrastructure.effect_model_store import (
    EffectCheckpoint,
    EffectModelStore,
    HashMismatchError,
    SplitProvenance,
    content_hash,
)
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar

E_DIM = 8

_MODEL_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
    e_dim=E_DIM, d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)


def _sidecar(script_file: str, n_lines: int = 2) -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card=script_file.rsplit("/", 1)[-1].removesuffix(".txt"),
        script_file=script_file,
        lines=tuple(
            SidecarLine(
                line_index=4 + i, line_kind="static",
                provenance=(ProvenanceKey(script_file, 0, "static", i),),
                script_api_type="Pump" if i == 0 else "PutCounter",
                script_param_keys=("Defined", "NumAtt") if i == 0 else ("CounterType",),
                script_text="Mode$ Continuous | AddPower$ 1",
            )
            for i in range(n_lines)
        ),
    )


@pytest.fixture
def corpus(tmp_path):
    """Two converted trees with sidecars, plus a checkpoint to read facts from."""
    cards = tmp_path / "cardsfolder"
    (cards / "s").mkdir(parents=True)
    write_sidecar(
        _sidecar("cardsfolder/s/serra_angel.txt", 3),
        sidecar_path_for(cards / "s" / "serra_angel.txt"),
    )
    tokens = tmp_path / "tokenscripts"
    tokens.mkdir()
    write_sidecar(
        _sidecar("tokenscripts/soldier.txt", 1),
        sidecar_path_for(tokens / "soldier.txt"),
    )

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    keywords = tmp_path / "kw.json"
    keywords.write_text("{}", encoding="utf-8")

    torch.manual_seed(0)
    store = EffectModelStore(tmp_path / "models" / "effect-model")
    store.save(EffectCheckpoint(
        encoder_config=AbilityEncoderConfig(
            vocab_size=32, e_dim=E_DIM, d_model=16, n_layers=1, n_heads=2,
            ff_dim=32,
        ),
        model_config=_MODEL_CONFIG,
        encoder_state={},
        model_state=EffectModel(_MODEL_CONFIG).state_dict(),
        provenance=SplitProvenance(
            vocab_path=str(vocab), keyword_definitions_path=str(keywords),
            vocab_hash=content_hash(vocab),
            keyword_definitions_hash=content_hash(keywords),
        ),
    ))
    return cards, tokens, store.latest_path(), vocab


def _config(corpus, tmp_path, **overrides) -> EncodeAbilitiesConfig:
    cards, tokens, checkpoint, _vocab = corpus
    defaults = {
        "cards_folders": (cards, tokens),
        "checkpoint": checkpoint,
        "output_root": tmp_path / "abilities",
        "variant": "taxonomy",
    }
    defaults.update(overrides)
    return EncodeAbilitiesConfig(**defaults)


class TestRowAlignment:
    def test_a_source_gets_one_row_per_sidecar_line(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        assert store.read("cardsfolder/s/serra_angel.txt").shape == (3, E_DIM)
        assert store.read("tokenscripts/soldier.txt").shape == (1, E_DIM)

    def test_a_row_resolves_from_its_provenance_key(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        sidecar = _sidecar("cardsfolder/s/serra_angel.txt", 3)
        key = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "static", 1)
        vector = store.vector_for(key, sidecar)
        np.testing.assert_allclose(
            vector, store.read("cardsfolder/s/serra_angel.txt")[1],
        )

    def test_rows_are_float32(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        assert store.read("cardsfolder/s/serra_angel.txt").dtype == np.float32

    def test_a_dropped_key_resolves_to_no_vector(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        script = "cardsfolder/s/serra_angel.txt"
        sidecar = ProvenanceSidecar(
            card="x", script_file=script,
            lines=_sidecar(script, 3).lines,
            dropped_keys=(ProvenanceKey(script, 0, "spell", 0),),
        )
        assert store.vector_for(
            ProvenanceKey(script, 0, "spell", 0), sidecar,
        ) is None


class TestTreeSubtrees:
    def test_each_source_tree_gets_its_own_subtree(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        root = config.output_root
        assert (root / "cardsfolder" / "s" / "serra_angel.taxonomy.npz").exists()
        assert (root / "tokenscripts" / "soldier.taxonomy.npz").exists()

    def test_the_subtree_mirrors_the_source_layout(self, corpus, tmp_path):
        """Letter-keyed on one side, flat on the other."""
        config = _config(corpus, tmp_path)
        run(config)
        card = cache_path_for(
            "cardsfolder/s/serra_angel.txt", root=config.output_root,
            variant="taxonomy",
        )
        token = cache_path_for(
            "tokenscripts/soldier.txt", root=config.output_root,
            variant="taxonomy",
        )
        assert card.parent.name == "s"
        assert token.parent.name == "tokenscripts"

    def test_tree_of_reads_the_folder_name(self, tmp_path):
        assert tree_of(tmp_path / "tokenscripts") == "tokenscripts"

    def test_sidecars_are_found_recursively(self, corpus, tmp_path):
        cards, _tokens, _checkpoint, _vocab = corpus
        assert len(iter_sidecars(cards)) == 1


class TestVariantIsolation:
    def test_a_variant_writes_beside_the_shipping_cache(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        root = config.output_root
        assert (root / "cardsfolder" / "s" / "serra_angel.taxonomy.npz").exists()
        assert not (root / "cardsfolder" / "s" / "serra_angel.npz").exists()

    def test_the_variant_resolves_its_own_checkpoint(self, tmp_path):
        config = EncodeAbilitiesConfig(variant="no-state")
        assert config.resolved_checkpoint().parent.name == "no-state"

    def test_full_resolves_the_shipping_checkpoint(self):
        config = EncodeAbilitiesConfig(variant="full")
        assert config.resolved_checkpoint().parent.name == "effect-model"

    def test_an_explicit_checkpoint_overrides(self, tmp_path):
        chosen = tmp_path / "elsewhere.pt"
        assert EncodeAbilitiesConfig(
            variant="no-state", checkpoint=chosen,
        ).resolved_checkpoint() == chosen

    def test_clean_removes_only_this_variants_files(self, corpus, tmp_path):
        root = tmp_path / "abilities"
        run(_config(corpus, tmp_path, variant="taxonomy", output_root=root))
        shipping = root / "cardsfolder" / "s" / "serra_angel.npz"
        shipping.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shipping, **{ARRAY_KEY: np.zeros((3, E_DIM))})

        removed = AbilityCacheStore(root, variant="taxonomy").clean()
        assert removed == 2
        assert shipping.exists()

    def test_cleaning_the_shipping_cache_leaves_variants_alone(
        self, corpus, tmp_path,
    ):
        root = tmp_path / "abilities"
        run(_config(corpus, tmp_path, variant="taxonomy", output_root=root))
        shipping = root / "cardsfolder" / "s" / "serra_angel.npz"
        shipping.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shipping, **{ARRAY_KEY: np.zeros((3, E_DIM))})

        assert AbilityCacheStore(root, variant="full").clean() == 1
        assert (root / "cardsfolder" / "s" / "serra_angel.taxonomy.npz").exists()


class TestIdempotence:
    def test_a_second_run_produces_the_same_rows(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        first = store.read("cardsfolder/s/serra_angel.txt").copy()
        run(config)
        second = AbilityCacheStore(
            config.output_root, variant="taxonomy",
        ).read("cardsfolder/s/serra_angel.txt")
        np.testing.assert_array_equal(first, second)

    def test_the_summary_counts_what_it_wrote(self, corpus, tmp_path):
        summary = run(_config(corpus, tmp_path))
        assert summary.sources == 2
        assert summary.rows == 4


class TestTaxonomyVariant:
    def test_it_emits_into_the_same_row_layout_without_an_encoder(self):
        """Every e-geometry check reads one file shape (FR-102)."""
        sidecar = _sidecar("cardsfolder/s/serra_angel.txt", 3)
        matrix = encode_sidecar(sidecar, variant="taxonomy", e_dim=E_DIM)
        assert matrix.shape == (3, E_DIM)

    def test_the_vector_is_a_function_of_the_api_type_and_params(self):
        first = SidecarLine(
            line_index=0, line_kind="static", provenance=(),
            script_api_type="Pump", script_param_keys=("Defined",),
        )
        same = SidecarLine(
            line_index=9, line_kind="triggered", provenance=(),
            script_api_type="Pump", script_param_keys=("Defined",),
        )
        different = SidecarLine(
            line_index=0, line_kind="static", provenance=(),
            script_api_type="Destroy", script_param_keys=("Defined",),
        )
        np.testing.assert_allclose(
            taxonomy_vector(first, E_DIM), taxonomy_vector(same, E_DIM),
        )
        assert not np.allclose(
            taxonomy_vector(first, E_DIM), taxonomy_vector(different, E_DIM),
        )

    def test_a_line_with_no_script_facts_still_yields_a_vector(self):
        bare = SidecarLine(line_index=0, line_kind="text", provenance=())
        assert taxonomy_vector(bare, E_DIM).shape == (E_DIM,)

    def test_another_variant_without_an_encoder_is_rejected(self):
        sidecar = _sidecar("cardsfolder/s/serra_angel.txt", 1)
        with pytest.raises(ValueError, match="needs an encoder"):
            encode_sidecar(sidecar, variant="full", e_dim=E_DIM)

    def test_a_source_with_no_lines_yields_an_empty_matrix(self):
        empty = ProvenanceSidecar(card="x", script_file="cardsfolder/x.txt", lines=())
        assert encode_sidecar(
            empty, variant="taxonomy", e_dim=E_DIM,
        ).shape == (0, E_DIM)


class TestTheRealEncoder:
    """The shipping variants read a trained encoder, not a hash embedding.

    Without this the only variant that can produce a cache is ``taxonomy``, the
    baseline — and the artifact the whole feature exists to ship cannot be
    built at all.
    """

    @pytest.fixture
    def trained(self, tmp_path):
        """A corpus plus a checkpoint carrying real encoder weights."""
        cards = tmp_path / "cardsfolder"
        (cards / "s").mkdir(parents=True)
        write_sidecar(
            _sidecar("cardsfolder/s/serra_angel.txt", 3),
            sidecar_path_for(cards / "s" / "serra_angel.txt"),
        )
        (cards / "s" / "serra_angel.txt").write_text(
            "name: serra angel\nstatic[0]: creatures you control get +1/+1\n"
            "static[1]: flying\nstatic[2]: vigilance\nstatic[3]: whatever\n"
            "static[4]: whatever\n",
            encoding="utf-8",
        )
        vocab = tmp_path / "vocab.txt"
        vocab.write_text(
            "\n".join([
                "[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "creatures",
                "you", "control", "get", "flying", "vigilance", "1",
            ]) + "\n",
            encoding="utf-8",
        )
        keywords = tmp_path / "kw.json"
        keywords.write_text("{}", encoding="utf-8")

        torch.manual_seed(0)
        encoder_config = AbilityEncoderConfig(
            vocab_size=12, e_dim=E_DIM, d_model=16, n_layers=1, n_heads=2,
            ff_dim=32,
        )
        from effects.domain.ability_encoder import AbilityEncoder

        store = EffectModelStore(tmp_path / "models" / "effect-model")
        store.save(EffectCheckpoint(
            encoder_config=encoder_config,
            model_config=_MODEL_CONFIG,
            encoder_state=AbilityEncoder(encoder_config).state_dict(),
            model_state=EffectModel(_MODEL_CONFIG).state_dict(),
            provenance=SplitProvenance(
                vocab_path=str(vocab), keyword_definitions_path=str(keywords),
                vocab_hash=content_hash(vocab),
                keyword_definitions_hash=content_hash(keywords),
            ),
        ))
        return EncodeAbilitiesConfig(
            cards_folders=(cards,),
            checkpoint=store.latest_path(),
            output_root=tmp_path / "abilities",
            variant="full",
        )

    def test_the_shipping_variant_writes_a_cache(self, trained):
        summary = run(trained)
        assert summary.sources == 1
        assert summary.rows == 3
        store = AbilityCacheStore(trained.output_root, variant="full")
        assert store.read("cardsfolder/s/serra_angel.txt").shape == (3, E_DIM)

    def test_the_rows_are_not_all_the_same(self, trained):
        """A cache of one repeated vector would pass every shape check and
        carry no information."""
        run(trained)
        rows = AbilityCacheStore(trained.output_root, variant="full").read(
            "cardsfolder/s/serra_angel.txt"
        )
        assert not np.allclose(rows[0], rows[1]) or not np.allclose(
            rows[1], rows[2]
        )

    def test_a_second_run_reproduces_the_same_bytes(self, trained):
        """The encoder adds noise to e while training and drops out; a cache
        built without eval() would differ run to run and every consumer reads
        the cache rather than the model."""
        run(trained)
        first = AbilityCacheStore(
            trained.output_root, variant="full",
        ).read("cardsfolder/s/serra_angel.txt").copy()
        run(trained)
        second = AbilityCacheStore(
            trained.output_root, variant="full",
        ).read("cardsfolder/s/serra_angel.txt")
        np.testing.assert_array_equal(first, second)


class TestHashCheck:
    def test_a_moved_vocabulary_stops_the_run_before_writing(
        self, corpus, tmp_path,
    ):
        _cards, _tokens, _checkpoint, vocab = corpus
        config = _config(corpus, tmp_path)
        vocab.write_text("[PAD]\n[UNK]\nmoved\n", encoding="utf-8")
        with pytest.raises(HashMismatchError):
            run(config)
        assert not (config.output_root).exists() or not list(
            config.output_root.rglob("*.npz")
        )
