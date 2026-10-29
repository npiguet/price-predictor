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
    encode_tree,
    read_sources,
    run,
    taxonomy_matrix,
    taxonomy_vector,
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

    def test_rows_are_float32(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        assert store.read("cardsfolder/s/serra_angel.txt").dtype == np.float32


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

    def test_a_missing_source_tree_is_refused_rather_than_skipped(self, corpus, tmp_path):
        """A relative default resolved from the wrong directory would otherwise
        encode no card at all and still report success."""
        cards, _tokens, _checkpoint, _vocab = corpus
        config = _config(corpus, tmp_path, cards_folders=(cards, tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="absent"):
            run(config)

    def test_sidecars_are_found_recursively(self, corpus, tmp_path):
        cards, _tokens, _checkpoint, _vocab = corpus
        assert len(read_sources([cards])[0]) == 1


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


class TestReadingTheCache:
    def test_a_variant_iterates_only_its_own_files(self, corpus, tmp_path):
        root = tmp_path / "abilities"
        run(_config(corpus, tmp_path, variant="taxonomy", output_root=root))
        shipping = root / "cardsfolder" / "s" / "serra_angel.npz"
        np.savez_compressed(shipping, **{ARRAY_KEY: np.zeros((5, E_DIM))})

        shapes = {
            path.name: matrix.shape
            for path, matrix in AbilityCacheStore(root).iter_matrices()
        }
        assert shapes == {"serra_angel.npz": (5, E_DIM)}

    def test_a_source_with_no_cache_reads_none(self, tmp_path):
        assert AbilityCacheStore(tmp_path).read("cardsfolder/a/absent.txt") is None


class TestIdempotence:
    def test_a_second_run_produces_the_same_rows(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = AbilityCacheStore(config.output_root, variant="taxonomy")
        first = store.read("cardsfolder/s/serra_angel.txt").copy()
        run(config)
        second = store.read("cardsfolder/s/serra_angel.txt")
        np.testing.assert_array_equal(first, second)

    def test_the_summary_counts_what_it_wrote(self, corpus, tmp_path):
        summary = run(_config(corpus, tmp_path))
        assert summary.sources == 2
        assert summary.rows == 4


class TestTaxonomyBaseline:
    def test_it_emits_into_the_same_row_layout_without_an_encoder(self):
        """Every e-geometry check reads one file shape (FR-102)."""
        sidecar = _sidecar("cardsfolder/s/serra_angel.txt", 3)
        matrix = taxonomy_matrix(sidecar, E_DIM)
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

    def test_a_source_with_no_lines_yields_an_empty_matrix(self):
        empty = ProvenanceSidecar(card="x", script_file="cardsfolder/x.txt", lines=())
        assert taxonomy_matrix(empty, E_DIM).shape == (0, E_DIM)


class TestOneEncodeForTheTree:
    """The encoder sees each distinct input once for the whole tree, not per card."""

    class _Runner:
        def __init__(self) -> None:
            self.calls: list[list] = []

        def unit_of(self, line, prose):
            return line.script_text, ()

        def encode(self, units, keys_of):
            self.calls.append(list(units))
            return np.arange(len(units) * E_DIM, dtype=np.float32).reshape(-1, E_DIM)

    @staticmethod
    def _card(name: str, texts: list[str]) -> ProvenanceSidecar:
        script_file = f"variant-scripts/{name}.txt"
        return ProvenanceSidecar(
            card=name, script_file=script_file,
            lines=tuple(
                SidecarLine(
                    line_index=i, line_kind="static",
                    provenance=(ProvenanceKey(script_file, 0, "static", i),),
                    script_text=text,
                )
                for i, text in enumerate(texts)
            ),
        )

    def test_a_line_shared_across_cards_is_encoded_once(self, tmp_path):
        runner = self._Runner()
        sources = [
            (tmp_path / "a.provenance.json", self._card("a", ["Flying", "Draw"])),
            (tmp_path / "b.provenance.json", self._card("b", ["Flying"])),
        ]
        encode_tree(runner, sources)
        assert runner.calls == [[("Flying", ()), ("Draw", ())]]

    def test_rows_scatter_back_in_sidecar_line_order(self, tmp_path):
        runner = self._Runner()
        sources = [
            (tmp_path / "a.provenance.json", self._card("a", ["Flying", "Draw"])),
            (tmp_path / "b.provenance.json", self._card("b", ["Draw", "Flying"])),
        ]
        first, second = encode_tree(runner, sources)
        np.testing.assert_array_equal(first, second[::-1])
        assert not np.array_equal(first[0], first[1])


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
        store = AbilityCacheStore(trained.output_root)
        first = store.read("cardsfolder/s/serra_angel.txt").copy()
        run(trained)
        second = store.read("cardsfolder/s/serra_angel.txt")
        np.testing.assert_array_equal(first, second)

    def test_a_line_past_the_window_is_reported_by_its_key(self, trained, caplog):
        """Spec Story 1 scenario 9 (FR-006)."""
        text = trained.cards_folders[0] / "s" / "serra_angel.txt"
        rows = text.read_text(encoding="utf-8").splitlines()
        rows[5] = "static[4]: " + " ".join(["flying"] * 600)
        text.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            run(trained)
        reported = [r.getMessage() for r in caplog.records if "truncated" in r.getMessage()]
        assert len(reported) == 1
        assert "cardsfolder/s/serra_angel.txt face 0 static[1]" in reported[0]
        assert "512" in reported[0]

    def test_no_line_inside_the_window_is_reported(self, trained, caplog):
        with caplog.at_level("WARNING"):
            run(trained)
        assert "truncated" not in caplog.text


class TestProseOfAVariant:
    def test_a_variants_source_script_is_not_read_as_prose(self, tmp_path):
        """The trainer encodes a variant from its script; the cache must too."""
        from effects.application.encode_abilities import prose_of

        source = tmp_path / "variant_bolt.txt"
        source.write_text(
            "Name:Variant Bolt\nA:SP$ DealDamage | NumDmg$ 4\n", encoding="utf-8",
        )
        variant = _sidecar("variant-scripts/variant_bolt.txt")
        card = _sidecar("cardsfolder/v/variant_bolt.txt")
        sidecar_path = tmp_path / "variant_bolt.provenance.json"
        assert prose_of(variant, sidecar_path) == []
        assert prose_of(card, sidecar_path) == source.read_text(encoding="utf-8").splitlines()


class TestTheWithheldKeyword:
    """The cache reads a withheld keyword the way training did: its definition
    with the token dropped, never a token training never updated (FR-070)."""

    def test_the_runner_encodes_the_keyword_as_its_expansion(self):
        from effects.application.extract_keyword_definitions import KeywordDefinition
        from effects.domain.ability_encoder import AbilityEncoder
        from effects.domain.ability_tokenizer import AbilityTokenizer
        from effects.infrastructure.ability_encoder_runner import AbilityEncoderRunner

        words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "flying", "reach",
                 "cant", "be", "blocked", "except", "by", "creatures", "with", "or"]
        tokenizer = AbilityTokenizer(
            {word: index for index, word in enumerate(words)},
            {"Flying": KeywordDefinition(
                keyword="Flying",
                reminder_template="cant be blocked except by creatures with flying or reach",
            )},
        )
        torch.manual_seed(0)
        config = AbilityEncoderConfig(
            vocab_size=len(words), e_dim=E_DIM, d_model=16, n_layers=1, n_heads=2,
            ff_dim=32,
        )
        state = AbilityEncoder(config).state_dict()

        def runner(withheld):
            return AbilityEncoderRunner(
                config, state, tokenizer, surface="prose",
                device=torch.device("cpu"), withheld_keyword=withheld,
            )

        withheld = runner("Flying").encode([("flying", ())])
        expansion = runner(None).encode(
            [("cant be blocked except by creatures with or reach", ())],
        )
        token = runner(None).encode([("flying", ())])
        assert np.allclose(withheld, expansion, atol=1e-6)
        assert not np.allclose(withheld, token, atol=1e-6)


class TestTruncation:
    """``prepare_line`` says when it cut a line, and a run says so once."""

    @pytest.fixture
    def tokenizer(self):
        from effects.domain.ability_tokenizer import AbilityTokenizer

        return AbilityTokenizer({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[MASK]": 3})

    def test_a_line_inside_the_window_is_not_truncated(self, tokenizer):
        from effects.domain.ability_encoder import prepare_line

        tokens = tokenizer.tokenize(" ".join(["x"] * 511))
        assert prepare_line(tokenizer, tokens).truncated is False

    def test_a_line_past_the_window_is_truncated(self, tokenizer):
        from effects.domain.ability_encoder import MAX_ABILITY_TOKENS, prepare_line

        tokens = tokenizer.tokenize(" ".join(["x"] * 512))
        line = prepare_line(tokenizer, tokens)
        assert line.truncated is True
        assert len(line) == MAX_ABILITY_TOKENS

    def test_training_batchers_sharing_a_log_report_a_text_once(self, tokenizer):
        """The trainer hands one log to every per-batch batcher."""
        from effects.domain.ability_encoder import TruncationLog
        from effects.domain.baselines import BaselineMasks
        from tests.unit.effects.builders import make_batcher

        reported = []
        log = TruncationLog(lambda text, keys: reported.append(text))
        long_text = " ".join(["x"] * 600)
        for _batch in range(3):
            batcher = make_batcher(
                tokenizer=tokenizer, sidecars=None, masks=BaselineMasks(),
                surface="script", e_dim=4, widths={},
                device=torch.device("cpu"), truncations=log,
            )
            batcher.stage({long_text: None, "x x": None})
        assert reported == [long_text]

    def test_a_text_is_reported_once_per_run(self):
        from effects.domain.ability_encoder import TruncationLog

        reported = []
        log = TruncationLog(lambda text, keys: reported.append((text, keys)))
        key = ProvenanceKey("cardsfolder/a/a.txt", 0, "spell", 0)
        log.note("long", (key,))
        log.note("long", (key,))
        log.note("other")
        assert reported == [("long", (key,)), ("other", ())]


class TestEncodingText:
    """FR-005: prose only for a line with no script, on the script surface."""

    def _line(self, script_text):
        return SidecarLine(
            line_index=0, line_kind="static", provenance=(),
            script_text=script_text,
        )

    def test_a_scripted_line_encodes_its_script(self):
        from effects.domain.ability_encoder import encoding_text

        chained = "Execute$ SV1 | Mode$ Attacks [SEG] SV1: DB$ Draw"
        assert encoding_text(self._line(chained), "draw a card.", "script") == chained

    def test_a_keyword_derived_line_encodes_its_prose(self):
        from effects.domain.ability_encoder import encoding_text

        assert encoding_text(self._line(None), "flying", "script") == "flying"

    def test_a_synthetic_land_mana_line_encodes_its_prose(self):
        from effects.domain.ability_encoder import encoding_text

        assert encoding_text(self._line(""), "{T}: add {G}.", "script") == "{T}: add {G}."


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
