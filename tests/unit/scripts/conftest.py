"""Put ``scripts/`` on the import path, as a script running from it would have."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402
from effect_knowledge_probes import common, labels  # noqa: E402

from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS, curated_fixture_records  # noqa: E402
from tests.unit.scripts.knowledge_probe_support import E_DIM, join_triggers  # noqa: E402


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


@pytest.fixture(scope="module")
def sidecars():
    return common.sidecar_cache(GEN3_SIDECAR_FOLDERS)


@pytest.fixture(scope="module")
def acting(sidecars):
    return common.acting_text_function(sidecars, "script")


@pytest.fixture(scope="module")
def items_by_record(records, sidecars, acting) -> dict[str, list]:
    """Every fixture record's board items by record id, triggers joined as a freeze joins them."""
    join = join_triggers(records)
    return {r.record_id: labels.record_items(r, sidecars, join, acting) for r in records}


@pytest.fixture(scope="module")
def keywords_of(sidecars):
    from effects.domain.damage_step_keywords import KeywordResolver

    return KeywordResolver(sidecars).keywords_of


@pytest.fixture(scope="module")
def probe_model(records, sidecars):
    """A gen-3 encoder, trunk and batcher at random weights over the fixture's
    texts: what a probe reads off a gen-3 checkpoint, without one."""
    import torch

    from effects.application.model_loading import evaluation_fields
    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer, Surface
    from effects.domain.baselines import baseline_masks
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.domain.input_layout import SlotKind
    from effects.domain.slot_widths import layout_widths
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions
    from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, make_batcher

    definitions = load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS)
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    widths = layout_widths()
    probe = make_batcher(tokenizer=bare, sidecars=sidecars, masks=baseline_masks("full"),
                         surface="script", e_dim=E_DIM, widths=widths,
                         device=torch.device("cpu"))
    vocab = dict(base)
    texts = [*probe.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    tokenizer = AbilityTokenizer(vocab, definitions, surface="script")
    batcher = make_batcher(tokenizer=tokenizer, sidecars=sidecars,
                           masks=baseline_masks("full"), surface="script", e_dim=E_DIM,
                           widths=widths, device=torch.device("cpu"))
    torch.manual_seed(0)
    encoder = AbilityEncoder(AbilityEncoderConfig(
        vocab_size=len(vocab), e_dim=E_DIM, d_model=16, n_layers=1)).eval()
    model = EffectModel(EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
        ability_features=widths[SlotKind.ABILITY], e_dim=E_DIM, vocab_size=len(vocab),
        encoder_d_model=16, d_model=16, n_layers=1, n_heads=2, ff_dim=32,
    )).eval()
    return common.ProbeModel(
        checkpoint=None, encoder=encoder, model=model, batcher=batcher,
        fields=evaluation_fields(records), device=torch.device("cpu"),
        surface=Surface.SCRIPT, e_dim=E_DIM)
