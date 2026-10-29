"""Shared builders for the effect-record unit tests.

Each test module used to hand-roll its own ``_record`` / ``_snapshot`` /
``_entity``. These three build the plain case; a test that needs something
specific passes it as an override, or keeps a small local wrapper that does.
"""

from __future__ import annotations

from pathlib import Path

from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    EntityState,
    GlobalState,
    InclusionTier,
    PlayerState,
    StateSnapshot,
)

EFFECTS_FIXTURES = Path(__file__).parents[2] / "fixtures" / "effects"
#: The converted card and token trees the gen-1 fixture records were collected against.
GEN1_SIDECAR_FOLDERS = (
    EFFECTS_FIXTURES / "gen1-sidecars" / "cardsfolder",
    EFFECTS_FIXTURES / "gen1-sidecars" / "tokenscripts",
)
GEN1_KEYWORD_DEFINITIONS = EFFECTS_FIXTURES / "keyword-definitions.json"
#: The seed and shard count of the short real runs over the gen-1 records.
GEN1_TRAINING_SEED = 11
GEN1_TRAINING_SHARDS = 3

_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


#: The tiers stage-one collection wrote: what a test snapshot claims by default.
STAGE_ONE_TIERS: frozenset[InclusionTier] = frozenset(
    {InclusionTier.REFERENCED, InclusionTier.CORE}
)


def make_snapshot(**overrides) -> StateSnapshot:
    """An empty turn-1 board collected at the stage-one tiers; ``overrides``
    replace any ``StateSnapshot`` field."""
    defaults: dict = {
        "global_": GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        "players": (),
        "entities": (),
        "tiers": STAGE_ONE_TIERS,
    }
    defaults.update(overrides)
    return StateSnapshot(**defaults)


def make_entity(entity_id: str, name: str, **overrides) -> EntityState:
    """A battlefield permanent controlled by ``P0``."""
    defaults: dict = {
        "id": entity_id, "name": name, "zone": "battlefield", "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _default_payload(kind: RecordKind, moment, subkind):
    if kind is RecordKind.RESOLUTION:
        if moment is Moment.ACTIVATION:
            return ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
            )
        return ResolutionPayload()
    if kind is RecordKind.PLAYABILITY:
        return {
            PlayabilitySubkind.DECISION: PlayabilityDecisionPayload,
            PlayabilitySubkind.ATTACKERS: PlayabilityAttackersPayload,
            PlayabilitySubkind.BLOCKERS: lambda: PlayabilityBlockersPayload(
                anchor_attacker="E1",
            ),
        }[subkind]()
    return {
        RecordKind.COMBAT: CombatPayload,
        RecordKind.CONTINUOUS: ContinuousPayload,
        RecordKind.TRIGGER: lambda: TriggerPayload(event=_EVENT, fired=True),
        RecordKind.REWRITE: lambda: RewritePayload(
            incoming=_EVENT, outgoing=_EVENT,
        ),
    }[kind]()


def make_record(kind: RecordKind = RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    """A valid record of ``kind`` with every envelope field defaulted.

    ``moment`` (resolution) and ``subkind`` (playability) select the default
    payload; any field, ``payload`` included, can be overridden.
    """
    game_id = overrides.get("game_id", "g1")
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": kind, "actor_player": "P0",
        "state": make_snapshot(),
    }
    moment = overrides.pop("moment", None)
    subkind = overrides.pop("subkind", None)
    if kind is RecordKind.RESOLUTION:
        moment = moment or Moment.RESOLUTION
    elif kind is RecordKind.PLAYABILITY:
        subkind = subkind or PlayabilitySubkind.DECISION
    # A discriminator on the wrong kind is forwarded as given, so a test can
    # still build the record the envelope check is meant to refuse.
    if moment is not None:
        defaults["moment"] = moment
    if subkind is not None:
        defaults["subkind"] = subkind
    if "payload" not in overrides:
        defaults["payload"] = _default_payload(kind, moment, subkind)
    defaults.update(overrides)
    return EffectRecord(**defaults)


def named_entity(name: str, **overrides) -> EntityState:
    """:func:`make_entity` with its id derived from ``name``."""
    return make_entity(f"E-{name}", name, **overrides)


def record_on_board(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    """A record of game ``game_id`` whose board holds ``entities``."""
    return make_record(
        game_id=game_id, ability=ability,
        state=make_snapshot(entities=tuple(entities)), **overrides,
    )


#: Records per gzip member, as the Java collector blocks them.
RAW_BLOCK_RECORDS = 256


class RawShardWriter:
    """One worker lifetime's raw shard, laid out as the Java collector writes it.

    ``{run_id}.{worker}-{lifetime}.jsonl.gz``, a concatenation of gzip members
    of :data:`RAW_BLOCK_RECORDS` records each, appended a block at a time.
    ``compressed=False`` writes the plain ``.jsonl`` shards collected before
    compression, which every reader still loads. Python never writes a raw
    shard outside the tests, so the writer lives here rather than beside the
    readers.
    """

    def __init__(
        self, directory, run_id: str, worker: int, lifetime: str, *,
        compressed: bool = True,
    ) -> None:
        suffix = ".jsonl.gz" if compressed else ".jsonl"
        self.path = Path(directory) / f"{run_id}.{worker}-{lifetime}{suffix}"
        self.compressed = compressed
        self._block: list[str] = []

    def __enter__(self) -> RawShardWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, record: EffectRecord) -> None:
        from effects.infrastructure.record_io import format_record_line

        self._block.append(format_record_line(record) + "\n")
        if len(self._block) >= RAW_BLOCK_RECORDS:
            self._flush()

    def _flush(self) -> None:
        import gzip

        if not self._block:
            return
        data = "".join(self._block).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(gzip.compress(data) if self.compressed else data)
        self._block = []

    def close(self) -> None:
        self._flush()


def fresh_interpreter_env(**overrides: str) -> dict[str, str]:
    """The environment for a ``sys.executable`` child that imports the tree
    under test, which an installed copy of the package would otherwise shadow."""
    import os
    from pathlib import Path

    import effects

    source = Path(effects.__file__).parents[1]
    return {**os.environ, "PYTHONPATH": str(source), **overrides}


def slots_of_kind(surface, kind) -> tuple:
    """A token surface's slots of one ``SlotKind``, in surface order."""
    return tuple(slot for slot in surface.slots if slot.kind is kind)


def make_preparer(
    directory, *, seed: int = 0, api_types=(), param_keys=(), rarity=None, **config,
):
    """A ``StepPreparer`` over a vocabulary of the specials alone, for tests
    about how it builds batchers rather than about what they read."""
    from effects.application.step_preparation import PreparerSettings, StepPreparer
    from effects.application.train_effect_model import TrainEffectModelConfig
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS
    from effects.domain.split import CorpusSplit
    from price_predictor.infrastructure.tokenizer_store import save_vocabulary

    vocab = directory / "vocab-script.txt"
    save_vocabulary({token: index for index, token in enumerate(SPECIAL_TOKENS)}, vocab)
    settings = dict(
        corpus="unused", vocab_path=vocab,
        keyword_definitions=directory / "no-keyword-definitions.json",
        cards_folders=(directory / "cardsfolder",),
    )
    settings.update(config)
    return StepPreparer(PreparerSettings(
        config=TrainEffectModelConfig(**settings), seed=seed, split=CorpusSplit(),
        rarity={} if rarity is None else rarity, api_types=tuple(api_types),
        param_keys=tuple(param_keys), widths={}, heads=frozenset(),
    ))


def make_batcher(
    *, tokenizer=None, sidecars=None, masks=None, surface: str = "script",
    e_dim: int = 4, widths=None, device=None, withhold_keyword=None,
    keyword_expand_p: float = 0.0, context_dropout: float = 0.0, rng=None,
    identity_table=None, noise=None, truncations=None, input_layout=None,
    resolver=None,
):
    """A ``SurfaceBatcher`` and its collaborators from one flat set of settings."""
    import torch

    from effects.application.ability_text import KeyResolver
    from effects.application.surface_batching import (
        Materializer,
        SurfaceBatcher,
        SurfaceSpec,
    )
    from effects.application.text_staging import TextStager
    from effects.domain.baselines import BaselineMasks
    from effects.domain.effect_head_input import INPUT_LAYOUT

    masks = masks or BaselineMasks()
    return SurfaceBatcher(
        stager=TextStager(
            tokenizer=tokenizer, surface=surface, withhold_keyword=withhold_keyword,
            keyword_expand_p=keyword_expand_p, truncations=truncations,
        ),
        resolver=resolver or KeyResolver(sidecars, surface),
        spec=SurfaceSpec(e_dim, widths or {}, masks, input_layout or INPUT_LAYOUT),
        materializer=Materializer(
            device or torch.device("cpu"), masks, identity_table, noise,
        ),
        context_dropout=context_dropout, rng=rng,
    )


def make_training_loop(
    config, *, training_shards=(), validation_samples=None, **facts,
):
    """A ``TrainingLoop`` over ``config`` and a corpus described by ``facts``.

    ``facts`` override ``CorpusFacts`` fields; by default the fixture holdout
    rule (20 permille, at most 8 carriers) and an empty rarity table.
    """
    from effects.application.train_effect_model import CorpusFacts
    from effects.application.training_loop import TrainingLoop
    from effects.domain.split import HoldoutRule
    from effects.domain.text_holdout import HoldoutUnit

    return TrainingLoop(
        config,
        facts=CorpusFacts(**{
            "holdout": HoldoutRule(unit=HoldoutUnit.TEXT), "rarity": {}, **facts,
        }),
        training_shards=list(training_shards),
        validation_samples=validation_samples or {},
    )


def make_cpu_training_loop(monkeypatch, **config):
    """A ``TrainingLoop`` over no corpus, on a real ``cpu`` device.

    Forced whatever ``torch.cuda.is_available()`` says: on this box it can
    read ``True`` even under ``CUDA_VISIBLE_DEVICES=""`` (its raw device count
    ignores the mask — the same disagreement ``autocast_enabled`` guards
    against), which would move a test's tensors onto a real GPU by accident.
    """
    import torch

    from effects.application.train_effect_model import TrainEffectModelConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return make_training_loop(TrainEffectModelConfig(corpus="unused", **config))


def one_parameter_module():
    """A module with one parameter, so backward and the optimizer have work."""
    import torch

    module = torch.nn.Module()
    module.p = torch.nn.Parameter(torch.ones(2))
    return module


class StubLosses:
    """Stands in for a loop's ``StepLossComputer``: a step's loss is
    ``loss(prepared)``, a ``StepLoss`` or a scalar tensor, and no MLM pass runs."""

    def __init__(self, loss) -> None:
        self.loss = loss

    def device_loss(self, prepared, modules, materializer, **kwargs):
        from effects.application.step_loss import StepLoss
        from effects.domain.effect_losses import LossParts

        computed = self.loss(prepared)
        if isinstance(computed, StepLoss):
            return computed
        return StepLoss(total=computed, shipped=computed.detach(), parts=LossParts(), terms={})

    def mlm_backward(self, loss, encoder, model) -> None:
        return None


def stub_validator(loop, on_score=None) -> None:
    """Swap ``loop``'s validator for one that scores every pass 1.0 without
    running the model, calling ``on_score(fields, margins)`` for each pass.

    Only ``score`` is stood in for, so the epoch's own choice of which pass
    feeds the margins and the floor is still the real one.
    """
    from effects.application.validation_pass import ProgressMargins, Validator

    class _Scored(Validator):
        def score(self, records, run, fields, collectors=()):
            if on_score is not None:
                on_score(
                    fields, any(isinstance(c, ProgressMargins) for c in collectors),
                )
            return 1.0

    loop.validator = _Scored(loop.config, loop.device, loop.losses)


def stub_below_the_loop(monkeypatch):
    """Replace what a ``TrainingLoop`` builds beneath itself — the encoder, the
    head, the tokenizer, the script-API vocabulary and the feature widths — so
    ``execute`` runs over records no sidecar describes.

    Returns the ``(model, encoder)`` stand-ins, from :func:`one_parameter_module`.
    """
    from effects.application import run_context, step_preparation
    from effects.application import training_loop as loop_module
    from effects.domain.effect_head_input import SlotKind

    model, encoder = one_parameter_module(), one_parameter_module()
    monkeypatch.setattr(run_context, "AbilityEncoder", lambda config: encoder)
    monkeypatch.setattr(run_context, "EffectModel", lambda config: model)
    monkeypatch.setattr(
        step_preparation, "load_tokenizer",
        lambda *a, **k: type("T", (), {"vocab_size": 8})(),
    )
    # The script-API vocabulary is read off every real sidecar; none here.
    monkeypatch.setattr(run_context, "script_vocabularies", lambda folders: ([], []))
    monkeypatch.setattr(
        loop_module, "feature_widths",
        lambda records: dict.fromkeys(SlotKind, 4),
    )
    return model, encoder


def write_validation_samples(directory, records) -> dict:
    """Both validation strata's samples under ``directory``, each ``records``."""
    from effects.infrastructure.record_io import write_shard

    samples = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        samples[stratum] = directory / f"{stratum}.jsonl.gz"
        write_shard(samples[stratum], records)
    return samples


def fields_everywhere(values):
    """``EntityFieldTargets`` holding ``values``, each present at every entity."""
    import torch

    from effects.domain.entity_field_targets import EntityFieldTargets

    return EntityFieldTargets(
        dict(values),
        {name: torch.ones(value.shape[:2], dtype=torch.bool) for name, value in values.items()},
    )


def stated_targets(text: str) -> dict[str, float | None]:
    """The value head's targets for ``text`` by name, None where masked."""
    from effects.domain.value_targets import VALUE_TARGET_NAMES, value_targets

    targets = value_targets(text)
    return {
        name: value if stated else None
        for name, value, stated in zip(VALUE_TARGET_NAMES, targets.values, targets.mask)
    }


def make_entity_targets(gate, fields=None, mask=None, index=None):
    """An ``EntityTargetBatch`` for a loss test. A mapping of ``fields`` is
    present everywhere; ``mask`` defaults to every entity real and ``index``
    to column zero, which no loss reads."""
    import torch

    from effects.domain.entity_field_targets import EntityFieldTargets, EntityTargetBatch

    if not isinstance(fields, EntityFieldTargets):
        fields = fields_everywhere(fields or {})
    if mask is None:
        mask = torch.ones_like(gate)
    if index is None:
        index = torch.zeros(gate.shape, dtype=torch.long)
    return EntityTargetBatch(gate, fields, mask, index)


def read_cache(abilities_root, sidecar_roots, *, surface, variant: str = "full"):
    """One variant's cache over freshly read sidecars, the way the evaluator
    reads it: :func:`cached_vectors` over a ``SidecarIndex``."""
    from effects.application.geometry_checks import cached_vectors
    from effects.infrastructure.ability_cache_store import SidecarIndex

    return cached_vectors(
        SidecarIndex(sidecar_roots, surface=surface), abilities_root, variant=variant,
    )


def keyword_score(qualifying: int, agreeing: int):
    """Gate 2's score with one observation per qualifying record: a combat
    with a single carrier."""
    from effects.application.gate_two import KeywordScore

    return KeywordScore(
        qualifying=qualifying, agreeing=agreeing, observations=qualifying,
    )


def gen1_training_config(corpus, out, **overrides):
    """A short real run's config over the application conftest's ``corpus``
    fixture, every draw the host half takes switched on."""
    from effects.application.train_effect_model import TrainEffectModelConfig

    settings = dict(
        corpus=str(corpus.root), cards_folders=GEN1_SIDECAR_FOLDERS,
        keyword_definitions=GEN1_KEYWORD_DEFINITIONS, vocab_path=corpus.vocab,
        model_output=out, seed=GEN1_TRAINING_SEED, batch_size=8, steps_per_epoch=7,
        shards_per_epoch=GEN1_TRAINING_SHARDS, epochs=2, curriculum_epoch=2, patience=5,
        e_dim=16, encoder_layers=1, encoder_d_model=32,
        # Every draw the host half takes, at rates that make each one fire.
        keyword_expand_p=0.5, context_dropout=0.2, mlm_weight=0.1,
        mlm_mask_prob=0.3, value_weight=0.05, api_weight=0.05, e_noise=0.1,
    )
    settings.update(overrides)
    return TrainEffectModelConfig(**settings)


def write_sidecar(sidecar, path: Path) -> None:
    """Write one sidecar where ``convert`` would, so a test can produce one
    without Forge; the inverse of ``sidecar_io.read_sidecar``."""
    import json

    from effects.infrastructure.sidecar_io import sidecar_to_dict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar_to_dict(sidecar), separators=(",", ":")), encoding="utf-8",
    )


def player_of(snapshot: StateSnapshot, player_id: str) -> PlayerState | None:
    """The snapshot's player of ``player_id``, or None."""
    return next((each for each in snapshot.players if each.id == player_id), None)
