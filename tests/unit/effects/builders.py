"""Shared builders for the effect-record unit tests.

Each test module used to hand-roll its own ``_record`` / ``_snapshot`` /
``_entity``. These three build the plain case; a test that needs something
specific passes it as an override, or keeps a small local wrapper that does.
"""

from __future__ import annotations

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
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot

_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


def make_snapshot(**overrides) -> StateSnapshot:
    """An empty turn-1 board; ``overrides`` replace any ``StateSnapshot`` field."""
    defaults: dict = {
        "global_": GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        "players": (),
        "entities": (),
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
        from pathlib import Path

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
    from effects.application.surface_batching import Materializer, SurfaceBatcher
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
        masks=masks, e_dim=e_dim, widths=widths or {},
        materializer=Materializer(
            device or torch.device("cpu"), masks, identity_table, noise,
        ),
        context_dropout=context_dropout, rng=rng,
        input_layout=input_layout or INPUT_LAYOUT,
    )


def make_training_loop(
    config, *, training_shards=(), validation_samples=None, prefetch_batches=True,
    **facts,
):
    """A ``TrainingLoop`` over ``config`` and a corpus described by ``facts``.

    ``facts`` override ``CorpusFacts`` fields; by default the fixture holdout
    rule (20 permille, at most 8 carriers) and an empty rarity table.
    """
    from effects.application.train_effect_model import CorpusFacts
    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config,
        facts=CorpusFacts(**{
            "holdout_permille": 20, "holdout_max_carriers": 8, "rarity": {},
            **facts,
        }),
        training_shards=list(training_shards),
        validation_samples=validation_samples or {},
        prefetch_batches=prefetch_batches,
    )
