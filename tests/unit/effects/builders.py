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
    TurnActionPayload,
    TurnActionSubkind,
)
from effects.domain.split import HeldOutCards
from effects.domain.state_snapshot import (
    EntityState,
    GlobalState,
    InclusionTier,
    PlayerState,
    StateSnapshot,
)
from effects.domain.text_holdout import HoldoutUnit, holdout_key

EFFECTS_FIXTURES = Path(__file__).parents[2] / "fixtures" / "effects"
#: Forge's keyword table, as the fixture records' runs read it.
FIXTURE_KEYWORD_DEFINITIONS = EFFECTS_FIXTURES / "keyword-definitions.json"
#: The seed and shard count of the short real runs over the gen-3 records.
FIXTURE_TRAINING_SEED = 11
FIXTURE_TRAINING_SHARDS = 3

#: Real gen-3 JSON: the goldens the connector's characterization tests emit
#: (``-Dcharacterization.update=true``), read until the gen-3 fixture exists.
JAVA_GOLDENS = (
    Path(__file__).parents[3] / "forge-connector" / "src" / "test" / "resources"
    / "characterization"
)
#: The one golden that is no collected record: an AI what-if blockers record
#: pinned for its envelope alone, with an empty payload no collector writes.
_ENVELOPE_ONLY_GOLDENS = frozenset({"record-what-if.jsonl"})
def java_golden_records() -> list[dict]:
    """Every whole record the Java goldens hold, as the raw JSON objects."""
    import json

    return [
        json.loads(line)
        for path in sorted(JAVA_GOLDENS.glob("*.jsonl"))
        if path.name not in _ENVELOPE_ONLY_GOLDENS
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def java_golden_entities() -> list[dict]:
    """Every entity of the Java goldens, the standalone snapshots' included."""
    import json

    snapshots = [record["state"] for record in java_golden_records()]
    snapshots += [json.loads(path.read_text(encoding="utf-8"))
                  for path in sorted(JAVA_GOLDENS.glob("snapshot-*.json"))]
    return [entity for snapshot in snapshots for entity in snapshot["entities"]]


#: Real gen-3 records cut from the T079 pilot (``scripts/make_fixture_records.py
#: --generation 3``), with the sidecars every key they name resolves to.
GEN3_RECORDS = EFFECTS_FIXTURES / "gen3-records.jsonl.gz"
GEN3_SIDECAR_FOLDERS = (
    EFFECTS_FIXTURES / "gen3-sidecars" / "cardsfolder",
    EFFECTS_FIXTURES / "gen3-sidecars" / "tokenscripts",
)


def gen3_fixture_records() -> list[dict]:
    """The gen-3 fixture's records, as the raw JSON objects the collector wrote."""
    import gzip
    import json

    with gzip.open(GEN3_RECORDS, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def curated_fixture_records() -> list:
    """The gen-3 fixture's records as a curated corpus holds them, parsed:
    every record but the ``turn_action`` ones, which ``build-corpus`` writes
    into no stratum, so no trainer reads one."""
    from effects.domain.records import RecordKind
    from effects.infrastructure.shard_io import read_shard

    return [r for r in read_shard(GEN3_RECORDS) if r.kind is not RecordKind.TURN_ACTION]


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
    if kind is RecordKind.TURN_ACTION:
        return TurnActionPayload()
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
    elif kind is RecordKind.TURN_ACTION:
        subkind = subkind or TurnActionSubkind.DRAW
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


def held_out_cards(
    names=(), script_files=(), *, texts=(), unit: HoldoutUnit = HoldoutUnit.TEXT,
) -> HeldOutCards:
    """A holdout of exactly these cards, script files and ``texts``, each
    text's :meth:`HeldOutCards.holds_text` key derived under ``unit`` as
    :func:`text_keyed_holdout` derives it."""
    return HeldOutCards(
        names=frozenset(names), script_files=frozenset(script_files),
        texts=frozenset(texts),
        keys=frozenset(holdout_key(text, unit) for text in texts), unit=unit,
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
        from effects.infrastructure.record_codec import format_record_line

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


def write_plain_shard(
    directory, lifetime: str, records, *, run_id: str = "run", worker: int = 0,
) -> Path:
    """A plain ``{run_id}.{worker}-{lifetime}.jsonl`` raw shard holding
    ``records``, written through :class:`RawShardWriter`."""
    with RawShardWriter(directory, run_id, worker, lifetime, compressed=False) as shard:
        for record in records:
            shard.write(record)
    return shard.path


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
    identity_table=None, noise=None, truncations=None,
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

    masks = masks or BaselineMasks()
    return SurfaceBatcher(
        stager=TextStager(
            tokenizer=tokenizer, surface=surface, withhold_keyword=withhold_keyword,
            keyword_expand_p=keyword_expand_p, truncations=truncations,
        ),
        resolver=resolver or KeyResolver(sidecars, surface),
        spec=SurfaceSpec(e_dim, widths or {}, masks),
        materializer=Materializer(
            device or torch.device("cpu"), masks, identity_table, noise,
        ),
        context_dropout=context_dropout, rng=rng,
    )


def make_training_loop(
    config, *, training_shards=(), validation_samples=None, selection=None, **facts,
):
    """A ``TrainingLoop`` over ``config`` and a corpus described by ``facts``.

    ``facts`` override ``CorpusFacts`` fields; by default the fixture holdout
    rule (20 permille, at most 8 carriers) and an empty rarity table.
    ``selection`` is the control score's ``FrozenSelection``
    (:func:`fixture_selection`); a loop without one runs only under
    :func:`stub_validator`.
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
        selection=selection,
    )


#: The strata a fixture selection set halves.
FIXTURE_STRATA: tuple[str, ...] = ("card-disjoint", "game-disjoint")


def fixture_selection_set(records, *, corpus_digest: str = "fixture"):
    """A selection set over ``records`` with every component holding items in
    both strata: both strata hold every game of ``records``, split in two by
    game, each half's items over its own games' records: two probe types, the
    real damage grid (its 64 cells, edited by ``grid_edits.GridEdits``) over
    records with an acting line and a creature, a rule test, and a deviance
    sample whose gate and three fields the freeze kept."""
    from effects.domain.effect_head_layout import GATE_TERM
    from effects.domain.grid_edits import VALUES
    from effects.domain.selection_set import (
        DEVIANCE,
        FrozenGrid,
        GridCell,
        ProbeRecord,
        SampleRecord,
        SelectionSet,
        TaggedRecord,
        item_key,
    )

    def creature(record):
        return next((e for e in record.state.entities
                     if e.pt is not None and e.is_creature and e.on_battlefield), None)

    games = sorted({r.game_id for r in records})
    halves = {"selection": games[::2], "reporting": games[1::2]}
    by_half = {half: [r for r in records if r.game_id in set(held)]
               for half, held in halves.items()}
    cells = tuple(GridCell(f"{a}x{b}", float(a >= b)) for a in VALUES for b in VALUES)
    probes, grids, rules, samples, errors = {}, {}, {}, {}, {}
    for stratum in FIXTURE_STRATA:
        probes[stratum], grids[stratum], rules[stratum], samples[stratum] = {}, {}, {}, {}
        for half, held in by_half.items():
            with_entities = [r for r in held if r.state.entities]
            probes[stratum][half] = {
                "dies": tuple(ProbeRecord(r.record_id, r.game_id, r.state.entities[0].id,
                                          float(i % 2)) for i, r in enumerate(with_entities[:12])),
                "affordable": tuple(ProbeRecord(r.record_id, r.game_id, None, float(i % 2))
                                    for i, r in enumerate(held[:10])),
            }
            damaged = [(r, creature(r)) for r in held if r.ability and creature(r) is not None]
            grids[stratum][half] = {"damage": FrozenGrid(cells, tuple(
                TaggedRecord(r.record_id, r.game_id, (entity.id,)) for r, entity in damaged[:2]))}
            tagged = [TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
                      for r in with_entities[:16]]
            rules[stratum][half] = {
                "r.board": {"fires": tuple(tagged[:8]), "does_not_fire": tuple(tagged[8:])},
            }
            samples[stratum][half] = tuple(SampleRecord(r.record_id, r.game_id, 0)
                                           for r in held[:20])
            for name in (GATE_TERM, "zone_outcome", "damage_taken", "pt_change"):
                errors[item_key(stratum, half, DEVIANCE, name)] = {"hard": 0.05, "soft": 0.05}
    return SelectionSet(
        corpus_digest=corpus_digest, r=4,
        halves={s: {h: tuple(g) for h, g in halves.items()} for s in FIXTURE_STRATA},
        probes=probes, probe_kinds={"dies": "binary", "affordable": "binary"},
        grids=grids, rule_tests=rules, deviance_samples=samples,
        standard_errors=errors, left_out={}, settings={"max_item_error": 0.1},
    ).sealed()


#: The rule map's outputs of :func:`fixture_selection_set`'s rule test.
FIXTURE_RULE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "r.board": ("gate", "zone_outcome", "damage_taken", "can_play", "trigger_fired",
                "created_objects"),
}


def fixture_selection(records):
    """A ``FrozenSelection`` of :func:`fixture_selection_set` whose strata
    both read ``records`` as one shard."""
    from effects.application.control_score import FrozenSelection

    records = list(records)
    return FrozenSelection(fixture_selection_set(records),
                           dict.fromkeys(FIXTURE_STRATA, lambda: [records]))


def control_score_of(value: float, weight: float = 0.5):
    """A ``ControlScore`` whose every component of both strata is ``value``."""
    from effects.application.control_score import ControlScore, StratumScore
    from effects.domain.selection_set import COMPONENTS

    return ControlScore({stratum: StratumScore(dict.fromkeys(COMPONENTS, value), {})
                         for stratum in FIXTURE_STRATA}, weight)


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
        self.denominators = None

    def use_denominators(self, denominators) -> None:
        self.denominators = denominators

    def device_loss(self, prepared, modules, materializer, **kwargs):
        from effects.application.step_loss import StepLoss
        from effects.domain.effect_losses import LossParts

        computed = self.loss(prepared)
        if isinstance(computed, StepLoss):
            return computed
        return StepLoss(total=computed, shipped=computed.detach(), parts=LossParts(), terms={})

    def mlm_backward(self, loss, encoder, model) -> None:
        return None


def stub_validator(loop, on_score=None, controls=None) -> None:
    """Swap ``loop``'s validator for one that scores every pass 1.0 without
    running the model, calling ``on_score(fields, margins)`` for each pass,
    and whose control score is ``controls[n]`` at its ``n``-th epoch (the
    last one repeated; 0.0 every epoch by default), with no cache prepared.

    Only ``score`` and the control score are stood in for, so the epoch's own
    choice of which pass feeds the margins and the floor is still the real one.
    """
    from effects.application.validation_pass import ProgressMargins, Validator

    scores = list(controls or [0.0])

    class _Scored(Validator):
        epochs = 0

        def prepare(self, run, selection, *, rule_outputs):
            return None

        def control(self, run, fields):
            value = scores[min(self.epochs, len(scores) - 1)]
            self.epochs += 1
            return control_score_of(value)

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
    import torch

    from effects.application import run_context, step_preparation
    from effects.application import training_loop as loop_module
    from effects.domain.effect_head_input import SlotKind
    from effects.domain.effect_head_layout import DISPERSED_FIELDS

    model, encoder = one_parameter_module(), one_parameter_module()
    # What validation reads off the model for gate 1's deviance margin.
    model.per_entity_dispersion = lambda: torch.zeros(len(DISPERSED_FIELDS))
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
    # No record here is a real one to prepare, so no pre-pass reads them.
    from effects.domain.loss_denominators import LossDenominators

    monkeypatch.setattr(
        loop_module, "loss_denominators",
        lambda *a, floor, **k: LossDenominators.of({}, floor=floor),
    )
    return model, encoder


def write_validation_samples(directory, records) -> dict:
    """Both validation strata's samples under ``directory``, each ``records``."""
    from effects.infrastructure.shard_io import write_shard

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


def fixture_training_config(corpus, out, **overrides):
    """A short real run's config over the application conftest's ``corpus``
    fixture, every draw the host half takes switched on."""
    from effects.application.train_effect_model import TrainEffectModelConfig

    settings = dict(
        corpus=str(corpus.root), cards_folders=GEN3_SIDECAR_FOLDERS,
        keyword_definitions=FIXTURE_KEYWORD_DEFINITIONS, vocab_path=corpus.vocab,
        model_output=out, seed=FIXTURE_TRAINING_SEED, batch_size=8, steps_per_epoch=7,
        shards_per_epoch=FIXTURE_TRAINING_SHARDS, epochs=2, curriculum_epoch=2, patience=5,
        e_dim=16, encoder_layers=1, encoder_d_model=32,
        # Every draw the host half takes, at rates that make each one fire.
        keyword_expand_p=0.5, context_dropout=0.2, mlm_weight=0.1,
        mlm_mask_prob=0.3, value_weight=0.05, api_weight=0.05, e_noise=0.1,
    )
    settings.update(overrides)
    return TrainEffectModelConfig(**settings)


def fixture_run(corpus, records, out, **overrides):
    """A real run's context over the application conftest's ``corpus``, on
    the CPU at a tiny trunk: the encoder, model, scoring materializer and
    preparer a training loop builds, ``records`` sizing the feature widths."""
    from dataclasses import replace
    from unittest.mock import patch

    import torch

    from effects.application import run_context
    from effects.application.step_preparation import StepPreparer
    from effects.domain.effect_head_input import feature_widths
    from effects.domain.effect_model import EffectModel
    from effects.domain.sampling import sampling_class

    loop = make_training_loop(fixture_training_config(
        corpus, out, prefetch_workers=0, **overrides))
    preparer = StepPreparer(run_context.preparer_settings(
        loop.spec, widths=feature_widths(list(records))))
    tiny = lambda config: EffectModel(replace(  # noqa: E731
        config, d_model=32, n_layers=1, n_heads=2, ff_dim=64))
    with patch.object(run_context, "EffectModel", tiny):
        torch.manual_seed(0)
        return run_context.build_run(
            loop.spec, preparer, frozenset(sampling_class(r) for r in records),
            device=torch.device("cpu"))


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
