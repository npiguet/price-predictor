"""The training loop itself, separated from the pure functions it drives.

``train_effect_model`` owns the split, the mixture and the schedule — all pure
functions of their inputs, all unit-testable with no torch. This module owns the
part that needs a GPU: assembling batches into tensors, stepping the optimizer,
and validating between epochs.

The loop's shape follows three constraints from the spec:

- **An epoch is a step count, not a pass over the corpus.** The corpus reaches
  10^8 records; a pass would be a week and would make ``--patience`` meaningless.
- **Validation runs on both strata every epoch, and the best checkpoint is
  chosen by the card-disjoint one** — the number that stands in for deployment
  to an unseen set, rather than in-distribution fit.
- **The batch groups each game's records together**, so one encode of an ability
  text serves every record in that game that mentions it.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from effects.application.train_effect_model import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    RANDOM_SEED,
    WEIGHT_DECAY,
    ContextCache,
    CorpusSplit,
    EarlyStopper,
    EpochResult,
    TrainEffectModelConfig,
    ability_text_of,
    learning_rate_at,
    plan_batch,
    sample_weights,
    sampling_class,
    variant_masks,
    warmup_steps,
    withheld_keyword_rules,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    collate_lines,
    prepare_line,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.effect_head_input import (
    SlotKind,
    act_features,
    build_effect_head_input,
    card_features,
    continuous_masked_keywords,
    global_features,
    player_features,
)
from effects.domain.effect_model import (
    EffectModel,
    EffectModelConfig,
    active_fields,
    collate_surfaces,
    entity_target_tensors,
    per_entity_loss,
)
from effects.domain.effect_targets import derive_targets
from effects.infrastructure.effect_model_store import (
    EffectCheckpoint,
    EffectModelStore,
    SplitProvenance,
    content_hash,
)
from effects.infrastructure.sidecar_io import SidecarCache
from price_predictor.infrastructure.tokenizer_store import load_vocabulary
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Owns one training run end to end."""

    def __init__(
        self,
        config: TrainEffectModelConfig,
        records: list,
        split: CorpusSplit,
        mix: dict[str, float],
    ) -> None:
        self.config = config
        self.records = records
        self.split = split
        self.mix = mix
        self.masks = variant_masks(config.variant)
        self.rng = random.Random(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.context_cache = (
            ContextCache(config.cache_refresh) if config.context_cache else None
        )

    # ── setup ───────────────────────────────────────────────────────────

    def _build_tokenizer(self) -> AbilityTokenizer:
        vocab = load_vocabulary(Path(self.config.vocab_path))
        definitions = {}
        keyword_path = Path(self.config.keyword_definitions)
        if keyword_path.exists():
            from effects.application.extract_keyword_definitions import (
                load_keyword_definitions,
            )

            definitions = load_keyword_definitions(keyword_path)
        return AbilityTokenizer(vocab, definitions)

    def _build_sidecars(self) -> SidecarCache:
        roots = {Path(f).name: Path(f) for f in self.config.cards_folders}
        if self.config.variant_scripts:
            roots["variant-scripts"] = Path(self.config.variant_scripts)
        return SidecarCache(roots)

    def _feature_widths(self) -> dict[SlotKind, int]:
        """Measure each slot kind's width from a real record.

        Measured rather than declared: the vocabularies the features are built
        from live in one module, and a constant here would be a second place to
        keep them in step.
        """
        probe = self.records[0]
        widths = {
            SlotKind.GLOBAL: len(global_features(probe)),
            SlotKind.ACT: len(act_features(probe)),
            SlotKind.PLAYER: 0,
            SlotKind.CARD: 0,
        }
        for record in self.records:
            if record.state.players:
                widths[SlotKind.PLAYER] = len(
                    player_features(record.state.players[0], record)
                )
            if record.state.entities:
                widths[SlotKind.CARD] = len(
                    card_features(record.state.entities[0], record)
                )
            if widths[SlotKind.PLAYER] and widths[SlotKind.CARD]:
                break
        return widths

    # ── the surface ─────────────────────────────────────────────────────

    def _encode_texts(
        self, texts: set[str], encoder: AbilityEncoder,
        tokenizer: AbilityTokenizer,
    ) -> dict[str, torch.Tensor]:
        """Encode each unique ability text once for the whole batch.

        This is what grouping a batch by game buys: the abilities on one board
        recur across that game's records, so one forward pass serves many.
        """
        if not texts:
            return {}
        hidden_keywords, _force = withheld_keyword_rules(
            self.config.withhold_keyword
        )
        ordered = sorted(texts)
        lines = []
        for text in ordered:
            tokens = tokenizer.tokenize(text)
            tokens = [t for t in tokens if t.text not in hidden_keywords]
            tokens = tokenizer.expand_keywords(
                tokens, probability=self.config.keyword_expand_p, rng=self.rng,
            )
            lines.append(prepare_line(tokenizer, tokens))
        batch = collate_lines(lines, tokenizer.pad_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        vectors, _hidden = encoder(**batch)
        return dict(zip(ordered, vectors))

    def _surface_for(self, record, vectors: dict, sidecars, e_dim: int):
        """One record's token surface, with the variant's masks applied."""

        def e_for(key):
            if self.masks.zero_e:
                return None
            try:
                line = sidecars.line_for(key)
            except KeyError:
                return None
            if line is None:
                return None
            text = line.script_text or f"{key.script_file}:{key.trait_kind}"
            vector = vectors.get(text)
            return None if vector is None else tuple(
                vector.detach().cpu().tolist()
            )

        return build_effect_head_input(
            record,
            e_for=e_for,
            e_dim=e_dim,
            context_dropout=self.config.context_dropout,
            rng=self.rng,
            masked_keywords=continuous_masked_keywords(record),
        )

    # ── stepping ────────────────────────────────────────────────────────

    def _loss_for(self, plan, encoder, model, tokenizer, sidecars, widths, step):
        records = plan.records
        if not records:
            return None
        texts = {
            text for record in records
            if (text := ability_text_of(record, sidecars)) is not None
        }
        vectors = self._encode_texts(texts, encoder, tokenizer)
        surfaces = [
            self._surface_for(record, vectors, sidecars, self.config.e_dim)
            for record in records
        ]
        batch = collate_surfaces(
            surfaces, e_dim=self.config.e_dim, widths=widths,
        )
        batch = {
            "slot_features": {
                k: v.to(self.device) for k, v in batch["slot_features"].items()
            },
            **{
                k: v.to(self.device)
                for k, v in batch.items() if k != "slot_features"
            },
        }
        hidden = model(**batch)
        outputs = model.per_entity(hidden)

        present = {sampling_class(record) for record in records}
        fields = active_fields(
            present_classes=frozenset(present), step=step,
            curriculum_step=self.config.curriculum_step,
        )
        targets = [derive_targets(record) for record in records]
        gate, field_targets, mask, index = entity_target_tensors(
            surfaces, targets, fields,
        )
        gathered = torch.stack([
            outputs[row, index[row]] for row in range(outputs.shape[0])
        ])
        loss, _parts = per_entity_loss(
            gathered, gate.to(self.device),
            {k: v.to(self.device) for k, v in field_targets.items()},
            mask.to(self.device), fields=fields,
        )
        return loss

    def _pools(self, records: list) -> tuple[dict, dict]:
        pools: dict[str, list] = defaultdict(list)
        for record in records:
            pools[sampling_class(record)].append(record)
        weights = {
            name: sample_weights(group, lambda r: r.record_id)
            for name, group in pools.items()
        }
        return dict(pools), weights

    # ── the run ─────────────────────────────────────────────────────────

    def execute(self) -> int:
        tokenizer = self._build_tokenizer()
        sidecars = self._build_sidecars()
        widths = self._feature_widths()

        encoder_config = AbilityEncoderConfig(
            vocab_size=tokenizer.vocab_size,
            e_dim=self.config.e_dim,
            e_noise=self.config.e_noise,
        )
        model_config = EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL],
            act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER],
            card_features=widths[SlotKind.CARD],
            e_dim=self.config.e_dim,
            vocab_size=tokenizer.vocab_size,
        )
        encoder = AbilityEncoder(encoder_config).to(self.device)
        model = EffectModel(model_config).to(self.device)

        optimizer = torch.optim.AdamW(
            [*encoder.parameters(), *model.parameters()],
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        warmup = warmup_steps(
            epochs=self.config.epochs, steps_per_epoch=self.config.steps_per_epoch,
        )

        training = [r for r in self.records if self.split.is_training_game(r.game_id)]
        pools, weights = self._pools(training)
        card_disjoint = [
            r for r in self.records if r.game_id in self.split.card_disjoint_games
        ]
        game_disjoint = [
            r for r in self.records if r.game_id in self.split.game_disjoint_games
        ]

        stopper = EarlyStopper(self.config.patience)
        store = EffectModelStore(self.config.resolved_model_output())
        provenance = self._provenance()
        step = 0

        for epoch in range(1, self.config.epochs + 1):
            encoder.train()
            model.train()
            running = 0.0
            for _ in range(self.config.steps_per_epoch):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate_at(step, warmup=warmup)
                plan = plan_batch(
                    pools, weights, self.mix,
                    batch_size=self.config.batch_size, rng=self.rng,
                )
                loss = self._loss_for(
                    plan, encoder, model, tokenizer, sidecars, widths, step,
                )
                if loss is None:
                    continue
                (loss / self.config.grad_accum).backward()
                if (step + 1) % self.config.grad_accum == 0:
                    clip_per_group(optimizer, MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach())
                step += 1
                if self.context_cache is not None:
                    self.context_cache.note_batch()

            result = EpochResult(
                epoch=epoch,
                train_loss=running / max(self.config.steps_per_epoch, 1),
                card_disjoint_loss=self._validate(
                    card_disjoint, encoder, model, tokenizer, sidecars, widths,
                    step,
                ),
                game_disjoint_loss=self._validate(
                    game_disjoint, encoder, model, tokenizer, sidecars, widths,
                    step,
                ),
            )
            logger.info(
                "epoch %d | train %.4f | card-disjoint %.4f | game-disjoint %.4f",
                result.epoch, result.train_loss, result.card_disjoint_loss,
                result.game_disjoint_loss,
            )
            if stopper.update(result.card_disjoint_loss):
                store.save(EffectCheckpoint(
                    encoder_config=encoder_config,
                    model_config=model_config,
                    encoder_state=encoder.state_dict(),
                    model_state=model.state_dict(),
                    provenance=provenance,
                    variant=self.config.variant,
                    best_val_loss=stopper.best,
                    epoch=epoch,
                ))
            if stopper.should_stop:
                logger.info(
                    "Early stop: %d epochs without a new card-disjoint best",
                    self.config.patience,
                )
                break
        return 0

    def _validate(
        self, records, encoder, model, tokenizer, sidecars, widths, step,
    ) -> float:
        if not records:
            return float("nan")
        encoder.eval()
        model.eval()
        by_game: dict[str, list] = defaultdict(list)
        for record in records[: self.config.batch_size * 8]:
            by_game[record.game_id].append(record)
        from effects.application.train_effect_model import BatchPlan

        total = 0.0
        batches = 0
        with torch.no_grad():
            for game_id, group in by_game.items():
                loss = self._loss_for(
                    BatchPlan({game_id: group}), encoder, model, tokenizer,
                    sidecars, widths, step,
                )
                if loss is not None:
                    total += float(loss)
                    batches += 1
        encoder.train()
        model.train()
        return total / batches if batches else float("nan")

    def _provenance(self) -> SplitProvenance:
        return SplitProvenance(
            held_out_cards=self.split.held_out_cards,
            card_disjoint_games=tuple(sorted(self.split.card_disjoint_games)),
            game_disjoint_games=tuple(sorted(self.split.game_disjoint_games)),
            vocab_path=str(self.config.vocab_path),
            keyword_definitions_path=str(self.config.keyword_definitions),
            vocab_hash=content_hash(Path(self.config.vocab_path)),
            keyword_definitions_hash=content_hash(
                Path(self.config.keyword_definitions)
            ),
            withheld_keyword=self.config.withhold_keyword,
        )


def config_summary(config: TrainEffectModelConfig) -> dict:
    """The run's settings, for the log line and the checkpoint's extras."""
    return {k: str(v) for k, v in asdict(config).items()}
