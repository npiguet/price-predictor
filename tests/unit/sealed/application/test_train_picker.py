"""Unit tests for the train-picker application + CLI validation."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from price_predictor.infrastructure.torch_training import kl_divergence, policy_entropy
from sealed.application.train_picker import (
    RANDOM_SEED,
    PreparedPool,
    TrainPickerConfig,
    _compute_losses,
    _EntropySchedule,
    _plackett_luce_log_prob,
    _sample_decks,
    _should_stop,
    _shuffle_train,
    _split_pools,
)
from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE

# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #

def _pool(n_spells: int, n_lands: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = n_spells + n_lands
    is_land = torch.zeros(1, n, dtype=torch.bool)
    is_land[0, n_spells:] = True
    pool_mask = torch.ones(1, n, dtype=torch.bool)
    return is_land, pool_mask


class TestSampler:
    def test_stops_at_spell_quota(self):
        torch.manual_seed(0)
        is_land, pool_mask = _pool(n_spells=30, n_lands=4)
        logits = torch.randn(1, 34)
        picks, mask = _sample_decks(logits, is_land, pool_mask, n_samples=8, temperature=1.0)
        assert picks.shape[0] == 8
        assert picks.shape == mask.shape
        for row in range(8):
            real_idx = picks[row][mask[row]]
            land_flags = is_land[0][real_idx]
            n_spells = int((~land_flags).sum())
            assert n_spells == NONLAND_DECK_SIZE

    def test_no_card_picked_twice(self):
        torch.manual_seed(1)
        is_land, pool_mask = _pool(n_spells=30, n_lands=5)
        logits = torch.randn(1, 35)
        picks, mask = _sample_decks(logits, is_land, pool_mask, n_samples=16, temperature=1.0)
        for row in range(16):
            real_idx = picks[row][mask[row]].tolist()
            assert len(real_idx) == len(set(real_idx))

    def test_lands_bucketed_not_counted(self):
        # Lands have the highest logits so they're picked early but don't count
        # toward the spell quota; the walk still ends with exactly 23 spells.
        is_land, pool_mask = _pool(n_spells=25, n_lands=3)
        logits = torch.zeros(1, 28)
        logits[0, 25:] = 10.0  # lands ranked highest
        picks, mask = _sample_decks(logits, is_land, pool_mask, n_samples=4, temperature=0.5)
        for row in range(4):
            real_idx = picks[row][mask[row]]
            land_flags = is_land[0][real_idx]
            assert int((~land_flags).sum()) == NONLAND_DECK_SIZE
            assert int(land_flags.sum()) == 3  # all 3 lands taken early

    def test_batched_across_b_and_s(self):
        torch.manual_seed(2)
        is_land = torch.zeros(3, 30, dtype=torch.bool)
        is_land[:, 25:] = True
        pool_mask = torch.ones(3, 30, dtype=torch.bool)
        logits = torch.randn(3, 30)
        picks, mask = _sample_decks(logits, is_land, pool_mask, n_samples=5, temperature=1.0)
        assert picks.shape[0] == 3 * 5


# --------------------------------------------------------------------------- #
# Plackett-Luce log-prob
# --------------------------------------------------------------------------- #

class TestPlackettLuce:
    def test_matches_hand_computation(self):
        logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
        picks = torch.tensor([[0, 2]])
        picked_mask = torch.tensor([[True, True]])
        lp = _plackett_luce_log_prob(logits, picks, picked_mask)

        step0 = 2.0 - torch.logsumexp(torch.tensor([2.0, 1.0, 0.5, -1.0]), 0)
        step1 = 0.5 - torch.logsumexp(torch.tensor([1.0, 0.5, -1.0]), 0)
        expected = step0 + step1
        assert lp.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_masked_steps_ignored(self):
        logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
        picks = torch.tensor([[0, 2, 1]])
        # The third step is a discarded pick; it must contribute 0.
        mask_full = torch.tensor([[True, True, True]])
        mask_short = torch.tensor([[True, True, False]])
        lp_short = _plackett_luce_log_prob(logits.clone(), picks.clone(), mask_short)
        # Recompute with only the first two real picks.
        lp_two = _plackett_luce_log_prob(
            logits.clone(), picks[:, :2].clone(), mask_full[:, :2],
        )
        assert lp_short.item() == pytest.approx(lp_two.item(), abs=1e-5)

    def test_differentiable_in_logits(self):
        logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]], requires_grad=True)
        picks = torch.tensor([[0, 2]])
        mask = torch.tensor([[True, True]])
        lp = _plackett_luce_log_prob(logits, picks, mask)
        lp.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

class TestLosses:
    def test_baseline_is_per_pool_mean(self):
        rewards = torch.tensor([[1.0, 3.0], [10.0, 0.0]])
        log_prob = torch.zeros(2, 2, requires_grad=True)
        entropy = torch.ones(2)
        aux_pred = torch.zeros(2, requires_grad=True)
        losses = _compute_losses(rewards, log_prob, entropy, aux_pred, 0.01, 0.1)
        assert losses.baseline.tolist() == pytest.approx([2.0, 5.0])

    def test_advantage_detached(self):
        rewards = torch.tensor([[1.0, 3.0]])
        log_prob = torch.zeros(1, 2, requires_grad=True)
        entropy = torch.ones(1)
        aux_pred = torch.zeros(1, requires_grad=True)
        losses = _compute_losses(rewards, log_prob, entropy, aux_pred, 0.01, 0.1)
        assert not losses.advantage.requires_grad

    def test_zero_advantage_zero_policy_gradient(self):
        # All sampled decks score equally -> advantage all zero -> no policy grad.
        rewards = torch.full((1, 4), 5.0)
        log_prob = torch.randn(1, 4, requires_grad=True)
        entropy = torch.ones(1)
        aux_pred = torch.zeros(1, requires_grad=True)
        losses = _compute_losses(rewards, log_prob, entropy, aux_pred, 0.0, 0.0)
        losses.policy_loss.backward()
        assert torch.allclose(log_prob.grad, torch.zeros_like(log_prob.grad))

    def test_aux_target_is_detached_baseline(self):
        rewards = torch.tensor([[2.0, 4.0]], requires_grad=True)
        log_prob = torch.zeros(1, 2)
        entropy = torch.ones(1)
        aux_pred = torch.zeros(1, requires_grad=True)
        losses = _compute_losses(rewards, log_prob, entropy, aux_pred, 0.0, 1.0)
        losses.aux_loss.backward()
        # Gradient flows to aux_pred, not back into rewards (target detached).
        assert aux_pred.grad is not None
        assert rewards.grad is None

    def test_normalize_advantage_unit_variance_per_pool(self):
        # GRPO normalization: each pool's advantage is zero-mean, unit-std
        # (population), independent of that pool's raw reward spread.
        rewards = torch.tensor([[1.0, 3.0], [100.0, 0.0]])
        log_prob = torch.zeros(2, 2)
        losses = _compute_losses(
            rewards, log_prob, torch.ones(2), torch.zeros(2), 0.0, 0.0,
            normalize_advantage=True,
        )
        adv = losses.advantage
        assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-5)
        assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(2), atol=1e-3)
        # The aux/baseline target stays the raw per-pool mean (unnormalized).
        assert losses.baseline.tolist() == pytest.approx([2.0, 50.0])

    def test_normalize_advantage_degenerate_pool_no_nan(self):
        # All samples equal -> std 0; the eps floor keeps the advantage finite
        # (and ~0, since the numerator is 0) rather than NaN/inf.
        rewards = torch.full((1, 4), 5.0)
        losses = _compute_losses(
            rewards, torch.zeros(1, 4), torch.ones(1), torch.zeros(1), 0.0, 0.0,
            normalize_advantage=True,
        )
        assert torch.isfinite(losses.advantage).all()
        assert torch.allclose(losses.advantage, torch.zeros_like(losses.advantage))

    def test_topk_objective_maximizes_selected_log_probs(self):
        # topk keeps the k highest-reward decks per pool and maximizes their
        # log-prob (max-likelihood), independent of any advantage/baseline.
        rewards = torch.tensor([[1.0, 9.0, 2.0, 8.0]])  # top-2 are idx 1, 3
        log_prob = torch.tensor([[-5.0, -1.0, -4.0, -2.0]])
        losses = _compute_losses(
            rewards, log_prob, torch.ones(1), torch.zeros(1), 0.0, 0.0,
            objective="topk", topk=2,
        )
        # policy_loss = -mean(log_prob[selected]) = -mean([-1.0, -2.0]) = 1.5
        assert losses.policy_loss.item() == pytest.approx(1.5)

    def test_topk_objective_ignores_advantage_magnitude(self):
        # Scale-invariance: scaling the rewards (without changing their order)
        # leaves the topk policy loss unchanged (only the ranking matters).
        log_prob = torch.tensor([[-5.0, -1.0, -4.0, -2.0]])
        base = torch.tensor([[1.0, 9.0, 2.0, 8.0]])
        a = _compute_losses(
            base, log_prob, torch.ones(1), torch.zeros(1), 0.0, 0.0,
            objective="topk", topk=2,
        )
        b = _compute_losses(
            base * 1000.0, log_prob, torch.ones(1), torch.zeros(1), 0.0, 0.0,
            objective="topk", topk=2,
        )
        assert a.policy_loss.item() == pytest.approx(b.policy_loss.item())

    def test_topk_clamps_to_n_samples(self):
        # k larger than the sample count keeps all samples without error.
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        log_prob = torch.tensor([[-1.0, -2.0, -3.0]])
        losses = _compute_losses(
            rewards, log_prob, torch.ones(1), torch.zeros(1), 0.0, 0.0,
            objective="topk", topk=99,
        )
        # All three kept: -mean([-1, -2, -3]) = 2.0
        assert losses.policy_loss.item() == pytest.approx(2.0)

    def test_topk_entropy_and_aux_match_reinforce(self):
        # Only the policy term differs between objectives; entropy and aux are
        # computed identically.
        rewards = torch.tensor([[1.0, 9.0, 2.0, 8.0]])
        log_prob = torch.randn(1, 4)
        entropy = torch.tensor([2.5])
        aux_pred = torch.tensor([3.0])
        rein = _compute_losses(
            rewards, log_prob, entropy, aux_pred, 0.01, 0.1, objective="reinforce",
        )
        topk = _compute_losses(
            rewards, log_prob, entropy, aux_pred, 0.01, 0.1,
            objective="topk", topk=2,
        )
        assert topk.entropy_loss.item() == pytest.approx(rein.entropy_loss.item())
        assert topk.aux_loss.item() == pytest.approx(rein.aux_loss.item())


# --------------------------------------------------------------------------- #
# Entropy schedule / early stop / KL
# --------------------------------------------------------------------------- #

class TestEntropySchedule:
    def test_held_constant_then_decays_on_plateau(self):
        sched = _EntropySchedule(initial=0.01, decay_after=3, factor=0.9)
        # 4 monotonically improving epochs: coef unchanged (arming happens but
        # each epoch also sets a new best, so no decay yet).
        for r in [1.0, 2.0, 3.0, 4.0]:
            coef = sched.update(r)
            assert coef == pytest.approx(0.01)
        # Now a plateau (no new best) after arming -> decay.
        coef = sched.update(3.5)
        assert coef == pytest.approx(0.009)
        coef = sched.update(3.4)
        assert coef == pytest.approx(0.0081)

    def test_no_decay_before_arming(self):
        sched = _EntropySchedule(initial=0.05, decay_after=5, factor=0.9)
        # Plateau immediately, never armed -> no decay.
        for r in [1.0, 0.5, 0.4]:
            coef = sched.update(r)
        assert coef == pytest.approx(0.05)


class TestEarlyStop:
    def test_fires_at_patience(self):
        assert not _should_stop(2, 3)
        assert _should_stop(3, 3)
        assert _should_stop(5, 3)


class TestKL:
    def test_zero_when_identical(self):
        logits = torch.randn(2, 6)
        pool_mask = torch.ones(2, 6, dtype=torch.bool)
        kl = kl_divergence(logits, logits.clone(), pool_mask).mean()
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_different(self):
        a = torch.tensor([[3.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 0.0, 3.0]])
        pool_mask = torch.ones(1, 3, dtype=torch.bool)
        assert kl_divergence(a, b, pool_mask).mean().item() > 0.0

    def test_gradient_finite_with_padding(self):
        # Masked positions make logp - logq = -inf - -inf = nan; the backward
        # must not propagate that nan into the logit gradient.
        logits = torch.randn(2, 6, requires_grad=True)
        ref = torch.randn(2, 6)
        pool_mask = torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, True, True, False, False]],
        )
        kl_divergence(logits, ref, pool_mask).mean().backward()
        assert torch.isfinite(logits.grad).all()


class TestEntropyMasking:
    def test_padding_excluded(self):
        logits = torch.tensor([[1.0, 2.0, 100.0]])
        full = torch.ones(1, 3, dtype=torch.bool)
        masked = torch.tensor([[True, True, False]])
        ent_full = policy_entropy(logits, full)
        ent_masked = policy_entropy(logits, masked)
        # Dropping the high-logit padded position changes the entropy and keeps
        # it finite (no nan from -inf * 0).
        assert torch.isfinite(ent_masked).all()
        assert ent_masked.item() != ent_full.item()

    def test_gradient_finite_with_padding(self):
        # Regression guard: masked positions have logp = -inf, p = 0, so the
        # entropy product is 0 * -inf = nan. The forward value is masked clean,
        # but autograd would otherwise backpropagate nan into every parameter
        # (which NaN-poisoned the picker on its first optimizer step).
        logits = torch.randn(2, 6, requires_grad=True)
        pool_mask = torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, True, True, False, False]],
        )
        policy_entropy(logits, pool_mask).sum().backward()
        assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------- #
# Pool split + shuffle determinism
# --------------------------------------------------------------------------- #

def _fake_pool(i: int) -> PreparedPool:
    arr = np.zeros((23, 4), dtype=np.float32)
    return PreparedPool(f"SET{i}", [f"c{i}"], arr, np.zeros(23, dtype=bool))


class TestSplitAndShuffle:
    def test_front_slice_is_validation(self):
        pools = [_fake_pool(i) for i in range(10)]
        val, train = _split_pools(pools, 0.2)
        assert [p.set_code for p in val] == ["SET0", "SET1"]
        assert [p.set_code for p in train] == [f"SET{i}" for i in range(2, 10)]

    def test_seed_42_deterministic_order(self):
        pools = [_fake_pool(i) for i in range(20)]
        _, train = _split_pools(pools, 0.2)
        order_a = [p.set_code for p in _shuffle_train(train, random.Random(RANDOM_SEED))]
        order_b = [p.set_code for p in _shuffle_train(train, random.Random(RANDOM_SEED))]
        assert order_a == order_b
        # The shuffle does reorder (not the identity), and the val slice is
        # untouched by the shuffle (operates on train only).
        assert order_a != [p.set_code for p in train]


# --------------------------------------------------------------------------- #
# CLI validation
# --------------------------------------------------------------------------- #

def _run_cli(argv: list[str]) -> int:
    from sealed.infrastructure.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


class TestCliValidation:
    def test_resume_and_picker_checkpoint_mutually_exclusive(self, capsys):
        code = _run_cli([
            "train-picker", "--pools-path", "p.txt",
            "--resume", "a.pt", "--picker-checkpoint", "b.pt",
        ])
        assert code == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_architecture_flag_rejected_on_resume(self, capsys):
        code = _run_cli([
            "train-picker", "--pools-path", "p.txt",
            "--resume", "a.pt", "--n-layers", "6",
        ])
        assert code == 2
        assert "n-layers" in capsys.readouterr().err

    def test_kl_requires_picker_checkpoint(self, capsys):
        code = _run_cli([
            "train-picker", "--pools-path", "p.txt", "--kl-coef", "0.1",
        ])
        assert code == 2
        assert "picker-checkpoint" in capsys.readouterr().err

    def test_missing_pools_path(self, capsys):
        code = _run_cli(["train-picker", "--scorer-checkpoint", "s.pt"])
        assert code == 2
        assert "pools-path" in capsys.readouterr().err

    def test_missing_scorer_fails_fast(self, tmp_path, capsys):
        pools = tmp_path / "pools.txt"
        pools.write_text("SET;A|B\n", encoding="utf-8")
        code = _run_cli([
            "train-picker", "--pools-path", str(pools),
            "--scorer-checkpoint", str(tmp_path / "nope.pt"),
        ])
        assert code == 2
        assert "scorer checkpoint not found" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# End-to-end use-case smoke test (CPU, tiny fixtures)
# --------------------------------------------------------------------------- #

WIDTH = 40  # 8 text + 32 deterministic features


def _make_scorer_checkpoint(path, width=WIDTH):
    from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
    from sealed.infrastructure.scorer_store import ScorerStore
    cfg = ScorerConfig(
        d_model=width, n_layers=1, n_heads=2, n_seeds=2, d_ff=32,
        mlp_hidden=16, dropout=0.0,
    )
    model = SetTransformerScorer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ScorerStore().save_checkpoint(
        model, opt, epoch=0, best_val_accuracy=0.5, config=cfg, path=path,
    )


def _make_cards(cards_dir, n_spells=25, n_lands=5):
    from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND
    letter_dir = cards_dir / "c"
    letter_dir.mkdir(parents=True)
    names = []
    rng = np.random.default_rng(0)
    for i in range(n_spells + n_lands):
        emb = rng.standard_normal(WIDTH).astype(np.float32)
        emb[-FEATURE_COUNT:] = 0.0
        if i >= n_spells:
            emb[-FEATURE_COUNT + IS_LAND] = 1.0
        name = f"card{i:02d}"
        np.savez(letter_dir / f"{name}.npz", embedding=emb)
        names.append(name)
    return names


class TestUseCaseEndToEnd:
    def test_full_run_writes_checkpoints(self, tmp_path):
        scorer_path = tmp_path / "scorer.pt"
        cards_dir = tmp_path / "cardsfolder"
        ckpt_dir = tmp_path / "picker"
        _make_scorer_checkpoint(scorer_path)
        names = _make_cards(cards_dir)

        pools_path = tmp_path / "pools.txt"
        pools_path.write_text(
            "\n".join(f"SET;{'|'.join(names)}" for _ in range(5)) + "\n",
            encoding="utf-8",
        )

        from sealed.application.train_picker import (
            TrainPickerUseCase,
        )
        config = TrainPickerConfig(
            pools_path=pools_path,
            scorer_checkpoint=scorer_path,
            cards_path=cards_dir,
            checkpoint_dir=ckpt_dir,
            epochs=2,
            batch_size=2,
            n_samples=4,
            patience=10,
            val_fraction=0.2,
        )
        result = TrainPickerUseCase().execute(config)

        from sealed.infrastructure.picker_store import PickerStore
        latest = ckpt_dir / "latest.pt"
        assert latest.exists()
        assert result.best_path.exists()
        loaded = PickerStore().load_checkpoint(latest)
        assert loaded.config.embedding_dim == WIDTH
        assert np.isfinite(result.best_val_reward)
        assert loaded.train_config is not None

    def test_evals_per_epoch_validates_more_often(self, tmp_path, monkeypatch):
        # With 4 train pools / batch_size 2 = 2 steps/epoch, evals_per_epoch=2
        # validates after every step → 2 evals in one epoch (vs 1 at default).
        import sealed.application.train_picker as tp
        scorer_path = tmp_path / "scorer.pt"
        cards_dir = tmp_path / "cardsfolder"
        _make_scorer_checkpoint(scorer_path)
        names = _make_cards(cards_dir)
        pools_path = tmp_path / "pools.txt"
        pools_path.write_text(
            "\n".join(f"SET;{'|'.join(names)}" for _ in range(5)) + "\n",
            encoding="utf-8",
        )

        calls = {"n": 0}
        real_validate = tp._validate

        def counting(*a, **k):
            calls["n"] += 1
            return real_validate(*a, **k)

        monkeypatch.setattr(tp, "_validate", counting)
        config = TrainPickerConfig(
            pools_path=pools_path, scorer_checkpoint=scorer_path,
            cards_path=cards_dir, checkpoint_dir=tmp_path / "picker",
            epochs=1, batch_size=2, n_samples=4, val_fraction=0.2,
            evals_per_epoch=2, patience=10,
        )
        tp.TrainPickerUseCase().execute(config)
        assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# US3: audits + distributional summaries
# --------------------------------------------------------------------------- #

class TestAuditCorrelation:
    def test_spearman_perfect_and_inverse(self):
        from sealed.application.train_picker import _audit_correlation
        assert _audit_correlation(
            np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]),
        ) == pytest.approx(1.0)
        assert _audit_correlation(
            np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]),
        ) == pytest.approx(-1.0)


class TestDistribSummaries:
    def _emb(self, *, color_idx, mana_value, power, toughness):
        from sealed.domain.card_embedding_layout import (
            COLOR_FLAGS,
            MANA_VALUE,
            POWER,
            TOUGHNESS,
        )
        emb = np.zeros(WIDTH, dtype=np.float32)
        emb[-FEATURE_COUNT + COLOR_FLAGS.start + color_idx] = 1.0
        emb[-FEATURE_COUNT + MANA_VALUE] = mana_value
        emb[-FEATURE_COUNT + POWER] = power
        emb[-FEATURE_COUNT + TOUGHNESS] = toughness
        return emb

    def test_summaries(self):
        from sealed.application.train_picker import _distrib_summaries
        cards = np.stack([
            self._emb(color_idx=0, mana_value=2, power=2, toughness=2),  # W creature
            self._emb(color_idx=1, mana_value=4, power=0, toughness=0),  # U noncreature
            self._emb(color_idx=0, mana_value=6, power=3, toughness=3),  # W creature
        ])
        pools = [PreparedPool("SET", ["a", "b", "c"], cards, np.zeros(3, dtype=bool))]
        summary = _distrib_summaries(pools, [[0, 1, 2]])
        assert summary.colors_mean == pytest.approx(2.0)  # {W, U}
        assert summary.creatures_mean == pytest.approx(2.0)
        assert summary.type_creature_share == pytest.approx(2 / 3)
        assert summary.cmc_hist == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0])


class TestAuditorEndToEnd:
    def test_audit_corr_present_with_auditor(self, tmp_path):
        scorer_path = tmp_path / "scorer.pt"
        auditor_path = tmp_path / "auditor.pt"
        cards_dir = tmp_path / "cardsfolder"
        _make_scorer_checkpoint(scorer_path)
        _make_scorer_checkpoint(auditor_path)
        names = _make_cards(cards_dir)
        pools_path = tmp_path / "pools.txt"
        pools_path.write_text(
            "\n".join(f"SET;{'|'.join(names)}" for _ in range(5)) + "\n",
            encoding="utf-8",
        )
        from sealed.application.train_picker import (
            TrainPickerUseCase,
            _load_scorer,
            _prepare_pools,
            _select_device,
            _validate,
        )
        from sealed.domain.picker_model import PickerConfig, PickerModel
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
        from sealed.infrastructure.pool_file_reader import parse_pools

        # The use case runs cleanly with an auditor configured.
        config = TrainPickerConfig(
            pools_path=pools_path, scorer_checkpoint=scorer_path,
            auditor_scorer_checkpoint=auditor_path, cards_path=cards_dir,
            checkpoint_dir=tmp_path / "picker", epochs=1, batch_size=2, n_samples=4,
        )
        TrainPickerUseCase().execute(config)

        # And _validate reports an audit_corr when an auditor is supplied, None
        # when it is not (FR-030).
        device = _select_device()
        scorer = _load_scorer(scorer_path, device)
        auditor = _load_scorer(auditor_path, device)
        prepared = _prepare_pools(parse_pools(pools_path), ConvertedCardLocator(cards_dir))
        picker = PickerModel(PickerConfig(embedding_dim=WIDTH, d_model=WIDTH, n_heads=8))
        picker.to(device)
        with_aud = _validate(picker, scorer, auditor, prepared, 2, device)
        without_aud = _validate(picker, scorer, None, prepared, 2, device)
        assert with_aud.audit_corr is not None
        assert without_aud.audit_corr is None

    def test_auditor_width_mismatch_fails_fast(self, tmp_path):
        scorer_path = tmp_path / "scorer.pt"
        auditor_path = tmp_path / "auditor.pt"
        cards_dir = tmp_path / "cardsfolder"
        _make_scorer_checkpoint(scorer_path, width=WIDTH)
        _make_scorer_checkpoint(auditor_path, width=36)  # mismatched
        names = _make_cards(cards_dir)
        pools_path = tmp_path / "pools.txt"
        pools_path.write_text(f"SET;{'|'.join(names)}\n", encoding="utf-8")
        from sealed.application.train_picker import TrainPickerUseCase
        config = TrainPickerConfig(
            pools_path=pools_path, scorer_checkpoint=scorer_path,
            auditor_scorer_checkpoint=auditor_path, cards_path=cards_dir,
            checkpoint_dir=tmp_path / "picker", epochs=1, batch_size=1, n_samples=2,
        )
        with pytest.raises(ValueError, match="wide"):
            TrainPickerUseCase().execute(config)

