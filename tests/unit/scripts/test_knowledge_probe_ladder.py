"""The knowledge probes' share, folds and probe ladders."""

from __future__ import annotations

import numpy as np
import pytest
from effect_knowledge_probes import (
    labels,
    ladder,
)

# ── the share and its minimum gap ───────────────────────────────────────


class TestShare:
    @pytest.mark.parametrize("rungs, expected", [
        pytest.param((0.5, 0.7, 0.9), 0.5, id="the vector's part of the gap"),
        pytest.param((0.60, 0.70, 0.60 + ladder.MIN_GAP - 1e-6), None,
                     id="not reported under the minimum gap"),
        pytest.param((0.60, 0.61, 0.60 + ladder.MIN_GAP + 1e-9), 0.01 / (ladder.MIN_GAP + 1e-9),
                     id="reported at the minimum gap"),
        pytest.param((0.5, 0.95, 0.8), 1.5, id="above one is kept"),
        pytest.param((float("nan"), 0.7, 0.9), None, id="an undefined rung gives none"),
    ])
    def test_the_share(self, rungs, expected):
        value = ladder.share(*rungs)
        assert value == (None if expected is None else pytest.approx(expected))

    def test_the_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        n = 400
        groups = np.array([f"t{i // 4}" for i in range(n)], dtype=object)
        y = rng.integers(0, 2, n).astype(float)
        oof0 = rng.normal(size=n)
        oof1 = y + rng.normal(scale=1.0, size=n)
        pred3 = y + rng.normal(scale=0.5, size=n)
        point = ladder.share(ladder.score("binary", y, oof0), ladder.score("binary", y, oof1),
                             ladder.score("binary", y, pred3))
        low, high = ladder.bootstrap_share(
            ladder.BootstrapColumns("binary", y, oof0, oof1, pred3, groups),
            ladder.BootstrapOptions(resamples=200))
        assert low <= point <= high

    def test_run_ladder_reports_each_probe_type_s_own_share(self):
        rng = np.random.default_rng(1)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        noise = rng.normal(size=(n, 3))
        data = ladder.LadderInput(
            kind="binary", y=y,
            groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            features={"0": noise, "1": np.column_stack([noise, y + rng.normal(size=n)])},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options=ladder.MlpOptions(hidden=8, epochs=3, device="cpu"),
                                   bootstrap=50)
        assert set(result["rungs"]) == {"0", "1", "3"}
        assert set(result["share"]) == {"linear", "mlp"}
        assert 0.0 < result["share"]["linear"]["value"] < 1.5

    def test_no_share_is_reported_when_the_model_knows_nothing_more(self):
        rng = np.random.default_rng(2)
        n = 1000
        y = rng.integers(0, 2, n).astype(float)
        data = ladder.LadderInput(
            kind="binary", y=y, groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            # The board alone reads the label as well as the model does.
            features={"0": np.column_stack([y + rng.normal(scale=0.3, size=n),
                                            rng.normal(size=n)]),
                      "1": rng.normal(size=(n, 2))},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options=ladder.MlpOptions(
                                       hidden=16, epochs=150, device="cpu"),
                                   bootstrap=0)
        assert result["share"] is None


# ── folds ───────────────────────────────────────────────────────────────


class TestFolds:
    def test_no_group_falls_on_both_sides(self):
        groups = [f"text-{i % 37}" for i in range(500)]
        folds = ladder.fold_assignment(groups)
        seen: dict[str, set[int]] = {}
        for group, fold in zip(groups, folds):
            seen.setdefault(group, set()).add(int(fold))
        assert all(len(f) == 1 for f in seen.values())
        assert set(folds.tolist()) == set(range(ladder.FOLDS))

    def test_assignment_is_a_function_of_the_names_alone(self):
        groups = ["b", "a", "c", "a", "d", "e", "f"]
        first = ladder.fold_assignment(groups)
        again = ladder.fold_assignment(list(reversed(groups)))[::-1]
        assert first.tolist() == again.tolist()

    def test_pooled_items_are_grouped_by_card(self, records, items_by_record):
        by_record = {r.record_id: r for r in records}
        for record_id, items in items_by_record.items():
            for item in items:
                if item.entity is None:
                    continue
                entity = by_record[record_id].state.entity(item.entity)
                assert item.group == labels.card_of(entity)

    def test_the_width_control_is_one_vector_per_text(self):
        a = ladder.width_vector("SP$ Draw | NumCards$ 1", 16)
        assert np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 1", 16))
        assert not np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 2", 16))


# ── the probes ──────────────────────────────────────────────────────────


class TestProbes:
    def test_the_mlp_reads_a_non_linear_label_on_cpu(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(600, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        folds = ladder.fold_assignment([str(i) for i in range(600)])
        mlp = ladder.mlp_oof(ladder.ProbeProblem(X, y, folds, "binary"),
                             ladder.MlpOptions(hidden=32, epochs=60, device="cpu"))
        linear = ladder.linear_oof(ladder.ProbeProblem(X, y, folds, "binary"))
        assert ladder.score("binary", y, mlp) > 0.8
        assert ladder.score("binary", y, linear) < 0.65

    def test_an_mlp_amount_stays_within_a_span_of_the_training_range(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(60, 2))
        X[-1] = 1e6
        y = rng.uniform(0, 3, size=60)
        folds = np.array([0] * 59 + [1])
        prediction = ladder.mlp_oof(ladder.ProbeProblem(X, y, folds, "amount"),
                                    ladder.MlpOptions(hidden=8, epochs=5, device="cpu"))
        low, high = y[:59].min(), y[:59].max()
        span = max(high - low, 1.0)
        assert low - span - 1e-6 <= prediction[-1] <= high + span + 1e-6

    def test_the_linear_probe_reads_an_amount(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(300, 3))
        y = 2 * X[:, 0] + rng.normal(scale=0.1, size=300)
        folds = ladder.fold_assignment([str(i) for i in range(300)])
        prediction = ladder.linear_oof(ladder.ProbeProblem(X, y, folds, "amount"))
        assert ladder.score("amount", y, prediction) > 0.95
