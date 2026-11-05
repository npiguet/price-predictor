

def test_the_relation_tables_norms_are_reported_beside_the_groups():
    """T104: each table set's pre-clip norm, read with the group norms."""
    import torch

    from effects.application.training_log import format_norms
    from effects.application.training_loop import _table_norms, relation_tables
    from effects.domain.effect_model import EffectModel, EffectModelConfig

    model = EffectModel(EffectModelConfig(global_features=2, act_features=2, player_features=2,
                                          card_features=2, d_model=8, n_heads=2, n_layers=2,
                                          ff_dim=16))
    tables = relation_tables(model)
    assert {name: len(params) for name, params in tables.items()} == {
        "rel_k_norm": 2, "rel_v_norm": 2}
    for params in tables.values():
        for param in params:
            param.grad = torch.full_like(param, 0.5)
    norms = _table_norms(tables)
    expected = (2 * tables["rel_k_norm"][0].numel()) ** 0.5 * 0.5
    assert abs(norms["rel_k_norm"].item() - expected) < 1e-4
    line = format_norms({"encoder": 1.0, "head": 2.0, **{k: v.item() for k, v in norms.items()}})
    assert "rel_k_norm" in line and "rel_v_norm" in line


def test_the_control_line_prints_each_stratums_components_then_control_and_best():
    """FR-075, contracts/cli.md: per stratum probes, grids, rule_tests,
    deviance and score, then the control score and the run's best."""
    from effects.application.control_score import ControlScore, StratumScore
    from effects.application.run_reporter import format_control

    def stratum(values):
        return StratumScore(dict(zip(("probes", "grids", "rule_tests", "deviance"), values)), {})

    control = ControlScore({"card-disjoint": stratum((0.4, 0.2, 0.6, 0.0)),
                            "game-disjoint": stratum((0.1, 0.1, 0.1, 0.1))}, 0.5)
    line = format_control(control, 0.33)
    assert line == (
        "card-disjoint probes 0.400 grids 0.200 rule_tests 0.600 deviance 0.000 score 0.300"
        " | game-disjoint probes 0.100 grids 0.100 rule_tests 0.100 deviance 0.100 score 0.100"
        " | control 0.200 | best 0.330")
    assert format_control(control, None).endswith("| best none")
