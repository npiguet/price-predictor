

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
