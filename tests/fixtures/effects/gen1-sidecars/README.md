Gen-1 sidecars and converted texts, copied from a full `python -m price_predictor convert` run of
the feature-023 converter. Real output, never hand-built; read by the tokenizer, holdout,
effect-family and kept-aside-tree tests.

`d/deadly_dispute` and `u/unexpected_windfall` were since replaced by the current converter's
output (sidecar and converted text): their old sidecars declared the spell's key on both the
additional-cost line and the spell line, which a sidecar reader now refuses. Each is now one spell
line.
