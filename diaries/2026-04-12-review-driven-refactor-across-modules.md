# April 12, 2026 — Review-driven refactor across modules

**TL;DR:** Two full code reviews of `price_predictor` (from separate sessions)
each turned into a large refactor plan, both executed end-to-end the same
day. A third session rebuilt CLAUDE.md from scratch.

The day started with a code review of the `price_predictor` module. Two
independent sessions reached similar diagnoses independently: `cli.py` had
grown to 800-odd lines of near-duplicate handlers, the card-name-to-printing
lookup was copy-pasted across five call sites, domain concepts like
`PriceBucket` and `BucketScheme` existed in a dedicated file but were
bypassed by inline hardcoded constants in both the trainer and the evaluator,
and the `_run_epoch` wrapper was dead code kept alive only by tests that
mocked it instead of the real thing.

The first refactor session attacked these in nine phases: introducing
`CardNameResolver` to consolidate the five lookup sites, publicising
`tokenize` and `extract_card_name` on the tokenizer, extracting a shared
`compute_regression_metrics` helper, restructuring the CLI into named
commands with shared-arg helpers, splitting `_train_loop` into
`_run_training_epoch` / `_run_eval_epoch`, deleting `get_cheapest_price` and
`build_price_map`, wiring `SAMPLING_BUCKETS` and `REPORTING_BUCKETS` into the
train and evaluate paths that had been ignoring them, making
`FeatureEngineering` a proper sklearn `BaseEstimator` with sklearn-convention
attribute names, and replacing the two heavily-mocked `_run_epoch` tests with
focused tests on real tensors. That last change took the test count from 526
to 529 passed while removing more than 1500 lines of production code (27
files, +1082 / -1545 in the commit).

The second refactor session was broader and went deeper into structural debt.
It extracted a `card_taxonomy.py` domain module so that `KNOWN_CARD_TYPES`,
`KNOWN_SUPERTYPES`, and `VALID_LAYOUTS` lived in one place instead of being
scattered across entities, parser, and feature engineering. It added
`release_year` as a normalized feature — the year the card was first printed,
normalized to [0,1] against a fixed MIN/MAX window — which bumped the dense
feature width from 18 to 19 and required updating every hard-coded count
assertion in the tests. It extracted `ConvertedCardDataset` and
`load_training_samples` so the four call sites (sklearn train, sklearn eval,
transformer train, transformer eval) stopped reimplementing the same
directory-scan-and-card-resolve loop. It extracted a `transformer_inference`
module to consolidate `_encode_and_pool`, `predict_batch`, and
`predict_shifted_log`, which had been duplicated across the predict CLI, the
evaluate pipeline, and the server. It decomposed several long functions,
cleaned up the server endpoint, and in the final commit deleted the dead
`run_eval` HTTP-client function that was defined but never wired to any
subcommand.

A shorter third session rebuilt CLAUDE.md from scratch, dropping a stale
duplicate "Active Technologies" section and writing a tighter architecture
overview that covers both Python packages and the forge-connector roles.

The pre-existing test failures (stale CLI default assertions, a
`cannot import name 'run_eval'` error in an integration test) showed up in
both refactor sessions and were fixed as part of the cleanup pass rather than
being left as known failures.
