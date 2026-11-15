# `match-outcomes`

Rules to respect when running or changing `match-outcomes`, its supervisor or its workers. Flags and defaults are `python -m sealed match-outcomes --help`; the command is specified in `specs/2026-04-23-distributed-match-generation.md`, and its effect-record collection in `specs/025-ability-effect-model-gen3/contracts/cli.md`. A collection run (`--effect-records`) also follows `src/effects/docs/collection-runs.md`.

## Deck choice

- Each match chooses its decks independently:
  - Side A is sampled from `--side-a-decks` if given, else built by one of the four Forge methods (weights 4:3:2:1) on a fresh pool from a random sealed-legal set.
  - Side B is rolled between the four Forge methods (combined weight 10) on a fresh pool of deck A's set and, if `--side-b-decks` is given, that file (weight `--side-b-decks-weight`) filtered to deck A's set.
- Mirror matches are excluded by equality of the card-name multiset. If no non-mirror file deck exists for deck A's set, the file roll falls back to the Forge methods.
- With both flags absent the command must reproduce the original behaviour: four Forge methods on both sides, random eligible set.
- `--best-of` must be a positive odd integer.
- With `--exclude-cards`, no collected game holds a held-out card. Without it, the same command collects the full-strength games that become the effect model's card-disjoint validation stratum.

## Workers

- The supervisor restarts crashed workers. Long Forge AI games can crash the JVM and that is expected. Why the workers are short-lived by design: `src/effects/docs/collection-runs.md` § Short-lived workers.
- `--workers` is 1 to 12.
- Every status interval the pool kills the longest-running worker and starts a fresh one; with `--effect-records` it passes over a worker still starting Forge, one asked to stop and one already killed.
- With `--effect-records DIR` the workers honour the stop protocol (`specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol): each recycles itself at a game end after `--recycle-seconds`, and the run's end or a first Ctrl-C asks each to finish its game, killing one only after `--stop-timeout`. A second Ctrl-C kills any worker still finishing. A match stopped after one of its games is abandoned: no outcome or cards-played row.
- With `--effect-records DIR` the worker logs go under `DIR/logs/`, never beside the shards, and the supervisor scans them for failure lines at every status line. The run exits 1 when it played matches and wrote no effect record.
- Without `--effect-records` the workers keep no log, the pool kills every worker at the end, and `--recycle-seconds` or `--stop-timeout` exits 2.

## Outputs

- `--effect-records` has no default: collection is an opt-in, and without the flag the two sealed corpora keep their exact format and content.
- The two sealed corpora are appended to `--outcomes-dir`, by default the `--effect-records` directory when given, so a collection writes nothing outside its run directory. Their formats: `src/sealed/CLAUDE.md` § Corpus file formats.
