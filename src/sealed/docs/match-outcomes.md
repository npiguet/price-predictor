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

- A game whose played-card collector could not credit a card to a side (a lobby name other than `p1`/`p2`) writes no rows: `PlayedCardCollector.requireComplete` throws and the worker reports an `Error generating match` line.
- `PoolGenerator.generate` prints to stderr how many pools are short of cards when held-out cards exhaust the replacement boosters.
- The supervisor restarts crashed workers. Long Forge AI games can crash the JVM and that is expected. Why the workers are short-lived by design: `src/effects/docs/collection-runs.md` § Short-lived workers.
- `--workers` is 1 to 12.
- Every status interval the pool kills the longest-running worker and starts a fresh one; with `--effect-records` it passes over a worker still starting Forge, one asked to stop and one already killed.
- With `--effect-records DIR` the workers honour the stop protocol (`specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol): each recycles itself at a game end after `--recycle-seconds`, and the run's end or a first Ctrl-C asks each to finish its game, killing one only after `--stop-timeout`. A second Ctrl-C kills any worker still finishing. A match stopped after one of its games is abandoned: no outcome or cards-played row.
- With `--effect-records DIR` the worker logs go under `DIR/logs/`, never beside the shards, and the supervisor scans them for failure lines at every status line. The run exits 1 when it played matches and wrote no effect record, and stops with exit 1 on workers whose engine is not `patched` with every hook, unless `--allow-degraded` (`src/effects/docs/collection-runs.md` § The workers' engine).
- With `--effect-records DIR`, `--cards-folder` names the converted tree the records' keys join against, and the run exits 1 before any worker starts when that tree's conversion is not at the connector JAR's Forge commit (`src/effects/docs/collection-runs.md` § The conversion the keys join against).
- Without `--effect-records` the workers keep no log, nothing judges their engine or reads a conversion, the pool kills every worker at the end, and `--recycle-seconds`, `--stop-timeout`, `--allow-degraded` or `--cards-folder` exits 2; with it, a missing `--cards-folder` exits 2.

## Directed forks

- `--directed-fork-probabilities FILE` gives the records each listed rule's registered side has and the count a side needs (`specs/025-ability-effect-model-gen3/contracts/cli.md` § `python scripts/rules_catalogue.py directed-rules`); its one writer is `scripts/rules_catalogue.py directed-rules --probabilities-output`, its one reader `infrastructure/directed_fork_probabilities_file.py`, which refuses a file of any other shape, the older map from rule to probability among them.
- With `--effect-records`, each rule's probability is recomputed at every status line's log scan, from the file's records plus the forks the workers' status lines report kept (`domain/directed_fork_needs.py`), and every worker spawned after it starts with those (`MatchOutcomeSupervisor._worker_caps`). A rule held at its starting value for the whole run spends the game's directed budget long after its side has filled.
- The supervisor cannot read the catalogue, so a rule is credited only through the tally, which counts a rule's kept forks and not each side's: a rule whose registered variants span two sides is refused at start (exit 1).
- A worker keeps the probabilities it started with for its whole lifetime, and the tally reaches the supervisor only at the next scan after the match end that printed it. A rule therefore overshoots its count by the forks kept between a scan and the recycling of the workers started before it.
- `run.json` records the starting probabilities in `caps.directed_rule_probabilities` and the last recomputed ones in `final_directed_rule_probabilities`.

## Outputs

- `--effect-records` has no default: collection is an opt-in, and without the flag the two sealed corpora keep their exact format and content.
- The two sealed corpora are appended to `--outcomes-dir`, by default the `--effect-records` directory when given, so a collection writes nothing outside its run directory. Their formats: `src/sealed/CLAUDE.md` § Corpus file formats.
