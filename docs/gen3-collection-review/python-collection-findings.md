# Python collection supervisors review (2026-11-09): verified findings

Two fresh reviewers each read one slice of the Python side of collection:
1. settings and run control: `sealed/infrastructure/effect_collection_flags.py`, `sealed/application/match_outcomes.py`, the
   match-outcomes command, `price_predictor/infrastructure/forge_jvm.py`, `effects/infrastructure/collector_connector.py`,
   `one_shot_workers.py`;
2. deck construction, card choice and holdout: `effects/application/collect_coverage.py`, `collect_variants.py`,
   `collector_common.py`, `holdout_cards.py`, `effects/domain/collection_caps.py`, `coverage.py`, `script_variants.py`,
   `text_holdout.py`, `deck_file.py`, and the effects CLI wiring.

Each slice's findings then went to a separate adversarial verifier. The verifiers re-traced the findings in the Python code,
the connector and the Forge `effect-record-hooks` source. They also ran light read-only scripts:
- a replay of `generate_variants` over the real Forge cardsfolder;
- counts over the real gen-3 shard `records-gen3/depleted/c51d1362-…3-mw7jmgkd7l0p.jsonl.gz` (4,864 records, 25 games);
- a sample of six `records-gen3/coverage-held-out/` shards;
- a scan of the old coverage worker logs.

Both reviewers found three of the findings independently (R1, R3 and R7); each of those is written once below. Line numbers are
as of repo HEAD 67a4e924, connector JAR source 2acdc5d7 and Forge HEAD 04e72e0518e; re-check them before editing.

Paths:
- Python paths are under `src/`;
- connector classes are under `forge-connector/src/main/java/com/pricepredictor/connector/`;
- Forge paths are under `../forge`.

Verdicts: CONFIRMED (traced end to end), PARTLY, NEEDS-TEST, REFUTED. "Corpus impact" names what the defect does to the
collected corpora or the run.

What the reviewers found sound:
- **Every setting reaches the JVM as Java reads it.** Every `-Deffect.*` property Python writes is spelled the way
  `CollectionCaps.java` reads it, and the defaults agree (sealed `cli.py` 1007-1048, `effects/domain/collection_caps.py` 68-90,
  `CollectionCaps.java` 60-120).
- **No leak into the sealed corpus.** Coverage and variant games cannot reach `match-outcomes.txt` or `cards-played.txt`:
  `CollectorSupervisor.start_worker` passes `output_file=None`, and `MatchWorkerMain` 223 builds no `SealedCorpus`.
- **Unique ids.** Ids are namespaced per JVM lifetime.
- **Truncated shards.** A truncated last gzip member is read up to the cut.
- **One holdout logic.** The holdout code (`text_keyed_holdout`, `select_holdout`, `holds_text`) is shared by holdout-cards,
  build-corpus, collect-variants and the held-out round.

---

## Slice 1: settings and run control

### R1: `loadable.txt` in the variant tree is staged as a card script and stops every Forge startup on the machine. CONFIRMED (rank 1 overall)
- **Mechanism:**
  - `VariantSidecarMain` (60, 97-110) stages the variant tree and only then writes `variants/loadable.txt`, the list of
    variant names Forge loaded. Python reads that list as `collect_variants.LOADABLE_FILE` (130, 142).
  - On any later run over the same tree, `ForgeEnvironmentInitializer.stageCustomCards` (131-159) copies every file
    `ScriptFiles.under` returns. `ScriptFiles` 15-20 keeps any `.txt`, so `loadable.txt` is copied into Forge's custom-cards
    directory and listed in the staging manifest.
  - Forge reads every `.txt` there (`CardStorageReader.collectCardFiles` 355-363). The list has no `Name:` line, so
    `CardRules.Reader.readCard` builds no face, and `getCard` (`CardRules.java` 639-662) builds rules with `mainPart = null`.
    `StaticData.java` 115-126 calls `card.getName()`, which dereferences `mainPart` and throws a NullPointerException. Nothing
    catches it (`FModel` 81-82).
  - The manifest is written before `FModel.initialize`, so the failed run leaves the staged copy in place. From then on every
    Forge JVM on the machine dies at startup: match-outcomes, collect-coverage, draft and evaluation. The pool restarts each
    one forever (R2).
  - The next staging run deletes the copy, then copies it back from the tree. The run is poisoned until someone deletes
    `loadable.txt` by hand.
  - The connector's own `STAGED_MANIFEST` comment (162-166) describes this failure for a `.txt` without `Name:`. That
    failure is why the manifest is a `.lst`.
  - `VariantRegistryTest.aNonScriptFileIsIgnored` tests a `loadable.lst`, so the test has drifted from the code.
- **On disk today:**
  - `output/effects/variant-scripts/loadable.txt` exists: gen-2's list, 15,654 names, beside 15,951 variant scripts.
  - The custom-cards directory (`../forge/forge-gui/data/custom/cards/`) holds only an empty `.staged-by-effects.list`, so
    the machine is not poisoned now.
- **Corpus impact:** no wrong records. `collect-variants` aborts at the sidecar pass, and every later collection command loses
  everything until someone intervenes.
- **Frequency:** certain on the first gen-3 `collect-variants` into the default `--variant-scripts`. On a fresh tree it fires
  on every top-up after the first, and the `--seed` help invites top-ups.
- **Fix:** rename the list to a non-`.txt` name (e.g. `loadable.lst`) on both sides, or write it outside the tree. Make
  `ScriptFiles.under` skip it, or stage only files with a `Name:` line. Before re-collecting, move the old `loadable.txt` aside
  (or point gen-3 at a fresh variant tree).
- **Tests:**
  - Java: a temp tree holding `x_variant_0.txt` and the loadable list; `ScriptFiles.under(tree)` returns no file without a
    `Name:` line, and `ForgeEnvironmentInitializer.initialize(tree)` succeeds twice in a row.
  - Python: `collect_variants.LOADABLE_FILE` equals the Java constant and does not end in `.txt`.

### R2: a worker that cannot start is retried forever. CONFIRMED (spawn failure); PARTLY (exit at startup)
- **Spawn failure** (CONFIRMED):
  - In `forge_jvm.py` 524-526 the monitor thread catches every exception from `spawn_worker`, prints it and loops at once,
    with no backoff or limit.
  - A missing connector JAR raises `FileNotFoundError` in `resolve_connector_jar` (76-85), through `build_forge_classpath`
    from `MatchWorkerConnector.start` (`match_worker_connector.py` 156-159). A missing `java` raises the same from `Popen`.
  - So the `except FileNotFoundError` around `supervisor.run()` in sealed `cli.py` 1838-1842 is dead code.
  - In collect-coverage, the castability consult catches `OSError` (`collect_coverage.py` 467-474), so the run carries on
    into rounds that can never finish.
  - The result is a run that never ends, spins about one core over 12 threads, and collects nothing.
- **Exit at startup** (PARTLY):
  - `forge_jvm.py` 512-522 restarts a worker after any exit, with no limit.
  - Triggers:
    - an unreadable `--exclude-cards` file (`HoldoutCards.fromSystemProperty`, `HoldoutCards.java` 59-71). Python never
      checks that the file exists (sealed `cli.py` 931, 1833);
    - a decks file where no deck loads (`MatchWorkerMain.loadDecks` 363-377);
    - R1's poisoned card database;
    - `WorkerLoop` 118-126, which exits non-zero after 10 consecutive failures.
  - Narrower than stated: the pool prints "Worker i exited (code N), restarting..." on every restart, so it is not silent.
    match-outcomes runs until Ctrl-C by design, so only coverage and variant rounds truly never end.
  - Each restart pays a full Forge init on 12 cores.
- **Corpus impact:** nothing collected, CPU wasted, and the run gives no stop reason.
- **Frequency:** low in a planned run, and certain after R1.
- **Fix:**
  - Before any thread starts, check the JAR, `java` and the `--exclude-cards` file.
  - Stop `run()` with an error after N consecutive spawn failures, or quick exits, in every slot.
  - Add a backoff.
- **Tests:**
  - `ForgeWorkerPool` whose `spawn_worker` raises `FileNotFoundError`: `run()` raises or returns within about 2 s, and
    `spawn_worker` is called a bounded number of times.
  - A fake process whose `wait()` returns 1 at once and a `should_stop` that is never true: `run()` ends with an error after
    N restarts.

### R3: a top-up `collect-variants` overwrites an earlier run's variant under the same name. CONFIRMED (both reviewers)
- **Mechanism:**
  - `collect_variants.py` 277-281 names the file `variant_name(card_name, len(generated))`, which is
    `"{card} Variant {index}"` (`script_variants.py` 144-151), and writes it with `path.write_text` unconditionally.
  - The budget (`variant_budget`, `volume_ceiling` 112-125) reaches every eligible script once the corpus holds five times
    the script count. So `generate_variants` makes one variant per eligible source, in an order shuffled by `--seed`.
  - Two seeds give two permutations of the same set, and a card that lands on the same index gets a different perturbation
    under the same file name.
  - `VariantSidecarMain` then rewrites the sidecars for the whole tree.
- **Measured** (the verifier replayed `generate_variants` over 33,682 Forge scripts, seed 42 against seeds 1-5):
  - 16,589 variants per run;
  - 2, 1, 0, 1 and 3 same-name collisions, with a different text every time (e.g. "Seal of Fire Variant 1475").
- **Corpus impact:**
  - Records of the earlier run join the later run's text, so text and label disagree.
  - The trait layout is unchanged (one parameter changes in place), so the join succeeds silently.
  - The scale is small: one round fills about 11,500 deck slots for about 16k variants, so about 47% of variants have records.
    That is about 0.5 corrupted variants per top-up, each with a handful of records.
- **Frequency:** at least one collision in about 63% of top-ups (R1 blocks top-ups today).
- **Fix:** name variants by a hash of the perturbed script, or put a run token in the name, or refuse to overwrite a file
  whose content differs.
- **Test:** two `generate_variants` calls into one directory with a one-card source and seeds that give different
  perturbations; the first `… Variant 0.txt` keeps its bytes (or the second call raises or picks a fresh name).

### R4: the pool can leave a JVM it never kills. CONFIRMED (a timing race)
- **Mechanism:**
  - The monitor thread spawns a worker and only then appends it to `_processes` under the lock (`forge_jvm.py` 505-510). It
    does not re-check shutdown after appending.
  - The supervisor polls its stop condition right after `_kill_oldest_worker` (548-553, 571), which is exactly when the
    monitor respawns.
  - `_terminate_all` (643-649) kills a snapshot of the list, and `join(timeout=10)` returns regardless.
  - Nothing in the worker watches its parent, and on Windows `start_new_session` does nothing, so the orphan outlives Python.
  - A crash-restart that coincides with stop or Ctrl-C does the same.
- **Corpus impact:**
  - The orphan's records are correct, but they come from the previous round's decks.
  - Its progress lines land in `{run_id}.progress.txt` after the next round truncated the file, so that round ends early.
  - `_ShardCounter` (`collect_coverage.py` 131-166) counts the orphan's shard once, mid-write. Its docstring's premise ("a
    round's workers are all terminated before `play_round` returns") is false here.
  - The orphan is never recycled, so it can sit in a runaway Forge loop.
- **Frequency:** low, at most about 0.5% of rounds.
- **Fix:** after appending under the lock, `_monitor_worker` re-checks `_shutdown_event` and kills the new process if it is
  set. Alternatively, spawn under the lock with a terminating flag.
- **Test:** a fake `spawn_worker` that blocks on an Event released inside a patched `_terminate_all`; every process spawned is
  killed when `run()` returns.

### R5: shard counting ignores `--workers` and uses every core. CONFIRMED
- **Where:** `CoverageRounds.count_new_shards` (`collect_coverage.py` 275-277) and the initial `counter.update` (418, 681) pass
  no `workers`. So `run_parallel` uses `os.process_cpu_count()`, which is 32 here, capped at the number of fresh shards
  (`parallel_map.py` 66-71).
- **Impact:** the first count over an existing corpus runs 32 processes, above the machine's 12-process budget. The cost is
  CPU only.
- **Fix and test:** pass `config.workers`; assert that `counter.update` receives `workers=config.workers`.

### R6: `--probe-keywords` is not validated; a misspelling silently gets no probes. CONFIRMED
- **Where:**
  - `effect_collection_flags.py` 105-111 accepts any string, and 345-362 announces it as enabled.
  - Java only trims the values (`WorkerProperties.csv` 117-128) and matches them exactly against `KeywordEdit.normalize`
    (lowercase, spaces become `_`): `ProbeScheduler` 120-121, `ForkBudget.mayProbe` 88-90.
- **Impact:** `Trample` or `first strike` is never probed, so gate 2 is under-sampled for that keyword.
- **Frequency:** only when an operator narrows the list with a non-canonical spelling. The default list is correct.
- **Fix and test:** validate against `EFFECT_PROBE_KEYWORDS` and normalise as the JVM does. A validator refuses (or
  normalises) `"first strike"` and `"Trample"`.

### R7: the held-out round's "full strength" lifts only the special bucket. CONFIRMED (both reviewers; minor)
- **Where:**
  - `held_out_round_caps` (`collect_coverage.py` 568-572) sets only `playability_rate=1.0`.
  - In `PlayabilityChannel.keeps` (101-111) and `DecisionSample` 18-22, the zone and plain rates are absolute shares of all
    candidates, so they stay at 1e-4 and 1e-3.
  - Special candidates go from 1% to 100%. The mana cap and the fork budgets are unchanged, so the docstring ("every record
    the held-out cards produce is kept") is false.
- **Impact:**
  - Special decision records are about 100 times denser in the card-disjoint stratum than in training.
  - Every candidate is now evaluated at every priority stop, which costs time; NEEDS-TEST for how much.
  - Gate 1 reads only resolution records from that stratum, so it is unaffected.
- **Fix and test:** decide what "full strength" means, then either set all three rates to 1.0 or reword the docstring. A test
  pins whichever is chosen.

### R8: staged variants stay in Forge's custom-cards directory after a variant run. PARTLY
- Cleanup happens only at the next staging (`clearPreviouslyStaged`). Workers only register (`MatchWorkerMain` 207-209).
- Narrower than stated:
  - The leftovers cannot enter pools or coverage decks, which are edition-based or explicitly named.
  - A rare whole-database card pick could surface one, keyed `cardsfolder/…` with no sidecar. It would read as an
    unconverted script, not as unjoinable.
  - The out-of-memory risk is unproven: the gen-2 variant run loaded about 16k staged scripts. NEEDS-TEST for the heap under
    `-Xmx1200m`.
- **Fix:** clear the staged set when the variant round ends.

### R9: a Ctrl-C between rounds can be lost. PARTLY; NEEDS-TEST
- The SIGINT handlers are installed per pool and never restored (`forge_jvm.py` 465-466), and `interrupted` is latched right
  after `run()` (`collector_connector.py` 111-112).
- Narrower than stated: counting more than one shard runs in a `ProcessPoolExecutor` whose children keep the default handler.
  A Ctrl-C there aborts the run with `KeyboardInterrupt` or `BrokenProcessPool`. Silent loss is limited to millisecond gaps
  in the parent, or to counts of a single shard.
- **Test:** send SIGINT to the handler after `play_round`; `supervisor.interrupted` is true.

### Only with two runs on one records directory
- **Shared deck files.** The deck files have fixed names: `coverage-decks.txt` (`collect_coverage.py` 305) and
  `variant-decks.txt` (`collect_variants.py` 477). `deck_file.write_deck_file` replaces the file, and every worker JVM reloads
  it at startup (`MatchWorkerMain.runForever` 217-218). One worker is recycled every 60 s, so two runs on one directory play
  each other's decks, count progress in their own progress file, and charge retirement on decks they never played. Fix: put
  `run_id` in the file name.
- **Counting another run's shards.** `_ShardCounter` marks a shard another live run is still writing as counted forever.
  `_play_and_check_growth` (`collect_variants.py` 490-501) counts any new shard, so a concurrent writer can mask an empty
  variant round.
- Both are rare under the one-job-at-a-time practice.

### Not bugs, or decisions for the user
- **Depleted match-outcomes runs also feed the sealed corpus.** Sealed `cli.py` 1801 hard-codes
  `output/sealed/match-outcomes.txt`, so `--exclude-cards` (870 held-out cards) depletes the pools behind the sealed scorer's
  rows too. This is a question of intent.
- **match-outcomes accepts the directed-fork and cap flags without `--effect-records`.** The caps are dropped in `start()`
  (137-141), while `directed_fork_error` still runs (sealed `cli.py` 1793-1799). This is UX only.
- **`sum_dropped_events` accepts a count truncated by a kill** (`effect_collection_flags.py` 385). Trivial: the count is
  diagnostic.

### Code quality (slice 1)
- **Two hand-built `-D` maps.** They live in sealed `cli.py` 1007-1048 and `collection_caps.py` 68-90, and only a test keeps
  them equal (`test_the_default_values_match_property_for_property`). Build both from one map of named property constants.
- **Java defaults only partly pinned.** The three rates and the directed-fork values are pinned (`test_collection_caps.py`
  213-223, 308-317). The mana cap, interventions, probes per game and seat values are not.
- **Scattered startup validation.** Two CLIs call the checks in different orders, and neither checks the JAR,
  `--exclude-cards` or `--probe-keywords`. Use one `CollectionSettings` value with a single `refusal()`.
- **`MatchWorkerConnector.start`** takes 14 parameters, including a `decks_only` boolean that hides two launch modes. Use two
  named constructors.
- **`MatchOutcomeSupervisor`** re-exports pool internals for tests only (`match_outcomes.py` 148-171).
- **History in comments.** `forge_jvm.py` 319 and 347 cite `final-fix-N.md`, and 619-620 cite a commit. Keep the invariants
  and drop the history.

---

## Slice 2: deck construction, card choice and holdout

### D1: coverage credits `turn_action` records, including every card drawn. CONFIRMED, worse than reported (rank 1 in this slice)
- **Mechanism:** `count_shard` (`collect_coverage.py` 118-128) applies `qualifying_cards` (`coverage.py` 153-169) to every
  record kind, and `TurnActionPayload.happened_events` (`records.py` 621-632) returns its events.
- **Measured** on the real depleted shard:
  - 921 `turn_action` records: 537 draw, 375 untap, 9 cleanup discard.
  - All 1,500 untap subjects resolve to `state.entities`, and so does the drawn card of each draw's `zone_change`.
  - `turn_action` supplies 1,579 of 4,302 card credits (37%).
- **Corpus impact:**
  - Every card drawn earns a credit. A card the AI never casts therefore makes progress in every round it is drawn: it never
    stalls, never retires, and is eventually marked satisfied at 50 credits with no trainable record.
  - It then leaves the decks and the "uncastable" residue report, though interventions need exactly those cards in games.
  - Untap credits also inflate lands, attackers and tap-ability artifacts.
  - This contradicts `src/effects/CLAUDE.md`, which says mere presence is not counted and `turn_action` records go to no
    stratum.
- **Frequency:** every round of every card-round coverage run. The held-out round counts resolution records only and is
  unaffected.
- **Fix:** count only the record kinds build-corpus keeps (apply `record_refusal`), or at least skip `RecordKind.TURN_ACTION`.
- **Test:** a `turn_action` fixture holding a draw `zone_change` and an untap event whose subjects are entities; `count_shard`
  returns an empty Counter.

### D2: the held-out round credits random-seat games that build-corpus drops. CONFIRMED (loses)
- **Mechanism:**
  - `count_text_shard` (`collect_coverage.py` 481-504) never reads `record.random_seat`.
  - build-corpus collects random-seat games per game (`corpus_survey.py` 453-454) and keeps them out of validation
    (`_candidate_games` 584-594).
  - It drops any random-seat game that names a held-out card (`_fate` 596-603). Every held-out-round game names one, so all
    of them are dropped.
- **Measured:** in six shards of the previous `records-gen3/coverage-held-out/` round, 8 of 29 games were random-seat. The
  flag's default is 0.0 (`effect_collection_flags.py` 36), so that round was run with a share above 0.
- **Corpus impact:** the round reaches `--min-text-games` partly on games the corpus never holds, then reports success. Texts
  end below gate 1's floor of 5.
- **Frequency:** every held-out round run with a random-seat share above 0, which is how the last one ran.
- **Fix:** collect the shard's random-seat games first, then credit only the other games. No game outlives a shard, so one
  shard is enough.
- **Test:** a fixture shard holding one game whose resolution record acts through a held-out key, with `random_seat=True` on
  some record; `count_text_shard` credits nothing.

### D3: with no holdout source, the collectors deck and vary held-out cards. PARTLY (mechanism confirmed; the escape rate is lower than reported)
- **Mechanism:**
  - `DeckHoldout.exclusions` is empty when no flag is given (`collector_common.py` 65-69, 210-214).
  - All three sources (`--exclude-cards`, `--split-from`, `--training-corpus`) are optional, and nothing warns
    (`effects/infrastructure/cli.py` 211-241, `collection_target` 148-154, `collector_config` 476-481).
  - The gen-3 quickstart (`specs/025-…/quickstart.md` 80) shows `collect-variants <run flags>` with no holdout flag.
- **collect-variants:**
  - `generate_variants` (252, 269) then skips nothing.
  - `_keep_unheld` (443-467) computes `trees.holdout(rule)`, but it uses only `drop_held_out_variants`. That function drops a
    variant only when a sidecar line still prints a held-out unit.
  - Under the template unit, a variant escapes only when a selector or protection swap lands on the held-out line itself
    and the card has no other held-out line. Numeric shifts never escape.
  - The verifier's replay over the 870 held-out cards found:
    - 223 get a numeric shift and 267 a swap;
    - so at most about 267 escapes (about 31% of held-out cards), and fewer in practice, since some swaps hit `DeckHints` or
      AI SVars.
  - Under `--holdout-unit text`, 263 of the 401 existing variants of text-held cards would escape.
  - The existing tree holds no variant of a template-held card, so the gen-2 run did pass a source.
  - An escaped variant also passes as board context: `record_names_held_out_card` checks only card and token script files,
    and variant files are in neither.
- **collect-coverage:** with no flag, the card round decks held-out cards (`collect_coverage.py` 401-406). Depleted cards have
  no records, so they carry top weight. build-corpus routes or drops those games, so games are lost; nothing leaks.
- **Corpus impact:** a holdout leak through variants, against the contract that no variant is generated from a held-out card.
  Coverage loses games.
- **Frequency:** every run where the operator omits the flag.
- **Fix:** refuse a collect-variants or coverage run with no holdout source. Or take the exclusions as the union of the list
  and `trees.holdout(rule).names`, and warn when the two differ.
- **Test:** a fixture whose only perturbable line is held out and has `ValidTgts$ Creature`. Run
  `generate_variants(held_out=frozenset())` and then `drop_held_out_variants`; no variant of that card survives. Or the CLI
  refuses the run.

### D4: holdout-cards writes split cards under a face name, so Java depletion misses them. CONFIRMED (loses; 4 cards)
- **Mechanism:**
  - `load_converted_cards` (`card_sources.py` 43-62) takes the first `name:` line (`fire_ice.txt` gives `fire`).
  - `holdout_report` (`holdout_cards.py` 107-111) restores the printed case through `load_first_printings`, which indexes face
    names.
  - Java checks `excluded.contains(card.getName())` (`PoolGenerator` 89, 120). A split card's name is "Fire // Ice"
    (`CardSplitType` COMBINE, `CardRules.java` 200-206), and `HoldoutCards.contains` (75-77) folds case only.
- **Measured:** 126 split-layout converted files. The current list holds 4 split first faces (Beck, Catch, Claim, Push), or 5
  under the text unit.
- **Corpus impact:** these cards stay in depleted sealed pools, and build-corpus routes away the games that hold them. A few
  games are lost; nothing leaks.
- **Fix:** write the combined name for a split card, or have `HoldoutCards.contains` test each face.
- **Test:** `HoldoutCards.of(List.of("Fire")).contains("Fire // Ice")`, or `holdout_report` writes "Fire // Ice" for a held-out
  split card.

### D5 (new, found by the verifier): a static's own continuous records credit no card. CONFIRMED (minor)
- In the measured shard, all 121 `continuous` records credited 0 cards: `refs.source` is null, and the payload holds
  contributions, not events.
- Anthem and lord cards progress only through other records, such as their own spell's resolution.
- **Fix:** credit the host through `record.ability`.
- **Test:** a `continuous` fixture whose ability names a host; `count_shard` credits that card.

### D6: multi-face cards are under-counted. PARTLY (minor); a related name ambiguity NEEDS-TEST
- `qualifying_cards` credits `entity.name`, which is `card.getName()` for the current state (`EntityJson` 73, `Card.java`
  968-979). Units are converted first-face names.
- What credits nothing:
  - a split card in hand ("Fire // Ice");
  - the half "Ice" on the stack;
  - an adventure on the stack;
  - a transformed back face, or an MDFC's back face.
- **Impact:** these cards are over-decked, and some are reported short. No record is wrong.
- **Fix:** credit through the entity's provenance `script_file`, mapped to its converted card.
- **New, NEEDS-TEST:** coverage decks name a split card by its converted first face ("fire"). `DeckMaterializer.lookUp`
  reaches `CardDb.getCard` through `allCardsByName`, which per Forge's own doc (`CardDb.java` 1022-1025) maps "Fire" to both
  "Fire // Ice" and "Start // Fire". The deck may then play a different card from the one the unit counts.
  - Test: materialize a deck naming "Fire", and check which card is played.

### D7: `--variant-volume` caps scripts, not records. CONFIRMED (documentation and behaviour mismatch)
- `variant_budget` (112-114) passes a script count as `limit`. The help text (65-70) says scripts; `src/effects/CLAUDE.md` says
  records.
- On a real corpus the budget exceeds the ~32k source scripts, so every perturbable script gets a variant. Records are bounded
  only because exactly one round of 500 matches is played: about 16k variants for 11,500 deck slots.
- **Fix:** decide which is meant; fix the code or the wording.

### D8: retirement is charged to cards whose decks were never played. PARTLY (minor)
- `GeneratedDecksIndex.randomDeck` (95-97) draws with replacement (`MatchGenerator` 257-297). 1,000 draws over 500 decks leave
  about e⁻² ≈ 13.5% unplayed, yet `retire_stalled` (`coverage.py` 244) charges every card in every written deck.
- It barely matters: in a played deck a card is often never drawn either (about half the time), and retirement needs three
  rounds in a row without progress.
- **Fix:** deal decks round-robin, or derive the decked set from the games played.

### Low
- **Record generation ignored.** The coverage and text counters ignore it (`collect_coverage.py` 118-128, 481-504), so gen-2
  shards would count. Today `output/effects/records/` is empty and gen-3 lives in its own tree.
- **A variant with no sidecar is kept** (`drop_held_out_variants` 426-431). There is no text to train on, so it is nearly
  harmless.
- **`_keep_unheld` keeps everything when no converted tree is found** (448-456), with a warning only. The default tree exists.
- **Perturbation oddities** (`script_variants.py` 62-90); no wrong labels, only wasted budget:
  - 3,784 `NumAtt$`/`NumDef$ +N` values never match `isdigit`, so most pumps are never perturbed;
  - 892 `-N` values are floored to 0 or 1, turning a debuff into nothing or a buff;
  - 271 scripts have a swappable `DeckHints:Type$` line that changes nothing.
- **Token makers not depleted.** Cards that make held-out tokens are not on the depletion list (`split.py` 131-140), so their
  games are routed away.
- **Deleted held-out variants stay staged** until the next staging. They are never decked.

### Refuted
- **D-B12: unresolvable names keep full weight in coverage decks.** The path exists (`CastabilityMain` 94-110 omits such
  names, and `rank_by_consult` ranks a missing verdict as castable). But every name Forge failed to resolve in the old
  coverage logs (155 distinct) is a non-deck type that `NON_DECK_TYPES` (`coverage.py` 21-35) now removes. The gen-3
  held-out logs show none.

### Code quality (slice 2)
- **Two holdout sources per collector.** `DeckHoldout` (the list) and `HoldoutRule` can disagree, and the list is optional.
  Derive exclusions from the rule and use the list as a check (D3).
- **Card identity.** Coverage keys cards by name; the holdout and build-corpus key them by `script_file`. Use one identity
  (D6).
- **Fixed-name files inside walked trees.** `coverage-decks.txt`, `variant-decks.txt` and `loadable.txt` sit in the records
  and variant trees. Name them per run, or keep them outside the trees (R1).
- **Misleading docstrings and help:** `held_out_round_caps` (R7), and `--variant-volume` and `--seed` (D7, R3).

---

## Fix order before re-collecting (both verifiers)
1. R1: keep the loadable list out of staging, and move the old `loadable.txt` aside before the first variant run.
2. D1 (and D5): count only the record kinds build-corpus keeps.
3. R2: check the JAR, `java` and the `--exclude-cards` file before starting, and limit consecutive failed starts.
4. D2: exclude random-seat games from the held-out round's count.
5. D3: refuse collector runs without a holdout source.
6. R3: give variants names unique per run, or refuse to overwrite.
7. R4, R5, R6, R7 and D4. The rest can wait.
