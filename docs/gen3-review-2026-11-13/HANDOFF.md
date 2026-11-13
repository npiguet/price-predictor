# Gen-3 review fixes: handoff to the coordinating session

You are coordinating the implementation of every fix that came out of the 2026-11-13 review of the gen-3 effect collection stack. You delegate each work package to an Opus implementer and have an Opus verifier check it. You do not implement.

- **Findings**: `findings.md` beside this file. Ids are `<slice> #<n>` (`C2 #1`, `V5 #14`). Verifier sections (`V1`-`V5`) override reviewer sections where they disagree: read the `V` entry for a finding before briefing an implementer on it.
- **Standing brief for implementers**: the memory file `reference_025_agent_brief.md` (working rules, tool gotchas, Forge engine facts). Point every implementer and verifier at it. Where it says "task ids from tasks.md", read "the work package from this handoff".
- **JVM reaper**: `reap-jvms.ps1` beside this file (a copy; the brief's path is in an old temp directory).
- Branch `025-ability-effect-model-gen3`; `../forge` on `effect-record-hooks` (new commits on top, never amend or rebase). Nothing is pushed unless the user asks.

## Do this first

**WP0 must land before anyone runs pytest over `tests/unit/sealed/`.** Six tests in `tests/unit/sealed/application/test_match_outcomes.py` run the real `taskkill /F /T /PID` against PIDs 100 and 1000-1002 (`V4 #10`). Until WP0 is merged, no agent runs that file, the `tests/unit/sealed` directory, or the full suite.

## Limits the user set

- At most 3 Opus agents at once. Mechanical lookups go to Sonnet or Haiku. Resume an interrupted agent by message; do not relaunch it.
- One heavy job at a time, machine-wide (any `mvn`, any JVM, the full pytest suite, GPU). Implementers run targeted tests only; you run the full suites between packages. At most 12 CPU-busy processes, with explicit worker counts.
- Long runs (collection, pilots, fixture regeneration that needs Forge workers) are launched by the user: give them the PowerShell command.
- Commit freely on the branch; never commit `output/` or `models/`; never push unprompted.
- Any edit under `specs/` or `experiments/` loads the `feature-workflow` skill first. Fixed defects, their reviews and redone runs stay out of experiment records.
- Ask decisions in plain text, never with the question pop-up.
- A change to anything persistent (wire schema, CLI flags and defaults, file layout, documented behaviour), as opposed to a plain bug fix, updates the spec files that describe it (`specs/025-ability-effect-model-gen3/` spec, plan, data model, contracts, quickstart) in the same package as the code. The verifier checks code and spec agree. WP13 is the final sweep, not the first time specs are touched.

## Decisions to get from the user before the affected package starts

Each has a recommendation. Ask them together at the start; packages that do not depend on an answer can begin meanwhile.

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| D1 | **Record schema changes.** The gen-3 schema is frozen "before the generation's first record is collected", but every gen-3 record collected so far was discarded and no shard exists. Fixes that change the wire: per-chain targets and `targets_at_cast` on effect halves (`X1 #2`, `X1 #14`, `V5 #25`); signed or per-mode amounts in run-parameter events (`V2 #1`); `face_change.to_state` spelling (`V2 #2`); hidden keywords in snapshots and events (`V2 #14`); a game-end marker (`P2 #5`); damage type on `damage_dealt` (`V1 #13`); attach targets (`V2 #8`); a lossy-copy stamp on fork records (`C3 #6`); bus-side `face_change` / `destroyed` / `sacrificed` / `player_lost` (`V2 #3`). | WP3, WP4, WP5, WP7, WP9, WP10, WP11 | Change the gen-3 schema in place and update `contracts/record-schema-delta.md` and `data-model.md`, since re-collection from scratch is already planned. |
| D2 | **Cut games.** Graceful worker stop (the worker finishes its game on a stop request and self-recycles at a game end), a game-end marker that lets the corpus build drop cut games, or both (`P1 #1`, `V3 #5`, `V5 #13`). | WP7, WP8, WP10 | Both: the marker is cheap and makes the build robust; the graceful stop removes most of the loss. Keep `taskkill` as a backstop after a timeout. |
| D3 | **Documented-design items the review still flagged.** (a) Interventions all spent at the game's first boundary on an empty board (`V3 #2`); (b) fork fidelity and the copier tally gate nothing, and fork records carry no lossy-copy mark (`V3 #8`); (c) `mirror_of` on directed forks is dead code (`V3 #13`); (d) `rule-counts` omits zero-tag rules (`V5 #22`). | WP5, WP11 | (a) add a per-boundary probability gate like the other schedulers and stop marking a no-target line as seen; (b) stamp the record, keep reporting only; (c) delete the dead pairing; (d) write a "declared, zero records" list. |
| D4 | **Legality and timing detector declarations.** 14 blocking rules, 11 targeting rules and 26 timing rules each share one unfiltered declaration (`V5 #16`). Fixing it means new anchor-side parameters in `legality_membership` / `verdict_bit` and rewriting those scenario declarations in `rules/scenarios/*.json`. Who edits the rules catalogue, and is a per-rule rewrite wanted or should rules the records cannot distinguish be dropped from the detector set? | WP11 | Add anchor-side parameters decided by `rule_outcomes.may_block` / a `may_target` twin, rewrite the declarations, and add a catalogue test that fails when two rules share an identical declaration. |
| D5 | **Test deletions.** About 240 candidates; the review coordinator read only 22. Delete after each is re-checked by the implementer, or only the ones with a named stronger sibling? `test_corpus_store.py::test_the_store_lays_out_the_dataset` is to be kept if the on-disk layout is a frozen contract. | WP12 | Implementer re-checks every candidate against its named sibling before deleting; anything doubtful moves to "strengthen". Keep the corpus-store layout test. |
| D6 | **Artifact layout.** Worker logs beside the shards versus under `logs/` (`V4 #20`); `match-outcomes --effect-records` appending to the shared `output/sealed/match-outcomes.txt`; a read-only `--count-from DIR` for `collect-coverage` so a run in its own directory sees earlier coverage (`V4 #5`); `collect-variants --continues RUN_ID` (`V4 #6`). | WP8 | Follow the artifact-layout skill: logs under `logs/`, add `--count-from` and `--continues`. Leave the sealed outcomes path to a separate decision in `src/sealed`. |
| D7 | **Unverified findings.** Low-severity reviewer findings that no verifier examined (most `#` entries in the F/C/P sections not cited in a `V` section). | all | The implementer first reproduces or traces each one test-first; a finding that does not reproduce is reported back with the evidence and dropped. |
| D8 | **Stale gen-3 test fixture.** `tests/fixtures/effects/gen3-records.jsonl.gz` is in a superseded payload shape (`V4 #19`). Regenerating it needs a small collection run after the Java fixes. | WP12 | The user launches a short pilot once WP1-WP7 are in; `scripts/make_fixture_records.py` cuts the new fixture from it. Until then, tests that need the current shape use the Java characterization records. |

### Answers (user, 2026-11-13)

- D1, D2, D3, D6, D7: as recommended.
- D4: per-rule rewrite. Add anchor-side parameters (decided by `rule_outcomes.may_block` and a `may_target` twin; timing narrowed by card type or keyword) and rewrite the declarations wherever the record carries the data; a scenario that states the same observable fact as another is declared an explicit duplicate of it (counted once, exempt from the uniqueness test); only what the record cannot show drops to `none` with its reason; add the catalogue test that fails when two rules share an identical declaration. The WP11 implementer edits the catalogue (it was written by Opus/Sonnet agents) and reports any declaration it could not express.
- D5: as recommended. The corpus-store on-disk layout is frozen for all of gen 3 (and probably later generations) unless a collection or coverage gap forces a change: keep `test_the_store_lays_out_the_dataset`.
- D8: as recommended, except that the coordinator (or a subagent) launches the short fixture pilot and cuts the fixture itself; this permission covers that pilot only, not collection runs.

## Work packages

Each package lists the findings it owns. "Tests" means: write the failing test first on the real path the test reviewers named, then fix. A test that pins the bug as intended behaviour (listed in the TC/TP sections under "PINNED" / "pins") is changed in the same package as its fix. File ownership is exclusive while a package is open: two packages that touch the same file do not run at the same time.

| WP | Scope | Findings | Depends on |
|---|---|---|---|
| **WP0** | Test safety. Patch the kill in `tests/unit/sealed/application/test_match_outcomes.py` (autouse fixture in `tests/unit/conftest.py`), give `FakeProcess.wait` the real signature; tag `VariantStagingTest.aTreeWithItsLoadableListInitializesForgeTwiceRunning` `integration`; pass `workers=1` in the counter tests that start per-core pools. | `V4 #10`, TP1 strengthen 1 and 16, TC4 strengthen 19 | none |
| **WP1** | Forge: game copy. `Card.copyFrom` copies the hidden-keyword lists; `GameCopier.makeCopy` calls `copyLastState()`, restores timestamps for every card, copies `saddledByThisTurn`, carries `PlayerStatistics` turns and the other omitted state; `AbilityMapper.find` verifies identity; `copyStateFrom` SVars per state. | `V1 #1-6`, `V1 #16`, `F1 #5-10`, `F3 #1`, `F3 #4`, `F3 #6` | WP0 |
| **WP2** | Forge: hooks and accessors. Buffered replacement ability reported separately from "ran"; trigger firing noted where the trigger runs; `cantBlockStatic` accessor; `cantHaveKeywords` keyed by static; missing or wrong outcome notes; per-slot failure counter the connector can read; `deepCopyParams` inside the containment; state-based checks inside the loop hook; cast-time targets available at the resolution hook. | `V1 #7-15`, `V1 #17` (ReplacementObserver), `F2 #1-9`, `F3 #2`, `F3 #3`, `F3 #5` | WP1 (both touch `Card.java`) |
| **WP3** | Connector: event capture. End-turn and end-combat spells keep their effect half; bestow/mutate fizzle; payment events counted; attach targets; permanent spell's own events `root`; rebaselines; duration buckets per the contract; zone names normalised; `choice_made` by API; `EventBuffer` dedup; `declined` reset; `conditions_met` line test; subscriber and short-arity failures counted; bus-side `face_change` / `destroyed` / `sacrificed` / `player_lost`; damage type. | `V2 #3`, `V2 #5-12`, `V2 #20`, `C1 #1-15`, `X1 #7`, `X1 #8` | WP2, D1 |
| **WP4** | Connector: snapshots, records, provenance. `RunParamNormalizer` as a per-mode table checked against Forge's mode names; `refs.targets` over the resolving chain and `targets_at_cast`; hidden keywords; `LayerContributions`; suppression without the static's can't-have rows; `StatDiffer` (face, abilities, deterministic order); Clone keyword keys and `AddTriggers$`; `ChainRenderer` keys; `copy_source`; phased-out permanents; `RecordReservoir`; `LibraryCapture` cache; hand entity for the acting face. | `V2 #1`, `V2 #2`, `V2 #4`, `V2 #13-19`, `C2 #1-16`, `X1 #2`, `X1 #3`, `X1 #6`, `X1 #10`, `X1 #14`, `X1 #15` | WP2, D1 |
| **WP5** | Connector: forks and probes. Hook-derived events on forced resolutions and probes; no-target burn and line key in `InterventionScheduler`; mapped-ability check; lossy-copy stamp; `ForkRun.take` counts discards; `AddPermanentOperation` and `ReplaceCardOperation` state; `ProbeScheduler`; no fork while the stack resolves; `StackOverflowError` guard. | `V3 #1`, `V3 #2`, `V3 #8`, `V3 #9`, `V3 #11-13`, `C3 #3-11`, `X1 #5` | WP1, WP3 (shared router), D1, D3 |
| **WP6** | Connector: directed rules. Land-in-hand check for the seven one-land-short rules; phase gate for `GameLossReplaced`; uniqueness for the six removal rules; active-player holder for the three sorcery rules; damage and counter checks for the five replace rules; counter-edit rules; one graveyard-replacement predicate. | `V3 #10`, `C4 #1-8` | WP5 |
| **WP7** | Connector: random seat, channels, worker. Divided allocation and cross-target constraints in `RandomChoices`; payment pre-check; trigger and mana reservoirs (flush, key, close order); game-end marker and graceful stop on the Java side; `ContinuousChannel` key; `ManaChannel`; blockers' responsible static and lure label; `RewriteChannel` and tally; failure report including Forge's counter, printed in a `finally`. | `V3 #3-7`, `V3 #14`, `V3 #15`, `C5 #1-16` | WP2, D1, D2 |
| **WP8** | Python: supervision and collectors. `try/finally` and a Job Object or parent check in `forge_jvm`; graceful stop request; kill timeout; worker-log scan surfaced in the round report and `run.json`; zero-record exit for `collect-coverage`; manifest per-command commits, run id and resolved caps; `--count-from`; `--continues`; CLI bounds; one-shot timeouts; `script_variants._apply` boundary; artifact layout. | `V4 #1-9`, `V4 #20`, `P1 #1-14` | WP7 (stop protocol), D2, D6 |
| **WP9** | Python: decoding. `shard_io` reports an early stream end and locates decode errors; unknown `generation` refused; gen-3 records in the old payload shape refused; closed-list checks at decode; `zone_change` destinations; `normalize_keyword` qualities; `line_or_none` tally; sign and face handling in `effect_targets` to match WP4; reading of the new wire members. | `V4 #11-18`, `P2 #1-12` | WP3, WP4 (wire), D1 |
| **WP10** | Python: corpus build and derived records. Unknown game not defaulting to training; cut-game detection; `grid_edits` five-colour pair and damage grid; `x_paid`; `_static_condition`; validation samples; `damage_step_keywords` step; protection copies (outcome, acting face, real targets); `_text_room`; lethality fork exclusion; empty rules catalogue refused before clearing; selection-set overwrite; manifest fields; `aimed_at`. | `V5 #1-13`, `V5 #23`, `P3a #1-18` | WP9, D1, D2 |
| **WP11** | Python: detectors, validation, scenarios. Per-kind emittable-types table; forced and degraded records; legality and timing declarations; `pt` value; combat step; parameter validation in `DetectorIndex.of`; zero-tag list; `sba_zone_move`; link pairing; presence per kind; fired-share floor; undecodable record reported; `--variant-scripts`; event params in `field_coverage`. Scenario edits in `rules/scenarios/*.json`. | `V5 #14-22`, `V5 #24`, `V5 #25`, `P3b #1-15` | WP9, WP10 (shared `rule_outcomes`), D3, D4 |
| **WP12** | Tests. Regenerated gen-3 fixture; gen-3 builder defaults; whole-catalogue detector test; unconditional patched-checkout assertion; state-based check in `RegisteredRulesTest`'s `twin`; a real match through `GamePlayer.playMatch` with records on; Java-written shard read by Python; the "strengthen" and "missing" lists not already covered by a fix package; the deletion candidates. | TC1-TC4, TP1-TP4, Forge test notes in F1-F3 | all fix packages, D5, D8 |
| **WP13** | Documents. `src/effects/CLAUDE.md`, `forge-connector/CLAUDE.md`, the 025 contracts, data model and quickstart brought in line with what changed; readability items the user wants done. | readability lists in every section | all |

Order: WP0; then WP1 and WP8-independent parts in parallel (WP8's non-protocol items do not depend on Java); WP2; then WP3, WP4 and WP7 in an order that respects file ownership (`BusBracketCollector`, `ClauseOutcomeRouter`, `GameRecording` are shared: assign each to one package at a time); WP5; WP6; WP9; WP10; WP11; WP12; WP13. Python packages WP9-WP11 can start on their Java-independent findings early.

## How each package runs

1. Brief one Opus implementer: the package row, the finding ids to read in `findings.md` (verifier entries first), the files it owns, the standing brief, test-first, targeted tests only, commit its own paths when green.
2. When it reports, run the heavy suites yourself, one at a time: Forge install if `../forge` changed, `mvn install -DskipTests` then the connector tests, the Python unit suite, then reap leftover JVMs.
3. Brief one Opus verifier with the package's finding list and the diff, without the implementer's reasoning: for each finding, is it fixed on the real path, does a test fail without the fix, and did the change break a neighbour. It is read-only apart from targeted tests.
4. Send confirmed gaps back to the same implementer (resume it). Record the package's state in the table below and commit.

## Before re-collection

After WP1-WP11: the fork fidelity audit (tasks.md T181, quickstart stage 5), a short pilot the user launches, `validate-corpus` and `rule-counts` on it, the fixture regeneration (D8), and T124 before stage 6. Ask the user before any collection run.

## Status

| WP | State | Commit(s) | Notes |
|---|---|---|---|
| WP0 | done; verified twice (all gaps closed); connector default suite 1657 passed with `VariantStagingTest` out of it. A full unit run on `f26fefea` still to run (comes with WP1) | `1cffd83a`, `2e85c898`, `f26fefea` | Unit suite 5561 passed on `1cffd83a`. Kill patched session-long at every binding found by scan; effects tests fail on an unrequested process pool (`workers=None` or wider than 4). Left: scan covers module attributes of the four packages only |
| WP1 | done; verified twice, third commit closes the last three verifier gaps (not re-verified; red run by the implementer). Connector suite on the third commit running | forge `34fa64c2ce5`, `29c85cb1f25`, `4152a7d59c5` | `newPTText`, `loseLifeMap`, `pipsToReduce`, `rollbackEffects` did not reproduce at a fork point. `makeCopy` 7.2 ms on a 24-permanent board. For WP5: `AbilityMapper.map` may return null; forks fire dies/leaves triggers for creatures killed before the first resolution; `BoardEdit.settle` should call `copyLastState()` after an edit; last-known copies are one shared object per card per fork (never mutate one); a fork's extra-turn ability with no counterpart is dropped and counted. Left: a chosen-card holder copied early keeps its early last-known snapshot; hooks firing against the copy's game during `makeCopy` has no test (WP12) |
| WP2 | implementer running | | |
| WP3 | not started | | |
| WP4 | not started | | |
| WP5 | not started | | |
| WP6 | not started | | |
| WP7 | not started | | |
| WP8 | Java-independent part implemented (unit suite 5690 passed; verifier found eleven gaps, implementer closing them); stop-protocol part (`V4 #1`, `P1 #1`) waits for WP7 | `de0acbab`, `fc191cfb`, `df1a29b7`, `5bfd62ca` | `P1 #13`, `P1 #14` did not reproduce. New: `--count-from`, `--continues`, `run.json` `invocations`, logs under `logs/`, decks under `decks/`, per-round progress files, job-object binding of worker JVMs, zero-record exit 1. Not done: counting crashed random-seat matches |
| WP9 | not started | | |
| WP10 | not started | | |
| WP11 | not started | | |
| WP12 | not started | | |
| WP13 | not started | | |
