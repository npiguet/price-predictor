# Gen-3 review fixes: summary (2026-11-13 to 2026-11-15)

The work that came out of the 2026-11-13 review of the gen-3 effect collection stack is finished apart
from the open items at the end. Branch `025-ability-effect-model-gen3`, last code commit `b68188f2`;
`../forge` on `effect-record-hooks` at `27700697699`. Nothing is pushed.

Final suites on that tree: connector 2015 passed, Python unit 6434 passed, none skipped or failed.
The integration suite was last run whole before the directed-fork work; see "Not verified".

## What each package did

| Package | Fixed | Did not reproduce / left |
|---|---|---|
| WP0 | Test hygiene: worker kills patched session-wide, effects tests fail on an unrequested process pool. | The scan covers module attributes of the four packages only. |
| WP1 (Forge) | Game-copy fidelity at a fork point; forks fire leave/dies triggers correctly; shared last-known copies. | `newPTText`, `loseLifeMap`, `pipsToReduce`, `rollbackEffects` did not reproduce. A chosen-card holder copied early keeps its early snapshot. |
| WP2 (Forge + connector) | The three hooks' listener signatures; optional triggers told as run when they go on the stack. | A keyword granted after a can't-have static cannot be brought back by the without-static view. |
| WP3 | Event attribution: end-of-turn and end-of-combat cleanup events, win/loss events once, unattach by the state-based check. | `C1 #6` did not reproduce; `C1 #14` stays (your decision). |
| WP4 | Trigger and rewrite payloads, counters (the two-counter-type split), forced halves. | `C2 #6` did not reproduce. A multi-type `AddCounter` replacement still writes one summed `counter_change`. Arc Trail's damage attribution. |
| WP5 | Forks: no fork advances the live game's timestamps; interventions draw once per turn against a 20-turn game; `replace_card` rewrites pending effects; failure and refusal tallies; five new goldens. | `C3 #5` and part of `C3 #11` did not reproduce. |
| WP6 | Directed rules: preconditions that make a fork's point survive (lands, lethal cards, own keywords, castable timing); list regenerated from counts. | 704.5v (battles) has no choice-free binding. |
| WP7 | Candidates and decision records; per-game record bound. | Rout-style "flash for an extra cost" gives one candidate with `timing: false` (known limit). The split-second finding never existed. |
| WP8 | Collection orchestration: `--count-from`, `--continues`, `run.json` invocations, logs under `logs/`, decks under `decks/`, job-object binding of worker JVMs, zero-record exit 1, 1 to 12 workers, the graceful stop protocol and the game-end marker. | `P1 #13`, `P1 #14` did not reproduce. |
| WP9 | Python reading of the changed wire; targets as cast. | Hidden-keyword, name and mana-cost changes are counted as unsupervised, not given columns. |
| WP10 | Corpus build: cut games dropped by the marker, the corpus digest independent of location, outcome signatures. | `P3a #16` (survey memory) needs measuring on the real corpus. |
| WP11 | Rule detectors: legality and timing rewritten per rule; on-data checks. | 607.2a / 607.5 through `exiled_with`. 443 scenarios have no detector (known, no test asserts it). |
| WP12 | Tests: about 121 Python tests deleted after reading each sibling, about 50 strengthened, about 25 written; Java tests and goldens; ten integration modules passing; two Java bugs found by tests and fixed (saga chapter keys in sidecars, the probe-keyword default). | The ledgers' "not done" and "skipped for budget" lists. |
| DF-A, DF-B | Directed forks keep their bound records; wrong-way variants inverted; a variant may force one play; every fork refuses a game restart; 22 rules taken off the list, each with its reason. | — |
| GAP-1 to GAP-4 (after two verifier passes) | Nine rules that forked the side their counts already held now fork the missing side; 118.11 off the list (your decision); the forced-play guards tested; a forced "yes, pay" really pays; four removed rules put back with forced plays (205.1a, 111.10p, 701.23f, 702.133a); 122.1d, 702.19e, 701.42b off; 118.12b rebound; 510.2 and 702.1c detectors corrected. | 702.70b (poisonous) stays off: no Forge card prints the keyword. |
| INT-FIX, GAP-5 (verified) | Fork copies failing on sideboard cards (Forge); a memory leak that held every recorded game and fork; null pointers in forced resolutions (forced ninjutsu and sneak are refused; a copied token keeps its paper card); integration tests stop workers by the protocol; the close-recording error masking. | `play_round` overshoots its match budget by in-flight games (unchanged; options listed below). |
| LAYOUT, LAYOUT-2 | Every file under `output/` and `models/` can be steered into a run directory: `--outcomes-dir`, `--win-rates-output`, `--embeddings-path`, `--checkpoint-dir`, `--logs-dir`, `--output` and `--reports-dir` on evaluate and the probes, `--report`, `--model-path`. Forcefield's record shape accepted (your option A). No unit test skips on an empty `output/`. | The flag defaults still name the old flat paths. |
| FIXTURE | New gen-3 test fixture (314 records, 57 games) cut from the 2026-11-15 pilot; old-format payloads always refused; the shim removed; test builders default to gen 3; a training bug fixed (an auxiliary head's targets depended on shard reading order). | 25 detectors with a broad "does not fire" side are allow-listed in a test, not narrowed. |
| STALL, BLINDKILL | Diagnosis: workers were not hung but in games that crawl (Forge's AI takes seconds per decision on boards of 40 to 66 permanents). Your decision: in effect-record collections the pool also kills its longest-running worker every 60 s, as it already did without effect records. | The killed game's records are dropped. |
| WP13 | Documents made true as a whole: effects, connector, sealed, draft, price-predictor and root docs, the 025 spec and contracts, the root spec, the quickstart, the experiment record's known limits. | Four large spec files were checked by search, not read whole; six code renames remain. |

## Wire changes (record format, generation 3, changed in place)

Contract: `specs/025-ability-effect-model-gen3/contracts/record-schema-delta.md`.

- Every finished game ends with a `turn_action` / `game_end` marker; a game without one was cut and is
  left out of the corpus and every count.
- Fork records carry `copy_dropped` (what the game copy left out, by kind); `mirror_of` is always null
  on a directed fork.
- A forced half may hold no events with `resolved` or `fizzled`, or be `declined`.
- `dice_rolled` results are `{final, natural}`; outcome parameters are written in key order.
- A rewrite record may name one line in both `replaced_by` and `buffered` (Forcefield's shape).
- Sidecars: saga and haunt chapter lines carry their own trigger keys; every line has `prose_offset`.
- Shards written before the marker existed, and gen-3 records in the old payload shape, are refused.

## CLI changes

Contract: `specs/025-ability-effect-model-gen3/contracts/cli.md`.

- Collection: `--stop-timeout`, `--recycle-seconds`, `--count-from`, `--continues`, `--outcomes-dir`;
  workers 1 to 12; exit 1 when a run played matches and wrote no record.
- Stopping: one Ctrl-C asks every worker to finish its game; a second kills. In effect-record
  collections the pool also kills its longest-running worker every 60 s.
- Paths: the flags listed under LAYOUT above. `scripts/make_fixture_records.py` requires its path flags.
- `run.json` in every run directory records commands, commits, inputs, worker stops, cut games,
  fork refusals and worker-log failures.

## The catalogue and the directed-fork list

- `rules/directed-fork-rules.txt` holds 49 rules; the startup check passes over all of them.
  43 rules are kept off the list, each with its own reason in `NOT_DIRECTED` of
  `scripts/rules_catalogue.py`; none is deferred without a reason.
- Every listed rule's fork was played and tagged by the Python detectors and yields the side its
  counts lack, against the checked-in counts.
- 111.10p, 701.44d and 701.50f name cards printed only in Commander sets: their forks bind only in a
  steered `collect-variants` round.
- 443 scenarios have no detector (`rules_catalogue.py validate` lists them; known, not asserted).

## What the 2026-11-15 pilots measured

- Game length over 1,946 games: median 4 s, 99th percentile 44 s, 99.5th 88 s; about 1 game in 400
  runs past 5 minutes (a crawl).
- With the once-a-minute kill, 6 workers held 16 to 31 matches per minute for 10 minutes; without it
  three of six workers were lost to crawling games within 9 minutes.
- `validate-corpus` passes all 25 judged checks on both runs; 1 fork copy failure in 15 minutes.
- Directed forks in 524 games: both sides recorded for 8 rules, one side for 5, never bound for 36.
  118.12b and 205.1a took most of the fork records.

## Before the full collection

1. Stage 0 of the quickstart is done (`output/effects/gen3/converted`, `inputs/2026-11-15-*`).
2. The fork fidelity audit (quickstart Stage 5, first block): a 20-minute scratch run.
3. The collection runs of Stage 5, then Stage 6. After `build-corpus`, regenerate the model's output
   lists from the corpus manifest and update the pinned widths (the quickstart has the step): the
   present lists come from pilot data.
4. The connector is built and installed from the current tree; rebuild it if anything changes.

## Open

- DECIDED 2026-11-15: the flag defaults stay as they are for now (two of `evaluate-effect-model`'s
  break the "one module, one tree" rule: `--win-rates` reads the sealed tree, `--output` is inside
  the model's run directory; pass both explicitly).
- DECIDED 2026-11-15: no gen-2 recount. The directed-fork list stays as generated from the
  2026-11-06 counts, which predate the detectors of 23 of the 49 listed rules; the collection's own
  `rule-counts` is what shows which listed rules were misjudged. The integration test that compared
  the committed counts with a gen-2 run is removed.
- 118.12b (Elfhame Sanctuary's optional search) forks in 99% of games (517 of 520 pilot games with a
  directed fork) and 510.2 in 29%, each taking one of a game's two directed forks; every other rule
  together got 12 forks. Nothing is crowded out in most games, but one fork a game is spent on a rule
  that needs about a hundred records. Regenerating the list from the collection's own counts after a
  first few thousand games (`rule-counts`, then `directed-rules --counts`) drops such rules.
- `play_round` overshooting its match budget: faster polling (cheap, about one extra game per
  worker) or a hard bound (changes the stop protocol).
- A later review pass over the model, training and evaluation code is still owed (deferred on
  2026-11-12).

## Not verified

- The integration suite as a whole on the final tree. One test moved there reads data that is not on
  this machine (`test_gen1_corpus_digest.py`, which reads the NAS);
  two were adjusted for the periodic kill and two for `--embeddings-path` without being run.
- Packages without an independent verifier: GAP-5, LAYOUT, LAYOUT-2, FIXTURE, BLINDKILL, WP13, and
  the second rounds of WP5 and WP6.
- The leak fix under a long run: the 10- and 15-minute runs show a flat heap and no out-of-memory
  death, nothing longer was run.
- How often each directed rule's precondition binds over a full collection.
