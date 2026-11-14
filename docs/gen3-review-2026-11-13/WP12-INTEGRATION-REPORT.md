# WP12, integration tests: the implementer's report (2026-11-14)

Scope: make the effects-related modules of `tests/integration/` true to the current system, run
them, add the missing ones, and measure gen-3 recorded games. Every module was run alone, under the
heavy-job lock, JVMs reaped after each run; all write under pytest temporary directories only.

## Commits

- `fcacd7f8` test: random-seat pilot stops its worker by the stop protocol (item 2)
- `b405fc01` test: the game-end marker decodes with mode, aimed_at and a board (item 3)
- `068045de` test: the fork module reads two runs instead of skipping
- `0f230025` test: coverage-isolation decks end their games inside the window
- `352bb716` test: convert writes the current provenance sidecars (item 7)
- `02e847bb` docs: `CollectorSupervisor.play_round` docstring (the overshoot bound was wrong)
- `aa5a54fc` test: two new modules through the real supervisor, plus a conftest helper (items 4, 5)
- `a62ea479` fix(effects): validate-corpus double-counted a rewrite naming its ability both ways
- this file's commit

No Java changed. Nothing in `specs/`, `experiments/`, the fixture, the shim or
`REFUSE_SUPERSEDED_GEN3_PAYLOADS` was touched.

## Modules: what each tests now, result, time

| Module | Result | Time |
|---|---|---|
| `test_effects_collection.py` (unchanged) | 3 passed | 57 s |
| `test_effects_coverage_isolation.py` | was RED (1 failed, 221 s); now 3 passed | 62 s |
| `effects/test_collect_coverage_stop.py` (new) | 6 passed | 37 s (158 s on a cold file cache) |
| `effects/test_random_seat_pilot.py` | 7 passed | 31 s |
| `effects/test_worker_recycle.py` (new) | 7 passed | 135 s |
| `test_effects_patched.py` | 5 passed | 95 s |
| `test_effects_steered_game.py` | 1 passed | 36 s |
| `test_effects_forks.py` | was 8 passed, 2 skipped (228 s); now 10 passed, none skipped | 56 s |
| `test_effects_variant_isolation.py` (unchanged) | 3 passed | 20 s |
| `test_convert_cli.py` | 4 passed | 21 s |

- Coverage isolation: red because a mirror of Llanowar Elves alone ends by decking out; four recorded
  games finished none in 180 s. Decks now pair the elves with Lightning Bolt; two workers.
- Random-seat pilot (item 2): `_BLOCK` / `cut_short` are gone. The worker gets a stop file; the pilot
  touches it, waits for exit 0 and the stop line (`stop file …`, games >= 1), and asserts every game in
  the shard equals `ended_games`, its last record being its marker with the run's mode and a board.
  Every charm is then counted.
- Marker (item 3): `test_effects_patched` collects until a game has ended and checks each marker:
  `patched`, generation 3, `aimed_at` None, a board with players, `records_withheld == 0`, no events,
  the game's last record. The steered game's marker carries `("702.19b", "704.5f")`, the run's mode,
  a board holding the variant.
- Forks: two of its tests ran with no fork budget and always skipped. Now one stage-one worker and one
  with interventions and probes on (until both are taken), each stopped at a game end.
- Convert (item 7): converts Forge's own The Fall of Lord Konda and Blind Hunter into a temp tree:
  every sidecar line has an int `prose_offset`, chapter lines claim triggers `[0]`, `[1]`, `[2]`, the
  two haunt lines claim disjoint triggers, `run.json` sits in the parent of both converted trees.

## New tests (item 4)

`effects/test_collect_coverage_stop.py`: one `collect-coverage` card round through the command line
(wording cards Lightning Bolt and Goblin Piker, empty holdout, 2 workers, budget 2 matches,
`--stop-timeout 300`). Asserts: exit 0; every worker log has the stop line naming its stop file, or no
startup line (killed starting); `run.json` `worker_stops` matches the logs exactly (killed 0, survived
0), `worker_survivors` empty; `cut_games == {before_rounds 0, by_round [0], total 0}` and every gen-3
game in the shards is in `ended_games`; each marker decodes `patched`, `aimed_at == ()`, a board,
nothing after it; layout: shards at the top, `logs/` holds exactly the two slot logs and
`round-0001.progress.txt` (no stop file left), `decks/` the run's decks file, progress lines equal the
ended games (>= 2); `run.json` finished, `stop_timeout 300`, `recycle_seconds 600`, `rounds 1`, `caps`
resolved (`aimed_at`, `directed_fork_rules` empty, `max_records_per_game 5000`), `hook_failures` with
`connector` and `forge`, `fork_refusals` a dict, no failure line of a lost match/game/log
(`conftest.assert_no_lost_output`; the others printed).

`effects/test_worker_recycle.py`: `MatchOutcomeSupervisor` (one worker, best of 1) inside
`run_recorded`, recycle 60 s (below the CLI's 300 s floor, so handed to the supervisor directly),
stop timeout 300 s, the whole `rules/directed-fork-rules.txt` at probability 1 after its real startup
check (`directed_fork_error`, one `DirectedRulesMain` JVM; asserted None). A watcher ends the run
(`request_shutdown`, as a first Ctrl-C) once the pool counted a recycle and the newest shard has an
ended game. Asserts: `recycled >= 1`, `stopped == 1`, nothing killed or crashed; stop-line reasons
`recycling after 60 s` then `stop file …`, each lifetime played >= 1 game; >= 2 lifetimes of slot 0,
every record decodes, generation 3, every game ended (marker count = games), ids unique across
lifetimes, game ids carry their shard's lifetime, markers `patched` with a board and `aimed_at` None;
`run.json` finished with `run_id`, `stop_timeout`, `recycle_seconds`, `worker_stops` equal to the
pool's, no survivors, `hook_failures`, `fork_refusals`, `matches` = outcome rows,
`matches_played` = progress lines; `logs/` holds only the slot log and the progress file, the run
directory only shards, `run.json`, `logs`; every directed-fork record has `fork`, not
`interventional`, `mirror_of` None, one listed rule in `aimed_at`, a `ForkOperation` (it skips, saying
the startup check passed, if no precondition held; in every run here directed records were written,
7.5 a game in one).

## Item 5: Java-written shards through the Python pipeline

No integration test did this; `test_worker_recycle.py` now does it on the run's shards:
`rule-counts --workers 2` exits 0, `generation gen-3`, `cut_games 0`, `records > 0`;
`validate-corpus` prints `[PASS] every shard line decodes` and `gen-3 games end with their game-end
marker: 0 of N` (exit 0 or 1 accepted: a report); `build-corpus --workers 2` must survey exactly the
shards' records and games (`Surveyed <records> record(s) in <games> game(s)`), no traceback, and either
write a manifest with `cut_games 0` or exit 1 with a known too-few-games refusal ("No game is admitted
to the card-disjoint stratum" or "the game-disjoint/card-disjoint validation sample is empty"), no
manifest left. Seen: both refusals, on 11 and 20 games. On 20 games validate-corpus held all 25 judged
invariants.

## Item 6: measurements

One measurement run (scratchpad only): `MatchOutcomeSupervisor`, 4 workers, best of 1, 15 minutes then
a graceful stop, random-seat share 0.5 at probability 0.5, the whole directed-fork list at its defaults
(0.25, 2 a game), every other cap at its default (interventions 2, probes 2 on all eight keywords),
stop and recycle 600 s. 228 games ended (108 Forge-AI, 120 with a random seat), 0 cut on disk, 8 JVM
lifetimes. Game length = first record to game-end marker, from record timestamps (excludes Forge
start, pool generation and deck building).

| | Forge-AI (n=108) | Random seat (n=120) |
|---|---|---|
| length median / p90 / max | 4.6 s / 17.5 s / 89 s | 4.7 s / 18.8 s / 423 s |
| records per game median / mean / max | 174 / 193 / 563 | 163 / 182 / 444 |
| gzipped bytes per game median / mean | 291 KB / 361 KB | 278 KB / 333 KB |
| raw JSON per game, mean | 7.2 MB (about 37 KB a record) | 6.6 MB |

- Games per hour per worker: 220 over the run's wall clock (4 workers x 932 s, Forge starts and two
  stalled JVMs included); about 280 without the two stalls; 319 counting recorded game time only.
- Smaller runs agree: 20 sealed games median 2.3 s, max 6 s (recycle test, directed list at 1);
  burn coverage games 1 to 3 s; the charm random-seat pilot 1 to 9 s.
- A JVM lifetime played 32 and 37 games before recycling (lived 619 s and 787 s).
- Two JVMs died of `OutOfMemoryError: Java heap space` (`-Xmx1200m`) after about 6 to 7 minutes in
  which their game wrote nothing (Forge AI `TimeoutException` in `chooseSpellAbilityToPlayFromList`
  repeated); one was a random-seat match (`random_seat_match_error` and `jvm_error`, 2 `crashed`).
  Their games had under 256 records, so nothing of them reached disk: they are invisible as cut games,
  and only `worker_stops.crashed` and `jvm_error` show the loss (2 of 230 games).
- The defaults: `--stop-timeout 600` covers every finished game seen (longest 423 s; at the stop all
  four workers finished within about 30 s); 300 s would have cut 1 of 228. `--recycle-seconds 600`
  gives 30-40 games a lifetime for a Forge start of 10 to 20 s, about 3% overhead. The recycle +
  timeout backstop (1200 s) never fired: the two hung games died of OOM first. Both defaults look right.
- Bytes: about 350 KB gzipped a game, so 1,000 games is about 350 MB (the 023 quickstart's gen-1 figure,
  "7,000 games under a gigabyte", is about 140 KB a game; no gen-2 figure was measured here).

## Product bugs found

- Fixed (`a62ea479`): `payload_checks.Rewrites._name_the_substituted_ability` summed
  `with_replaced_by + with_buffered`, so a record carrying both counted twice; the real run read
  `330/312 (105.8%)`. Now each record counts once (`_ResultCounts.naming`); unit test
  `test_a_record_naming_its_ability_both_ways_counts_once` (seen red). Python only.
- Docstring fixed (`02e847bb`): `CollectorSupervisor.play_round` said the progress file overshoots
  the budget by at most the worker count; a real round of budget 2 with 2 workers played 5 (burn games
  of 1-3 s end between two 2 s polls). The contract's wording ("one game per worker more than the
  polling lets through") was already right.

## Reported, not fixed (owner's decision or a Forge change)

- `validate-corpus` FAILS "outgoing is null on the results that rewrote nothing" on the 15-minute run:
  34 rewrite records, all Forcefield in one game, carry `replaced_by` AND `buffered` (both the same
  printed line), `buffered_ran: false`. Forcefield's `ReplaceDamage` runs in place (so `replaced_by`
  names it, truthfully) and its sub-ability (exile the effect) is the buffered one; the contract
  (`record-schema-delta.md`: a buffered damage replacement "writes `replaced_by: []`") and the
  validator rule (`_disagrees`, pinned by `test_a_buffered_ability_also_named_as_ran_instead_fails`) do
  not foresee a `ReplaceDamage` with a sub-ability. Also: the exile never ran (`buffered_ran: false`
  on all 34), so the "next time" shield kept applying for the rest of the game (34 records); whether
  that is stock Forge's behaviour (`executeReplaceDamageBufferedSA` sums `PreventedAmount`) or the
  instrumentation's was not traced. Decide the contract and check Forcefield on stock Forge.
- Fork copies that throw in Forge's `GameCopier` (`StaticEffect.removeMapped` -> `Couldn't map <card>`):
  in the 15-minute run 52 intervention, 29 protection, 21 probe and 9 directed fork copies (111),
  where validate-corpus counts 2,295 fork records, at least 798 forks by game, turn and kind, and
  prints "211 damage-step probe forks, 1609 interventional forks"; every fork kind, many cards (Zurgo's Vanguard, Cryptic Cruiser, Serpent of the
  Endless Sea, …). A static effect's source or affected card is not in the copier's card map. Each is
  counted and the fork discarded; no record is wrong. A Forge fix (map leniently, or drop the stale
  entry) would recover about one fork in eight.
- Forced-resolution bodies that throw: `CopyPermanentEffect.getProtoType` -> `CardFactory.readCard`
  NPE (`paperCard` null: a fork's copy has no paper card), and `ChangeZoneEffect.changeKnownOriginResolve`
  NPE (`CardCollection.getFirst()` on a null). One each.
- `fork_fidelity` lines: 42 in the 15-minute run, all "0 difference(s) … no counterpart {ACTIVATION=1}"
  or `COMMAND` (e.g. a Reconfigure Effect), i.e. the `copy_dropped` stamp's cases.
- Fork refusals: 7 `intervention nothing reached the stack`. No `moved the live game`, no
  `game_end_marker_unwritten`, no match error other than the OOM one, no record withheld.

## Not done

- The steered-game and patched tests still end their worker by a kill (they need no whole game but
  the first). The collection and coverage-isolation modules too.
- No measurement run with a Forge-AI-only or random-seat-only collection: the mixed run's games are
  split by `random_seat` instead.
- Nothing left red in the modules above. The coordinator should run the Python unit suite (one unit
  test and `payload_checks.py` changed); no connector re-run is needed (no Java change).
