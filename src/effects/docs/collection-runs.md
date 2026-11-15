# Collection runs

Rules to respect when running, babysitting or changing `collect-coverage`, `collect-variants`, `validate-corpus`, `field-coverage` or a `sealed match-outcomes --effect-records` run, and when changing how workers are recycled, stopped or killed. This file does not describe the commands:

- Flags and defaults: each command's `--help`; the shared collection flags are defined in `sealed/infrastructure/effect_collection_flags.py`.
- The variant rounds: `specs/2026-10-30-ability-effect-model-gen3.md` § 8.15, `domain/variant_rounds.py`, `application/steered_variants.py`.
- The stop protocol: `specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol, and Java `WorkerStop`. The pool: `ForgeWorkerPool` in `price_predictor/infrastructure/forge_jvm.py`.
- What a user launching a collection must know: `specs/025-ability-effect-model-gen3/quickstart.md` Stage 5.

## Collection is an opt-in

- Collection rides a command that already exists: `python -m sealed match-outcomes --effect-records DIR`. `--effect-records` has no default there, because a default would turn every self-play run into a collection run. Without the flag no shard is opened, and `match-outcomes.txt` and `cards-played.txt` keep their exact format and content.
- With the flag, `match-outcomes.txt` and `cards-played.txt` go to `--outcomes-dir`, by default the `--effect-records` directory, beside the shards: the collection writes nothing outside its run directory. Every shard reader finds shards by suffix, so neither file is read as one.
- The instrumentation costs no extra simulation: it rides matches that were going to be played anyway.
- `MatchWorkerMain` has a records-only mode: with `-Deffect.records.dir` set and `-Doutput.file` absent, neither sealed writer is constructed. `collect-coverage` and `collect-variants` run their workers that way. Their decks are built for coverage and not for a fair self-play sample, so their matches in the sealed corpus would corrupt the scorer's training data.

## Short-lived workers

- Every collecting supervisor runs its Forge workers under `-Xmx1200m` and recycles each one: a worker stops itself at the first game end after `--recycle-seconds` of its JVM's life, and `ForgeWorkerPool` starts a fresh one at once.
- The heap and the recycle are one measure, and neither is a tuning knob. Forge misbehaves late in a JVM's life, with loops that do not terminate and allocation that grows without bound. The small heap turns a runaway worker into an early, cheap death the pool restarts, where a large one lets one worker drag the machine down hours into a run. The recycle, with the kill that backs it, bounds the pathologies that do not allocate.
- Do not raise the heap, lengthen the recycle or propose removing either without saying what contains Forge instead: either change hides the pathology. The price, a handful of GC crashes per overnight run, each costing the game in flight and nothing else, is paid deliberately.
- A worker whose game has not ended `--stop-timeout` past its recycle time is killed.
- The recycle is asked only at a game end, so it never ends a game that crawls. On a large board each Forge AI priority decision can take seconds (the combat prediction under its damage and pump checks, with a mana-payment simulation inside each step), and a game then runs ten minutes or more while its worker log and shard stay silent. Only the kill above ends it, and its records are cut. Such a worker is not hung: a thread dump shows `main` waiting on a "Game AI Eval" thread a second or two old, a new one at each decision, with the heap flat.

## Stopping a worker

- Each worker slot has a stop file, `logs/<run_id>.<slot>.stop`, deleted once the pool's run ends. A round's end, the run's end and a first Ctrl-C create every running worker's file and wait for each to finish its game, write its game-end marker, close its shard and exit 0 with its stop line.
- The workers run in process groups of their own (`forge_jvm.graceful_popen_options`), so a console Ctrl-C reaches the supervisor alone and never a worker JVM.
- A worker still starting Forge, with no startup line in its log, is killed at once: it has no game to cut (`killed_starting`).
- No worker is restarted once the round's budget is reached, so a round ends at most one game's length, bounded by `--stop-timeout`, after its budget.
- The kill (`taskkill /F /T`, `ForgeWorkerPool._kill`) is the backstop only. It ends a worker still playing `--stop-timeout` after the request, one overdue past its recycle time by `--stop-timeout`, every worker on a second Ctrl-C, and a survivor of a crash.
- A round is counted only once every worker of its pool has ended.
- Each round reports, and `run.json` records, how the workers ended (`worker_stops`) and how many games the counts left out as cut (`cut_games`).

## Restarts are the normal condition

- An eight-hour run is hundreds of JVM lifetimes across its worker slots, and a crash or a kill ends a JVM mid-game without a stop. Record identity, game boundaries and any resume logic must be correct across hundreds of restarts and across a JVM ended mid-game. This is a correctness constraint, not a throughput one.
- Anything a worker counts from zero is namespaced by the JVM lifetime, not by the worker slot. Counted per slot, `record_id` values repeat and one `game_id`, the corpus's join key, merges unrelated games. The id and shard-name layout: `record-format.md` § Shards.
- A file boundary is not a game boundary. A kill or a crash cuts a game and its remainder is never written. A game that ended writes a `turn_action` / `game_end` marker as its last record, and a game whose shard holds no marker is a cut one: `build-corpus` drops it whole, and the collectors' counters and `rule-counts` count nothing from it (`domain/game_ends.py`).
- A worker that stops at a game end closes its shard whole, and the shard writer writes its block in flight at every game end, right after the marker (`RecordShardWriter.endGame`). A killed or crashed worker therefore loses the game in progress only: that game's records in complete 256-record blocks stay in the shard, its last 255 at most are absent, and so are its end-of-game records (fired triggers, mana reservoirs, the outcome, the marker), so it reads as cut.
- A resume must not assume a clean shutdown left a marker.
- Run `python -m effects validate-corpus --effect-records <dir>` over the first few minutes of every pass. A defect of this kind shows in the first minute of shards and raises nothing for the rest of the run.

## Both collectors

- A holdout source is required at startup: `--exclude-cards`, `--split-from` or `--training-corpus` (exit 1 without one). The held-out round, `--only-cards`, needs none. Passing `--exclude-cards` and `--split-from` together is refused: two spellings of one holdout is a disagreement nothing would report.
- Point `--training-corpus` at the depleted corpus. Discovery recurses, so a tree also holding the full-strength corpus would count validation records toward coverage, and a card whose only records are held out would read as satisfied. A directory without a `holdout-cards.txt` is refused rather than run undepleted.
- A collector's run directory is its `--effect-records`: shards at the top level, every worker log and progress file under `logs/`, decks files under `decks/`.
- Each run's decks go to a file of its own (`decks/coverage-decks.<run_id>.txt`, `decks/variant-decks.<run_id>.txt`), because every worker reloads the file at its start.
- A round ends once its progress file counts `--decks-per-round` matches and every worker has finished the game it was playing. How workers are stopped and recycled: § Stopping a worker.
- There is one progress file per round, so a worker the backstop kill could not end appends to its own round's file and never to the next round's count.
- A gen-3 game whose shard holds no game-end marker credits no card, held-out text or variant, as `build-corpus` drops it.
- A run that plays rounds and writes no record in its own shards exits 1. A run that plays none exits 0 when every unit was already done, and 1 when units remain that no deck could hold.
- After every round the supervisor scans the worker logs for failure lines, prints them by kind with the hook-failure counts and fork refusals, and writes the run's totals into `run.json`. A fork refusal counts toward no failure threshold or exit code.
- `--count-from DIR` counts an earlier run's records directory toward satisfaction. A directory overlapping `--effect-records` or another `--count-from` is refused, since shard discovery recurses and would count a shard twice.
- A missing `--split-from`, `--wording-cards` or other list file exits 1 at startup.
- The castability consult and the sidecar pass are one-shot JVMs with a scaled timeout (`one_shot_workers.scaled_timeout`). On a timeout the consult ranks every card castable; the sidecar pass exits 1.
- Every record carries `aimed_at`, `[]` where the run names no rule.
- Both take the random-seat flags (`collection-internals.md` § Random seat).

## `collect-coverage`

- It exists because sealed self-play only plays what sealed pools contain, so a card in no sealed-legal set never appears in a record.
- A card the consult leaves unanswered while answering others is one Forge's card database does not hold, so no deck can name it. It leaves the unit and the decks. No share of such cards stops the run; a large share means `../forge` is older than the converted cards.
- The castability verdict only ranks, never drops. Being in a game is the precondition an interventional fork starts from, and a card the consult calls uncastable is exactly the one an intervention has to force into play.
- Satisfaction counts acting hosts, event subjects and referenced refs, not snapshot presence: a card sitting on the battlefield while something else resolves teaches nothing about it. It counts only the gen-3 records `build-corpus` keeps. A `continuous` record credits its static's host. Every entity is counted as the converted card its printed script file is, so "Fire // Ice" in hand, "Ice" on the stack or a back face all credit `fire`.
- Retirement is the only thing that makes the run terminate: a card with no new qualifying record for `--no-progress-rounds` rounds whose games showed it retires. A stalled round is charged only to the cards its games showed, because a worker draws its decks with replacement and some written decks are never played; a card never shown retires after `UNSEEN_RETIRE_FACTOR` times as many rounds.
- The per-round progress file under `--effect-records`'s `logs/` is what lets a round end. It is not a sealed corpus and nothing downstream reads it.
- The coverage unit and the deck candidates come from `output/cardsfolder/` alone; a token script is not a deckable card.
- Cards of a non-deck type (scheme, vanguard, plane, phenomenon, conspiracy, contraption, attraction, dungeon) are excluded. Forge's card database does not load them as deck cards, so a deck holding one never materializes, and with no records they would weigh the most and land in most decks.
- `--directed-fork-rules` is a label only here: no directed fork runs, and the rules go to every record's `aimed_at`.
- The two residues it reports (cards judged uncastable, castable cards short of `--target-records`) both fall to interventional resolutions.

## `collect-variants`

- It reads Forge's source scripts, not the converted ones, perturbs one parameter and plays the result. A variant exists on the script surface only.
- Perturbation limits, each from a script that broke a run:
  - A numeric parameter is never perturbed to zero. A zero amount can make a free ability that changes nothing, which the Forge AI re-activates without end.
  - A signed parameter keeps its sign, so a pump stays a pump and a debuff a debuff.
  - Deck-building hints and `AIPreference` SVars are never perturbed.
  - A card-selecting parameter swaps its selector from a checked-in whitelist, and is never a duration, type name, chain label or description that reads like one. A `Duration$ Permanent` swapped to a selector ended an animation at once and a state trigger re-fired it, thousands of records a game.
  - A protection keyword swaps its quality to one from `PROTECTION_QUALITY_WHITELIST`.
  - The whitelists exist because a shifted `CounterType` or a generated selector produces a script that fails to load rather than a card that plays differently.
- A variant is named `<card> Variant <h>`, `h` from the SHA-256 of its perturbed script, so a file already under that name is the same script. It is reused and never rewritten, since an earlier run's records join it by key. Every variant stays in the tree.
- No variant is generated from a held-out card. Perturbing a parameter changes the text, so the variant's text would not be held out and would reach training.
- No variant of any other card may print a held-out text or template. Such a variant is left out of the decks, and deleted with its sidecar only when this run wrote it. The command takes `build-corpus`'s holdout flags and must be passed the same values.
- The sidecar pass runs before the first game: a variant without a sidecar produces records nothing can join, and the join is designed to fail loudly.
- The loadable names go to `loadable.lst`, never a `.txt`: the whole tree is staged into Forge's custom-cards directory, and Forge reads every `.txt` there as a card script. Only loadable variants are decked.
- When the run ends, however it ends, the staged scripts are taken back out of Forge's custom-cards directory. A run killed too hard to do so leaves them, and the first JVM the next command of any package starts removes them (`forge_jvm.clear_stale_staging`). While a `collect-variants` process is running, every other command leaves its staging alone (`.staged-by-effects.owner`).
- The command exits 1 when the run's shards hold no records. Every deck materializing as basic lands, which is what a staged script Forge never registered produces, otherwise plays its matches and reports success having collected nothing.
- A variant's games are counted from this run's own shards, so another run writing to the directory never counts. The exception is the runs `--continues RUN_ID` names: an interrupted run restarts into the same directory with the same `--seed`, the earlier run's shards are credited first, and a finished variant is not played again. `--continues` resumes the untargeted round only.
- A steered deck names its rule in a fourth deck-file column. Java `GeneratedDeck.parse` splits with limit 4, so the rule never folds into the last card's name. A steered game's records carry both decks' rules as `aimed_at`.
- The steered round needs `--rule-counts`, the corpus's own gen-3 counts.

## `validate-corpus`

- Meant for the first few minutes of a collection pass, not for a finished corpus.
- Every invariant reports the number it measured whether it passed or not. A threshold a run barely clears is what an operator needs to see.
- Judged and watched are separate. A check with no settled threshold prints `[WATCH]` and cannot fail a run; giving it a floor flag promotes it to a verdict without changing what is measured. A new number is published watched first.
- An invariant that fails on every good run is worse than no invariant. "Each `game_id` names one game" therefore exempts two cases precisely: the mana-reservoir flush (a resolution half whose gen-3 `outcome` is written null; before gen-3, one holding a `mana_produced` and otherwise only the `tapped` events of its cost; recognised by its payload, not by the acting card) and a fired trigger record the per-line trigger reservoir writes at the game's end. The measurement says how many records the exemption covered.
- A gen-3 cost half standing alone, its item countered or still on the stack at the game's end, is a paired link.
- Several effect halves per `link_id` are accepted only when they are modes of one root.
- The share of fired trigger records is measured over games that ended in the window only. A fired trigger record is written at its game's end, so a running game holds its negatives alone.
- Games without a game-end marker are watched, since a game still running at the window's end has none. The check fails only when no game wrote a marker while one is followed by a later game of its JVM lifetime.
- A shard line that does not decode fails "every shard line decodes", naming the shard and line; the other shards are read on.
- `--variant-scripts` names the variant tree for the key joins. A keyword key in a tree the run was not given fails the keyword-join check as unchecked.

## `field-coverage`

- Names every record field a corpus never varied; on gen-3 records it ends with one row per field of the gen-3 snapshot, then one row per parameter of each event type the corpus holds. Why it exists: `collection-internals.md` § Dead fields.
