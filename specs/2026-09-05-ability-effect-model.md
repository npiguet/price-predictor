# Ability effect model

The ability effect model is a pretrained model of what each card ability does in play. It ships two artifacts: a per-ability-line embedding cache (one fixed-width vector `e` per unique ability line, computed offline) and a state-conditional effect head that predicts an ability's effect on a given game state. Training data is a corpus of game-effect records collected from instrumented Forge matches. Rationale, rejected alternatives, corpus statistics, and Forge feasibility evidence: [`experiments/2026-09-04-ability-effect-model-design.md`](../experiments/2026-09-04-ability-effect-model-design.md).

**Scope:** corpus collection, the provenance join (§ Ability identity), the keyword-definition extractor, the model, training, the cache and its consumer-side layout contract, and the evaluation battery and gates. Consuming `e` in the scorer, picker, or draft agent is a separate later feature.

# Package layout

- New Python package `src/effects/` (hexagonal: `domain` → `application` → `infrastructure`), CLI `python -m effects <subcommand>`.
- `effects` imports from `price_predictor` (tokenizer, vocabulary builder, `forge_jvm` worker helpers) and `sealed` (`manabase.compute_basic_lands`). Never the reverse.
- Java collectors live in `forge-connector`. The engine hooks live as one commit per hook on the `effect-record-hooks` branch of the sibling `../forge` checkout — the attribution hooks, the missing cause at the `Destroyed` firing site, the trigger-fire and playability logging points (§ Collectors), the mana-production point, the per-static layer accessors and their `…Without` recombinations, and the combat damage assignment. A Forge upgrade rebases the branch. This repository carries no exported copy; the inventory of what the hooks are lives in `PatchHooks.REQUIRED`, which is also what detection reads. Workers detect hook presence at startup, run degraded (bracket-only attribution) on stock Forge, and report which hooks are absent.
- `forge-connector` carries the effect-collector classes, instrumentation in `MatchWorkerMain` behind the `--effect-records` argument, and `KeywordDefinitionMain`.
- Model artifacts under `models/effects/`; corpus and generated files under `output/effects/`.

# Corpus

Location: `output/effects/records/`, shard files named `{run_id}.{worker}-{lifetime}.jsonl`, one per worker JVM lifetime, one JSON record per line, append-only. Readers load every `*.jsonl` in the directory and tolerate a trailing partial line.

The record schema is fixed before stage one (§ Stages); later stages widen the corpus without invalidating earlier records.

## Record envelope (every record)

| Field | Contents |
|---|---|
| `record_id` | `{run_id}.{worker}-{lifetime}.{counter}` — unique across the run's shards |
| `run_id`, `timestamp`, `game_id` | run UUID, ISO 8601 UTC, and `{run_id}.{worker}-{lifetime}.{game counter}`; both carry the worker slot, since each worker counts independently, and the JVM lifetime, since each worker JVM counts from zero and is replaced hundreds of times per run |
| `kind` | `resolution` \| `rewrite` \| `continuous` \| `combat` \| `trigger` \| `playability` |
| `moment` | `resolution` kind only: `activation` (cost half) \| `resolution` (effect half) |
| `subkind` | `playability` kind only: `decision` \| `attackers` \| `blockers` |
| `link_id` | joins the two halves of a resolution pair; absent on a half with no partner (a `fizzled`, `countered`, or `declined` cost record, an interventional effect half) |
| `mirror_of` | `fork = true` records only: the `record_id` of the same-game real record the fork mirrors (the real resolution, or the real combat record), where one exists |
| `variant_of` | `synthetic = true` records only: the name of the card the perturbed script was derived from |
| `mode` | `patched` \| `degraded` — attribution channel available when collected (§ Collectors) |
| `interventional`, `fork`, `synthetic` | booleans; `fork` marks a record collected on a game copy, `interventional` a forced resolution, `synthetic` a perturbed-script card |
| `actor_player` | the acting ability's controller; combat records: the active player; `playability` records: the player whose decision point is logged |
| `ability` | provenance key(s) of the acting line (§ Ability identity); the chosen `option` line (the converted format's modal-option line type) on modal resolutions; absent where no single line acts (`combat`, `playability` — candidates or anchored attacker live in the payload) |
| `state` | pre-event state snapshot |
| `payload` | per-kind object holding the fields below |

`mode`, `interventional`, `fork`, `synthetic`, and `ability_unresolved` (why an acting line could not be keyed) never reach the model. Probe records (§ Collectors) are combat records with `fork = true` and `interventional = false`. Interventional resolutions are resolution records with both `interventional = true` and `fork = true`, storing the fork's own state.

## State snapshot

State is stored as data — names plus dynamic attributes. The tensor representation is derived at training time, so representation changes are code changes, never re-collection.

| Block | Fields |
|---|---|
| `global` | turn, phase, active player, priority player, stack size, combat substep, command-zone emblems |
| `players[]` | id, life, hand/library/graveyard sizes, poison, energy, this-turn counters (creatures died, spells cast, lands played), floating mana by color, untapped production by color |
| `entities[]` — identity | name, face, copy-source; token-script id for tokens; zone; controller |
| `entities[]` — computed characteristics | current type line (types, supertypes, subtypes), colors, mana value, power/toughness decomposed (base, continuous boosts, counter contribution) |
| `entities[]` — board state | tapped, summoning sickness, damage marked, counters by type, combat status (attacking; blocking / blocked-by refs), attached-to, face-down |
| `entities[]` — granted abilities | attachment-granted abilities (provenance keys); temporarily granted keywords and abilities (read from the timestamped change tables; provenance keys where they resolve to lines) |
| `entities[]` — stack extras | a stack entity's own targets, announced per-target amounts (divided damage), and chosen counts for "up to N targets" |
| `refs` (in `state`) | targets (entity/player refs), source entity ref, chosen modes, X and announced values, resolution-time engine choices |
| `pending_event` | rewrite and trigger records: a reference to the payload's incoming event, naming the entities it is about to affect (the event itself lives in the payload) |

- Characteristics are computed (post-layer), never printed. Exception: a continuous record's snapshot has the acting static's own contributions removed from every layer channel it wrote (boosts, granted keywords, types, colors, names).
- Inclusion tiers, in order:
  1. referenced objects (target, affected, source) — every entity-valued ref is carried as an entity, in whatever zone it sits, with that zone recorded;
  2. core: global, battlefield entities, and command-zone effect cards (player-scoped continuous effects with no permanent, carried as entities with the originating line's key);
  3. unreferenced stack contents;
  4. unreferenced hand and graveyard contents.

  Tiers 1 and 2 are in every snapshot from stage one; tier 3 joins at stage two and tier 4 at stage three. A record carries the tiers current at its collection; readers treat an absent tier as uncollected, not empty.
- Perspective is not stored: controllers are absolute ids, and mine/opponent tags are derived at training time relative to `actor_player`.

## Events

Effect payloads carry typed event lists: `{type, subjects (refs), params, duration, attributed_to}`. The type vocabulary is the union of Forge trigger types, bus events, and bracket diffs (§ Collectors). The canonical member list and per-type field normalization live in `src/effects/domain/event_schema.py`; a checked-in completeness test maps every Forge effect API class to a covered event type or an explicit exclusion.

**What the vocabulary now promises.** Three types the schema once declared are retired rather than wired: `cost_adjusted`, `damage_assignment_ordered`, and `name_change`, because each named a fact a payload field already carries under another name (`playability/decision`'s `candidates[].cost_after_adjustment`, `combat`'s `assignment_choices`, `continuous`'s `contributions[].name` respectively). `SUPERSEDED_EVENT_TYPES` in `event_schema.py` names all three and where the information actually lives; giving one fact two spellings would let a reader believe a corpus missing the event was incomplete rather than correct. Every other declared type is reachable by one of two mechanisms:

- **A bus subscription**, where Forge already broadcasts the fact on its own game-event bus and a collector listens. Four types arrive this way — `energy_change`, `radiation_change`, `speed_changed`, `day_night_changed` — and `energy_change` is the channel's canary: Aether Hub is the cheapest sealed-legal card that exercises it.
- **A clause hook plus an API-emitter table**, where nothing is broadcast and the collector instead wraps the ability clause itself. Before and after a `SpellAbility` resolves, a table keyed by the effect's `ApiType` name (not its effect class — a rename costs a missing event rather than a compile error) says which event, if any, that API produces and how to build it from the clause's own parameters and whatever a memo captured beforehand (`ApiEvents` in `forge-connector`). A handful of APIs — coin flips, clash, votes, piles, venture, copies, and the restriction-granting effects, plus the two outcomes that live at an engine choke point rather than inside one effect class (`card_revealed` at `GameAction.reveal`, `damage_prevented` at `ReplacementHandler.runSingleReplaceDamageEffect`) — report through an explicit call the effect itself makes into the same per-clause channel, because the fact they carry (which side called correctly, what the ballot held, which cards the restriction actually landed on) exists only inside the effect's own resolution logic and not in its before/after parameters. This mechanism accounts for 22 of the 29 types that were unreachable for the life of the project before this plan.

**A declared type nothing emits is indistinguishable, in a corpus, from one that is merely rare — and that is exactly the state 29 of the vocabulary's types were in, silently, before this was noticed.** Two checks, neither sufficient alone, stand where that silence was — and the guard itself carried a version of the same blind spot, at a different level, until a Task 10 fix round closed it:

- `test_no_declared_type_is_unreachable_by_surprise` (`test_event_schema_completeness.py`) statically scans the connector's Java source for a reference to each type's constant, and fails if the set of types with no reference at all (`KNOWN_UNEMITTED`) changes without the test being updated to match. **It checks against the full 78-member declared vocabulary; until this fix round it checked against only the 57 types with their own params row, so the other 21 were never compared at all** — the same "scans references, not fires" failure mode, recurring one level up, inside the guard. Widening it surfaced 15 genuinely unreferenced types (`KNOWN_UNEMITTED` is no longer empty, and that is the check finally telling the truth, not a regression); even at full width it still **cannot** catch a type referenced from code that never runs on the expected path: `damage_prevented`, `spell_copied`, and four more found by the Task 10 measurement (`token_created`, `dice_rolled`, `spell_countered`, `regenerated`, `library_shuffled`, `player_won` — six in total) were all referenced in the connector for the whole of their respective corpora while firing zero times from an ability's own resolution, and a scan that never executes the code it reads cannot see that.
- `event_type_coverage` (`validate_corpus.py`) measures, over an actually collected window, which declared types the corpus's records contain at all. It is watched, not judged — a short window legitimately misses types that are rare or depend on which decks were drawn — but it is never silent: every run's report names exactly which declared types it did and did not observe, so the gap the static guard cannot see — at any width — has somewhere to show up. This is why it is the guard's permanent counterpart rather than a one-off measurement.

The itemized rulings behind the less obvious corners of the vocabulary — `coin_flipped`'s two result dialects and its `FlipUntilYouLose` loss count, `damage_prevented`'s relationship to its `rewrite` sibling, `restriction_change`'s subject convention, `vote_taken`'s ballot, the two `choice_made`-family keys that are not spelled like their effect classes, Clash's reveal, the simulated-game guard, the 15 types the guard's own domain hid, and the six declared types that still do not fire, with their card counts — are recorded in `023-ability-effect-model/contracts/record-schema.md` § Events, alongside the field-level contract they each constrain.

## Per-kind payloads

| Kind / moment | Payload |
|---|---|
| `resolution` / `activation` | costs paid: mana by color, permanents tapped, life paid, cards sacrificed / discarded / exiled as costs; `outcome` ∈ {resolved, fizzled, partially_fizzled, declined, countered}, where `declined` is an optional effect the controller was offered and turned down. `resolved` and `partially_fizzled` have a linked effect half; `fizzled`, `countered`, and `declined` have none — the cost record stands alone |
| `resolution` / `resolution` | attributed event list; attribution granularity is the sub-ability (the sidecar maps sub-ability links to lines — § Ability identity; the root line is the fallback) |
| `rewrite` | incoming event; outgoing event **or null where nothing was rewritten in place**; `result` ∈ {replaced, not_replaced, prevented, updated, skipped}; `replaced_by`, the provenance key(s) of the ability that ran instead (empty where none). One record per replacement evaluation, `not_replaced` included — those are the channel's negatives, the same role the trigger channel's non-fired evaluations play, and are deliberate rather than noise. Parameter maps deep-copied at the hook |
| `continuous` | per-entity contributions (P/T boost, keywords, types, colors, name), coalesced per (game, static, board hash) — one record per stable board |
| `combat` | declared attackers, block assignments, damage-assignment choices (inputs), damage-step event list (outcome); one record per damage step, so a first-strike combat produces two |
| `trigger` | the event, fired flag; non-fired negatives drawn from same-event-type evaluations at roughly 1:1 |
| `playability` / `decision` | candidates: per candidate the ability key, rules verdict (can-play, affordable, has-legal-target), legal-target refs, cost after adjustment, and the responsible static's key where one applies |
| `playability` / `attackers` | legal-attacker refs (absolute), one record per combat-setup decision point; per forbidden attacker, the responsible static's key where one applies |
| `playability` / `blockers` | anchored attacker ref, per-entity legal-blocker bits with the responsible static's key on forbidden blocks where one applies, attacker `min_blockers` (cardinality constraints such as menace) |

Verdicts are the rules-level checks only; the AI's policy judgments (e.g. "another time", "life in danger") are never recorded.

**Why the `rewrite` payload is wider than `{incoming, outgoing}`.** That pair assumed a replacement edits the event's parameter map. Forge mostly does not: `ReplacementHandler` runs the `ReplaceWith$` ability and then records a `ReplacementResult`, and for `Prevented`, `Skipped` and `NotReplaced` it never touches the map. "Enters tapped", "if it would die, exile it instead" and "prevent that damage" are substitutions — a *different* ability runs — so the before/after maps the hook copied were byte-identical, and the collector was right to drop them. Measured on the corpus this feature is collecting: **34 rewrite records in 1.88M**, against a 7% `--kind-mix` share, with **87% of candidates dropped as identity**. The missing facts — which of the five results happened, and which ability ran instead — were available at the hook all along. The schema is fixed before collection and later stages may add `kind` values or snapshot tiers but never redefine a field; this obeys that. `incoming` keeps its meaning and type, `result` and `replaced_by` are new fields, and `outgoing` only *widens* to admit null, which is the honest spelling of the case the old shape expressed as a self-copy and then discarded. Full argument in `023-ability-effect-model/contracts/record-schema.md`.

## Collection caps and budgets

Flags live on every collecting supervisor (`match-outcomes`, `collect-coverage`, `collect-variants`).

| Flag | Default | Meaning |
|---|---|---|
| `--mana-cap` | 1 | cap on resolution records per unique (mana ability, mana produced) pair, per game |
| `--playability-rate` | 0.1 | fraction of `decision`-subkind logging points sampled; `attackers`/`blockers` records are sampled at `--legality-rate` instead |
| `--legality-rate` | 0.1 | fraction of `legality`-subkind logging points kept, sampled **after** the dedup |
| `--interventions-per-game` | 2 | interventional resolutions per game (stage three) |
| `--probes-per-game` | 2 | damage-step probe forks per game (only for keywords whose canary failed — § Evaluation, gate 2) |
| `--probe-keywords` | _(none; probes disabled)_ | the canary-failing keywords to probe, comma-separated |
| `--snapshot-tiers` | `1,2,3` | snapshot inclusion depth, a prefix of `1,2,3,4`; stage three collects at `1,2,3,4` |

Continuous records need no cap: coalescing per stable board is the cap.

Every one of these is a per-worker-process quantity the supervisor cannot observe, so each travels to
the JVM as an `effect.*` system property; the flag set, the property set and the JVM's
`CollectionCaps` record are one set, pinned by a test that reads both sides. `--snapshot-tiers` is
run-level rather than per-collector on purpose: a depth chosen per call site makes `state.tiers` a
proxy for *how* a record was collected, and the first corpus put tier 4 on the interventional records
and nowhere else — a perfect predictor of a flag the schema forbids the model to see.

# Ability identity

The provenance sidecar is the join between runtime trait objects and converted lines. The join key is printed provenance: (script file, face, trait kind, index within that kind's slice of the face's raw trait list). Converted-line ordinals are never a key.

- `python -m price_predictor convert` writes a sidecar next to every converted card: `<name>.provenance.json` (same pairing convention as the `.npz` embedding files). Sidecars never alter the converted text.
- Per rendered line, the sidecar carries:
  - the provenance key list — a line merged from several traits carries several keys; a trait deduplicated away maps to no line;
  - the sub-ability links the line covers, as index paths below the trait — an event attributed to an unmapped link falls back to the root line;
  - the trait's script API type, its parameter-key list, and its script line as text — the stage-four primary encoding surface, so every command that encodes reads it from the sidecar and needs no Forge-cardsfolder path of its own;
  - role spans over the prose — character ranges tagged `cost` \| `effect` \| `trigger-condition` \| `target-spec`.
- Runtime keys are computed from the trait accessors. Granted abilities resolve through the grantor accessors to the donor card's printed line; copied abilities through the original-ability back-reference; copy-spell effects (Fork, Reverberate) carry only a copied flag and resolve through the stack object's source card.
- `python -m price_predictor convert` also converts Forge's token scripts, from `../forge/forge-gui/res/tokenscripts/` into `output/tokenscripts/`, with sidecars of their own. They stay out of `output/cardsfolder/` because converted token and card filenames collide (Ajani's Pridemate is both) and the sealed pipeline treats that tree as its card corpus. Commands that read converted text take it as another `--cards-folder`.
- The provenance key's script-file component is the path including its script tree (cardsfolder, tokenscripts, or variant-scripts), since the same filename occurs in more than one.

# Collectors

All collectors write the one schema. Instrumentation is opt-in per run: `python -m sealed match-outcomes --effect-records output/effects/records/` forwards the flag to the Java workers; `python -m effects collect-coverage` runs the same instrumented workers over coverage decks. Adding `--effect-records` to `match-outcomes` leaves that command's own outputs unchanged.

- **Channels.** Stage one, no Forge patch (`mode = degraded`): the public event bus plus a bracket around stack resolution; events attribute to the resolving ability by bracket. The patch (`mode = patched`) adds three attribution hooks: the trigger-handler cause channel, the shared replacement execution point (parameter map deep-copied before the call), and a threaded currently-resolving-sub-ability pointer.
- **Bracket rules.** State-based-action deaths attribute to the bracket they follow; combat damage attributes to its damage-step bracket.
- **Continuous effects.** Read from the per-card layer tables ((timestamp, static id)-keyed) after a recompute; static id 0 entries (temporary pumps) stay attributed to the resolution bracket.
- **Stat-change diffs.** Stat changes that fire no trigger (pumps, anthem recomputes) are captured by recomputing and diffing computed stats inside the bracket whenever the stats-changed bus event fires, coalesced per bracket; trigger-channel events take precedence where both report.
- **Mana abilities.** Mana records ride the inline-path cast/resolution triggers, exist only under `mode = patched`, and honor `--mana-cap`.
- **Trigger-fire logging.** The hook sits at the trigger handler's condition evaluation.
- **Playability logging.** Hooks sit at the AI's candidate computation, combat-setup legality, and the legality/cost-adjustment checks. The logger snapshots defensively: a verdict may be abandoned mid-evaluation, and the legality check mutates the checked ability's targets, so the logger never reuses a checked ability object.
- **Interventional resolutions.** From stage three, these run Forge's game simulator on a fork with chosen targets/modes: unaffordable candidates go through the play-without-paying-mana path, the ability is located on the copy and verified through the provenance key, and the record stores the fork's state. Force-resolving drains the fork's stack, so fork records attribute by bracket. An intervention writes the effect half only: the forced cast pays no real cost, so no activation record is written and the effect half carries no `link_id`. Every intervention counts against `--interventions-per-game`; within that budget, at most 2 forks (a fixed constant, independent of the flag) target the same real resolution.
- **Probes.** Stage three, contingent per keyword on the damage-step canary: fork at declare-blockers after blocks lock, strip one keyword from one participant below the layer system (with the keyword-cache refresh), resolve the damage step. Both branches run under an installed seeded random source restored in a `finally`; one concurrent game per JVM while probes are on. The fork branch is recorded as an ordinary combat record with `fork = true`; the real-vs-fork diff is computed only at evaluation time.
- **Budgets and guards.** Every fork counts against its per-game budget flag. Each fork's copy is score-checked against the live game at creation, before any perturbation (Forge's copy-score guard); a mismatch logs a warning and discards the fork — it still counts against the budget, and no record is written.

## The operating condition: workers are short-lived on purpose

Every collecting supervisor runs its Forge workers under `-Xmx1200m` and kills the longest-running one every status interval (60 s), restarting it immediately. **Neither is a tuning knob.** Forge misbehaves late in a JVM's life — loops that do not terminate, allocation that grows without bound — and the two together are the containment: a small heap turns a runaway worker into an early, cheap death the pool restarts, and the hard recycle bounds the damage of the pathologies that do not allocate. Raising the heap or lengthening the interval hides the pathology rather than containing it, and costs far more than the restarts do. They are one measure, and a proposal to remove either has to explain what contains Forge instead.

The price is paid and accepted: a handful of GC crashes per overnight run, and a crashed worker costs the block of records in flight and nothing else.

**The consequence a reader has to draw from this is about correctness, not throughput.** Restarts are the normal operating condition, not an accident: an eight-hour collection run is roughly 530 JVM lifetimes across six worker slots, and many of them are killed mid-game with `taskkill /F /T`, which runs no JVM shutdown hook. So:

- **Anything a worker counts from zero must be namespaced by the JVM lifetime, not by the worker slot.** The first collected corpus was not: `record_id` and `game_id` were `{run_id}.{worker}.{counter}` over counters living in one JVM's heap, so every restart reissued ids the previous lifetime had already used — 71–75% of record ids were repeats, and 804 `game_id` values covered 31,662 games, which merged unrelated games under the corpus's own join key. The lifetime segment in the id is what fixes that, and any future per-process counter needs the same treatment.
- **Game boundaries do not coincide with process boundaries.** A game can be cut in half by a kill, and the next lifetime starts mid-nothing. A reader must never infer a game boundary from a file boundary, and a record's game must be readable from the record.
- **Resume logic must survive a JVM killed mid-game.** A shard ends at its last complete 256-record block, so the last ≤255 records of every lifetime — about 0.5% of a run, always the tail of a game — are simply absent. Any resume that assumes a clean shutdown wrote a marker is wrong.
- **A validator, not a post-mortem, is what catches this.** `python -m effects validate-corpus` measures these invariants over the first few minutes of a pass, because every defect above was already visible in the first minute of shards and none of them raised anything for eight hours.
- **A check that cries wolf is worse than no check.** The validator's own game-boundary invariant failed 168 of 177 games on a *healthy* corpus, because mana activations are reservoir-sampled and flushed at game end carrying the snapshot of the turn the mana was made. That is the documented behaviour of the mana collector, not a defect, and an operator who sees it fail every run stops reading the whole checklist. So the case is exempted precisely — a resolution half whose every event is `mana_produced`, and nothing else — and the report says how many records the exemption covered. For the same reason a check with no settled threshold prints `[WATCH]` and cannot fail a run: the number is reported every pass, and it becomes a verdict when an operator passes a floor rather than by guessing one now.

# Synthetic script variants

From stage four, `python -m effects collect-variants` emits perturbed card scripts and collects engine-ground-truth records for texts that never existed, through the same instrumented workers and the same caps and budgets.

- Perturbations: each variant edits one parameter of one script read from `--forge-cards-path` (default `../forge/forge-gui/res/cardsfolder/`) — a numeric parameter shifted by up to ±3 or doubled, floored at zero, or a selector swapped for one drawn from the checked-in whitelist in `src/effects/domain/script_variants.py`.
- `--variant-volume` (default 0.2): the cap on variant records as a fraction of the real records already in `--effect-records`.
- Perturbed scripts are written to `output/effects/variant-scripts/` and loaded from there as custom cards; they never enter `output/cardsfolder/` or Forge's own tree.
- A variant exists on the script surface only. Converted prose is a script's hand-written description, so a perturbed script's description still describes the original — editing `NumDmg$ 3` leaves the prose saying three. Variants are therefore never converted, carry no prose surface, and are excluded from the paired-encoding loss; their records train the script side alone. The provenance key of a variant line is the perturbed script's own, so `collect-variants` writes a sidecar per variant script and needs no converted tree.
- Variant records carry `synthetic = true` and `variant_of` — corpus metadata, never model inputs. A variant of a held-out card is held out with it, so the card-disjoint split cannot leak through a one-parameter edit.
- Variant cards are decked and scheduled exactly as coverage decks are (§ Coverage collector), over the variant set rather than the converted corpus, with the same rounds and `--decks-per-round`.
- Variant matches write effect records only — never `match-outcomes.txt` or `cards-played.txt`. Games played with perturbed cards are not sealed self-play and must not feed the scorer or encoder corpora.

# Coverage collector

`python -m effects collect-coverage` reaches the converted cards sealed pools never contain, by putting them in decks that get played.

- Works in rounds. A round rebuilds the weighted decks, plays `--decks-per-round` (default 500) of them as matches, and recounts coverage; `--no-progress-rounds` below counts these.
- Reads the held-out card list from `--split-from PATH` (a checkpoint) when one is given, so coverage decks exclude the same cards the trainer holds out; without it the run builds over every card and its games are usable only by a checkpoint whose split holds nothing they contain.
- Builds 40-card decks over the whole converted card corpus, not sealed-legal sets. Deck candidates and the coverage unit come from the `output/cardsfolder/` entry of `--cards-folder` alone, since a token script is not a deckable card. Decks are weighted toward cards with the fewest effect records: 23 nonlands plus basics from `compute_basic_lands`. Excluding the held-out cards matters here: a deck of 23 cards drawn corpus-wide would otherwise contain one almost every game, and the whole coverage corpus would fall to the training exclusion.
- Consults Forge castability to weight slots toward cards Forge actually plays. The consult only ranks; it never drops a card from deck building, because being in a game is the precondition a stage-three intervention forks from, so every uncovered card still gets slots.
- Two residues are reported at the end of a run and fall to interventional resolutions (stage three): cards the consult judges uncastable, and cards it judges castable that never reach `--target-records`. A retired card goes to the residue matching its consult verdict.
- Runs instrumented matches over those decks with the same worker and flags; `--effect-records` (default `output/effects/records/`) names the destination shard directory — on `match-outcomes` the same flag is the opt-in and has no default.
- `--target-records` (default 50): the per-card goal — a card is satisfied once that many records, counted over every shard in `--effect-records`, have it as the acting line's host, an event subject, or a referenced ref. Merely sitting on the battlefield in a snapshot does not count, and the unit is not resolution records specifically, since a vanilla or keyword-only creature has no acting line and could never produce one.
- The run stops when every card is satisfied or retired: a card that gains no new qualifying record across `--no-progress-rounds` (default 3) consecutive rounds is retired, so the stop condition always terminates.
- Coverage matches write effect records only — never `match-outcomes.txt` or `cards-played.txt`. They are not sealed self-play and must not feed the scorer or encoder corpora.

# Model

Two transformers trained jointly; training-only auxiliaries are filtered out of the saved artifact.

## Ability encoder

- **Surfaces.** Prose is the sole surface through stage three, with the script-API classification auxiliary standing in for script structure; from stage four the Forge script line is the primary surface and converted prose the paired secondary.
- **Vocabulary.** Effects-side vocabulary at `models/effects/vocab.txt`, built by `python -m effects build-vocab`: wraps the shared vocabulary utility (`--cards-folder`, repeatable, defaulting to `output/cardsfolder/` and `output/tokenscripts/`; `--vocab-path`, default `models/effects/vocab.txt`; `--target-size`, default 5000); the scan covers converted cards, converted token scripts, and the keyword-definition file (`--keyword-definitions`, § Keyword definitions); seeded specials `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]`. `--keyword-definitions` defaults to `output/effects/keyword-definitions.json`; `--surface` (`prose` by default) selects the scan and the default of `--vocab-path`.
- **Tokenization.** Whole-token only: words outside the vocabulary map to `[UNK]`, with no subword fallback; unknown keywords are covered by forced expansion, unknown subtypes and token names by the next vocabulary rebuild.
- **Script-surface tokenization.** At stage four `build-vocab --surface script` adds the sidecars' script lines to the scan and writes `models/effects/vocab-script.txt`, a path of its own so the rebuild never overwrites the prose vocabulary that stage-one-to-three checkpoints record. The script tokenizer splits compound selectors compositionally (`Creature.nonDragon+OppCtrl` → `Creature`, `nonDragon`, `OppCtrl`).
- **Number tokens.** A monotone numeric embedding: a shared learned base vector plus log1p(n) times a learned direction.
- **Prose tokens.** Each token adds a role embedding (`cost` \| `effect` \| `trigger-condition` \| `target-spec`) looked up from the sidecar's role spans.
- **Keyword-expansion dropout.**
  - With probability `--keyword-expand-p` a keyword token is replaced by its definition text; keywords unknown to the vocabulary are always expanded. From stage four the definition is the captured script on the script surface and the reminder template on the prose surface, falling back to the template where no script exists.
  - Parameterized keywords instantiate the template with the instance's own values; a keyword referenced without an instance (inside another definition) expands with the template's generic wording.
  - Keywords whose body lives on the host card (saga chapters, class levels) never expand.
  - Keywords inside an expansion stay tokens, themselves subject to dropout on other samples.
- **Pooling.** A `[CLS]` aggregation token pools to the bottleneck `e` (`--e-dim`), with additive Gaussian noise during training (`--e-noise`).
- **Training-only heads.** MLM over masked tokens (`--mlm-weight`, `--mlm-mask-prob`) and a script-API classification head from `e` (API type plus parameter-key set; `--api-weight`). Stage four adds the paired-encoding loss — asymmetric, stop-gradient on the script side — over the lines that have both surfaces; synthetic variants have only the script surface and contribute no pairing term.

## Effect head input

One token per slot; position ids reset at each `[CARD]`:

```
[GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e ... [CARD] e ...
```

| Slot | Carries |
|---|---|
| `[GLOBAL]` | turn/phase/priority fields, whose turn, stack size, combat substep, emblems, and the record kind/moment/subkind flag |
| `[ACT]` | the acting line's `e`, announced values (X, kicker, chosen modes, resolution-time choices), and the resolution outcome flag (from the paired cost record; `resolved` when there is no partner); empty for combat records and the `attackers`/`blockers` playability subkinds |
| `[PLAYER]` (per player) | the snapshot's player fields plus the targeted flag |
| entity `[CARD]` token | structured features (type line with subtype-token mean, colors, mana value, P/T decomposition) plus overlay (tapped, sickness, damage, counters, combat status, damage-assignment choices, attached-to, temporarily granted keywords and abilities, controller tag, targeted flag, source flag, zone, face-down, pending-event fields; for stack entities, their own targets, announced per-target amounts, and up-to-N counts) |
| entity ability tokens | the entity's ability `e` vectors following its `[CARD]` token — printed lines plus attachment-granted lines only; temporary grants ride the overlay |

- Controller tags are mine/opponent relative to `actor_player`.
- Numeric overlay and player scalars enter as raw values plus a log1p copy; nothing is binned.
- A `playability`/`decision` record trains as one example per candidate, the candidate's `e` in `[ACT]`.
- Context-ability dropout: each context ability token is dropped with probability `--context-dropout` during training.

## Output heads (kept in the saved artifact)

- **Per-entity head.** One shared head mapped over every `[CARD]` and `[PLAYER]` output: an affected/unaffected gate, then conditional field groups:
  - permanents and stack entities: zone outcome (categorical: stayed, died, exiled, to hand, library top, library bottom, transformed, face up/down, phased out, blinked, countered), tap state, damage taken, counters delta by type, P/T delta with duration, type/color delta with duration, keywords gained/lost, control change, attached-to;
  - players: life delta, cards drawn/discarded/milled, library events (scry, surveil, tutor, reveal, reorder), mana delta;
  - legality bits: target-legal relative to `[ACT]`; attacker-legal; blocker-legal, with `min_blockers` read at the anchored attacker.
- **Created-objects head.** At `[GLOBAL]`: K = 4 group slots, each `{present, scripted flag, token-script id or characteristic fields (P/T, type flags, keyword flags), count}`. Slots follow a canonical order (sorted by token-script id, characteristics-only groups last by descending count); an overflow flag covers more than 4 distinct groups.
- **Verdict head.** At `[ACT]`: playability verdict bits (can-play, affordable, has-legal-target), predicted cost paid, trigger-fired bit.

Each record kind is an input variant of this one surface:

| Record | Input variation | Supervises |
|---|---|---|
| `resolution` / `activation` (cost half) | the outcome flag on `[ACT]` | per-entity cost outcomes (mana delta, tap state, life, zone outcomes); the verdict head's cost paid |
| `resolution` / `resolution` (effect half) | targets, modes, announced values and the outcome flag on `[ACT]`; the targeted flags | per-entity fields; created-objects slots |
| `rewrite` | pending-event overlay on the affected entities | per-entity fields and created-objects slots, **derived from `outgoing` and therefore only where `outgoing` is non-null** — an in-place parameter edit. The substitution outcomes (`prevented`, `skipped`, most `replaced`) and every `not_replaced` carry a null `outgoing` and so supervise nothing here; the substituted ability's own effects arrive as its `resolution` record. `result` is the natural label for those records and **is not wired to a head** — see below |
| `continuous` | the acting static's own layer-channel contributions masked from the entity inputs, structured features included (§ State snapshot) | per-entity contribution fields |
| `trigger` | pending-event overlay | the verdict head's trigger-fired bit |
| `playability` / `decision` | the candidate's `e` in `[ACT]` | the verdict head; per-entity target-legality bits |
| `playability` / `attackers`, `blockers` | `[ACT]` empty; the anchored attacker via the source flag (`blockers`) | per-entity legality bits: attacker-legal, or blocker-legal with `min_blockers` at the anchored attacker |
| `combat` | declared combat and the damage-assignment choices in the overlays, `[ACT]` empty | per-entity damage-step outcomes |

**Open: `result` as a supervision target.** Most `rewrite` records now supervise nothing, because
`derive_targets` reads the payload's events and a substitution has no rewritten event to read. `result`
is a five-way categorical over exactly the records that are otherwise unsupervised, and it is a
rules-level judgment about an ability — the same shape as the verdict head's playability bits and
trigger-fired bit, which is where it would belong. It is recorded and **deliberately not wired**: this
document does not describe a rewrite-result head, and adding one is a modelling decision with its own
loss weight, class balance (`not_replaced` will dominate) and gate, not a consequence of the payload
change. Until it is taken, `result` is corpus content the validator judges and the trainer ignores.

## Losses

- Counts and open-ended magnitudes (damage, counters, life, draws, mana) use Poisson-family count regression. Signed delta fields decompose into a direction (negative, zero, positive — categorical) and a nonnegative magnitude under the count loss; gate 1's deviance (§ Evaluation) is computed over the magnitudes. Closed vocabularies (zone outcome, token-script id) are categorical. Gates and bits are binary.
- Loss is normalized per record over entity count. The affected gate trains on every entity; conditional fields train only where the gate's target fires.
- Field curriculum: the sparse group — keywords gained/lost, control change, attached-to, the duration on every delta that carries one, type/color delta, and counters delta by type — enables at `--curriculum-step` training steps. Every other field trains from step zero, as do all three heads' remaining outputs.

# Training

`python -m effects train-effect-model` — joint, end to end, from random init.

- **Batches.** Each batch (`--batch-size` records, stretched by `--grad-accum`) mixes several games, groups each game's records together, and encodes each unique ability text once. Context gradient is live re-encoding by default; `--context-cache` switches to the stop-gradient momentum cache (refreshed every `--cache-refresh` batches) when the 8 GB GPU budget requires it.
- **Schedule.** An epoch is `--steps-per-epoch` optimizer steps with validation between epochs; `--epochs` bounds the run; early stopping after `--patience` epochs without a new card-disjoint validation best.
- **Sampling mixture.** Per batch, set by `--kind-mix` and renormalized over the classes present in the corpus:

  | Class | Share | Records |
  |---|---:|---|
  | `resolution-effect` | 30% | `resolution` / `resolution` |
  | `combat` | 20% | `combat` |
  | `continuous` | 12% | `continuous` |
  | `playability-decision` | 10% | `playability` / `decision` |
  | `resolution-cost` | 8% | `resolution` / `activation` |
  | `trigger` | 8% | `trigger` |
  | `rewrite` | 7% | `rewrite` |
  | `playability-legality` | 5% | `playability` / `attackers`, `blockers` |

- **Rarity weighting.** Within a class, records weight ∝ effective_games^(−0.5), capped at 20× the weight of the most-observed ability text. Effective games is the count of distinct games contributing a record of that unique ability text — games, not raw records.
- **Records with no acting text.** `combat` and `playability-legality` sample uniformly within their class; a `decision` record's per-candidate examples key on the candidate's text.
- **Splits.** Card-disjoint validation: cards first printed in the newest sets (printing order read from `--printings-path`), taken newest-first until they cover at least 8% of the cards under `output/cardsfolder/` — token scripts and variant scripts are not in the printing order and stay out of the denominator; every game with a record naming a held-out card is excluded from training entirely. Game-disjoint validation: 10% of the remaining games. Best checkpoint by card-disjoint validation loss.
- **Split provenance.** The checkpoint records the split it trained against — the held-out card list, and the `game_id` set enumerating every held-out game across both strata — and `evaluate-effect-model` reads the split from the checkpoint instead of recomputing it. The corpus is append-only and grows between runs, so a recomputed split would not be the trained-against one and the gates would score partly on trained-on games. For the same reason `--split-from PATH` makes a run inherit another checkpoint's split, vocabulary, and keyword-definition paths: every variant run inherits from the `full` run it is a baseline for, and `evaluate-effect-model` fails fast when a `--variant-checkpoint` records a different split than `--checkpoint`.
- **Records outside the recorded split.** Every reported check scores only the recorded `game_id`s, so games appended after a training run are ignored rather than re-derived into a stratum.
- **Vocabulary drift.** The checkpoint also records a hash of the vocabulary and keyword-definition files it trained with. The inference commands hash the files they actually use — the recorded paths, or an explicit override — and fail fast on a mismatch, and `evaluate-effect-model` compares those hashes across `--checkpoint` and every `--variant-checkpoint` alongside the split. `build-vocab` overwrites its target in place, so a rebuild between training and encoding would otherwise silently re-index the embedding table.
- **Checkpoints.** Saved under `--model-output`, which defaults to `models/effects/effect-model/` for `--variant full` and `models/effects/effect-model/{variant}/` otherwise, as `{timestamp}.pt` plus `latest.pt`, so variant runs never overwrite the shipping checkpoint. Each checkpoint records the `--vocab-path` and `--keyword-definitions` it trained with, plus its split, and the inference commands default to those. The saved artifact keeps the encoder, the effect-head trunk, and the per-entity, created-objects, and verdict heads; the MLM, script-API, and pairing heads are filtered at save time.
- **Variants.** Trained on the identical pipeline as evaluation baselines (`--variant`): `identity` (a free embedding per unique text replaces the encoder), `state-only` (all `e` inputs zeroed), `no-state` (every input zeroed except `[ACT]` and the record-kind flag; the entity ability `e` tokens are zeroed too — the average-effect control), `taxonomy` (`e` replaced by an embedding of the sidecar's API type and parameter keys).

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--records-dir` | `output/effects/records/` | corpus shard directory |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | converted text and provenance sidecars; repeatable |
| `--variant-scripts` | _(none; `output/effects/variant-scripts/` at stage four)_ | the perturbed-script tree and its sidecars |
| `--split-from` | _(none; compute the split)_ | inherit another checkpoint's split, vocabulary, and keyword-definition paths |
| `--vocab-path` | `models/effects/vocab.txt` | tokenizer vocabulary; a stage-four run points it at `models/effects/vocab-script.txt` |
| `--printings-path` | `resources/AllPrintings.json` | first-printing order for the card-disjoint split |
| `--keyword-definitions` | `output/effects/keyword-definitions.json` | keyword definitions for expansion dropout |
| `--model-output` | per variant (above) | checkpoint directory |
| `--variant` | `full` | `full` \| `identity` \| `state-only` \| `no-state` \| `taxonomy` |
| `--e-dim` | 64 | width of the bottleneck `e` |
| `--e-noise` | 0.05 | σ of the train-time additive Gaussian noise on `e` |
| `--keyword-expand-p` | 0.25 | keyword-to-definition expansion probability |
| `--context-dropout` | 0.15 | drop probability per context ability token |
| `--mlm-weight` | 0.1 | MLM auxiliary loss weight |
| `--mlm-mask-prob` | 0.15 | MLM masking probability |
| `--api-weight` | 0.05 | script-API auxiliary loss weight |
| `--curriculum-step` | 10000 | step at which the sparse field group enables |
| `--batch-size` | 32 | records per batch |
| `--grad-accum` | 1 | gradient-accumulation factor |
| `--kind-mix` | the listed mixture | per-batch quotas over the eight sampling classes (`resolution-cost`, `resolution-effect`, `combat`, `playability-decision`, `playability-legality`, `continuous`, `trigger`, `rewrite`), `class=share` pairs |
| `--context-cache` | _(off)_ | stop-gradient momentum cache for context `e` |
| `--cache-refresh` | 500 | batches between momentum-cache refreshes |
| `--steps-per-epoch` | 5000 | optimizer steps per epoch |
| `--epochs` | 40 | epoch bound |
| `--patience` | 5 | epochs without a card-disjoint validation best before stopping |

Hardcoded (mirroring the sealed encoder's conventions):

| Constant | Value |
|---|---|
| encoder | d_model 256, 4 layers, 4 heads |
| effect-head trunk | d_model 256, 6 layers, 4 heads |
| `ff_dim` | 4 × d_model |
| dropout | 0.1 |
| optimizer | AdamW |
| learning rate | 1e-4, constant after warmup |
| warmup | linear over the first 5% of scheduled steps (`--epochs` × `--steps-per-epoch`) |
| gradient clip | per-parameter-group max-norm 1.0 |
| seed | 42 |

# Embedding cache and card representation

`python -m effects encode-abilities`:

- Runs the trained encoder (`--checkpoint`; `--cards-folder`, repeatable, defaulting to `output/cardsfolder/` and `output/tokenscripts/`; `--variant-scripts`, stage four; `--vocab-path` and `--keyword-definitions`, defaulting to the paths the checkpoint records from its training run) over every converted card, token script, and — at stage four — variant script, writing one file per card under `output/effects/abilities/`, in a subtree named for its source tree (`abilities/cardsfolder/…`, `abilities/tokenscripts/…`, `abilities/variant-scripts/…`) mirroring that tree's layout: a float32 array of shape (n_lines, e_dim), row-aligned with the source's sidecar — rendered lines for a converted tree, script lines for the variant tree. Per-tree subtrees keep colliding filenames apart, and the cache lives outside `output/cardsfolder/` because the sealed pipeline's `encode-cards --clean` deletes every `.npz` under that tree. Idempotent; `--clean` removes only files this command wrote.
- `--variant` selects both ends together, so a variant is never encoded with another variant's weights: `full` (the default) reads `models/effects/effect-model/latest.pt` and writes `<name>.npz` — the shipping cache every consumer reads; any other variant reads `models/effects/effect-model/{variant}/latest.pt` and writes `<name>.{variant}.npz`, beside the shipping cache rather than replacing it. `--checkpoint` overrides the resolved default.
- The `taxonomy` variant has no encoder: the command emits its `e` construction (the taxonomy lookup over the sidecar's API type and parameter keys) into the same row layout, so every `e`-geometry check reads one file shape.
- Cache-time keyword handling matches inference: known keywords stay tokens, unknown keywords are always expanded.
- The cached vector is the primary surface's `e`: prose through stage three, script from stage four (§ Stages).

Downstream card representation:

- A card is a mini-sequence: a `[CARD]` token carrying the structured features, followed by the card's ability `e` rows.
- Position ids reset to 0 at each `[CARD]` (additive local positional embeddings), so cross-card order carries no signal.
- Multi-face cards separate faces with an `[ALTERNATE]` token tagged with the face's layout, taking its values from the converted format's `layout:` line (the authority on the layout vocabulary); positions continue across faces.
- Consumers read the cache file + sidecar pair: a cache path's subtree names the source tree and the rest of the path names the file within it.

# Evaluation

`python -m effects evaluate-effect-model` (`--checkpoint`, default `models/effects/effect-model/latest.pt`; `--variant-checkpoint NAME=PATH`, repeatable; `--records-dir`, `--cards-folder`, and `--variant-scripts` with the trainer's defaults; `--vocab-path` and `--keyword-definitions` from `--checkpoint`'s recorded training paths) runs the battery over the trained variants and reports per record kind and per stratum. Splits come from `--checkpoint`'s recorded split, never recomputed.

Prediction checks run the loaded checkpoints against records from `--records-dir`. Checks over `e` geometry (gate 3, the decodability battery, the ward canary, the scorer smoke test) read the `output/effects/abilities/` caches, so `encode-abilities` runs first for every variant they cover. Held-out strata: unique-text (line texts appearing on no training card), shared-text (texts also on training cards), novel combinations of seen sub-abilities, numeric extrapolation (seen effect, unseen magnitude). Metrics condition on affected entities and are reported per field, class-balanced within each categorical field; every kind also reports the `state-only` variant as its floor.

**Reported checks:**

- nearest-neighbor inspection and UMAP colored by effect category
- ward canary: ward's `e` closer in cosine distance to each functional twin's `e` than the median of ward's distances to all bare single-keyword `e` vectors; the twins are a checked-in list of ability texts that spell out ward's behavior without the keyword (`src/effects/domain/ward_twins.py`)
- zero-shot keyword: one implemented keyword withheld from training, its occurrences always expanded
- role-polarity probe: predicted mana-pool sign for `{R}` in cost vs effect position (the effect-position half needs mana records, so the probe is informative from stage two)
- scaling calibration: predicted sweeper deaths as a function of board size
- matched real-vs-fork prediction agreement: pairs are the same ability resolved for real and forked in the same game, joined by `mirror_of`
- the `identity` variant on the game-disjoint split: the in-distribution memorization ceiling
- linear-decodability battery on pooled per-card `e` (concatenated mean and max over the card's ability rows), reported side by side with the sealed encoder at `models/sealed/encoder/latest.pt` on the same feature table (ridge harness from [`experiments/2026-08-28-encoder-preferences.md`](../experiments/2026-08-28-encoder-preferences.md))
- pooled-`e` scorer smoke test: the same pooled `e` concatenated with that sealed encoder's vector, written into a scratch copy of the cards folder — never `output/cardsfolder/`, which the sealed pipeline ships from — and `train-scorer` Phase A re-run against it; informational
- `taxonomy` variant comparison: what the full encoding surfaces add over the script API taxonomy alone, on held-out effect prediction and the decodability battery
- average-effect control: the `no-state` variant compared on the decodability battery, the scorer smoke test, the ward canary, and scaling calibration
- probe-diff re-check (stage three, probed keywords only): gate 2's canary re-run over the real-and-fork combat pairs joined by `mirror_of`

**Gates** (numeric). Gates 1 and 3 block shipping the model or the cache. Gate 2 blocks nothing: it is a per-keyword routing decision whose failure schedules that keyword's probe at stage three.

1. **Identity-baseline gate** — card-disjoint split, unique-text stratum, resolution records: the `full` model beats the `identity` variant by ≥ 0.05 absolute affected-gate F1, ≥ 0.05 absolute zone-outcome accuracy on affected entities, and ≥ 5% relative reduction in mean Poisson deviance over the count-valued fields. All three must hold.
2. **Damage-step keyword canary**, run per keyword over first strike, double strike, deathtouch, lifelink, trample, indestructible, wither, and infect; it decides whether that keyword gets a probe.
   - Qualifying records: combat records from the game-disjoint validation split in which the keyword's presence changes the damage-step outcome. At least 200 are required.
   - Perturbation: remove the keyword from the carrying participant — its ability token where the keyword is printed or attachment-granted, and the overlay's temporarily-granted-keywords channel where it is not.
   - Pass: the affected fields move in the keyword's rules direction in ≥ 70% of qualifying records.
   - The qualifying predicate and the rules direction are one row per keyword in a checked-in table (`src/effects/domain/damage_step_keywords.py`), alongside the fields each keyword affects.
   - A keyword that fails the threshold, or has fewer than 200 qualifying records, gets the probe.
3. **Collapse canaries** — over the full cache, one vector per unique ability text: mean pairwise cosine similarity over 10,000 random pairs ≤ 0.5, and the top principal component explains ≤ 30% of total variance.

Run results land in the design record's Outcome section, never in this spec.

# Stages

Each stage widens the corpus without invalidating earlier records.

- **Stage one** — no Forge patch (`mode = degraded`): bus + bracket collection of resolution and combat records; the provenance sidecar; the keyword-definition extractor (reminder-template form); the trainer with the per-entity, created-objects, and verdict heads (fields whose record kinds are absent contribute no loss), the script-API auxiliary, and keyword-expansion dropout over reminder-text definitions. Evaluation at this stage runs `encode-abilities` and `evaluate-effect-model` with the shipping gates 1 and 3, the gate-2 routing canary, and every reported check whose records exist — matched real-vs-fork agreement and the probe-diff re-check wait for stage three, the role-polarity probe for stage two (§ Evaluation).
- **Stage two** — the patch set (`mode = patched`): cause-attributed triggers, mana records, rewrite records, sub-ability attribution; playability, continuous, and trigger records; snapshot tier 3; the coverage collector.
- **Stage three** — interventional resolutions; snapshot tier 4; the damage-step probe for keywords whose canary failed.
- **Stage four** — the script surface: the script-surface vocabulary, the compositional script tokenizer, the asymmetric paired-encoding loss, keyword definitions upgraded from reminder templates to captured scripts, and `collect-variants`.

The damage-step canary covers the damage-step family only, at every stage; evasion keywords are not part of any canary gate (their supervision arrives with playability records at stage two).

# Keyword definitions

`python -m effects extract-keyword-definitions` (Java `KeywordDefinitionMain`) writes the keyword-definition file (`--output`, default `output/effects/keyword-definitions.json`): keyword → reminder-text template for all keywords, plus — from stage four — the generated implementation script captured as text at the keyword factory for the script-generated majority.

# CLI summary

```
python -m effects build-vocab
    [--surface prose|script]     default prose; script adds the sidecars' script lines
    [--cards-folder PATH ...]    default output/cardsfolder/ + output/tokenscripts/
    [--vocab-path PATH]          default models/effects/vocab.txt (script: vocab-script.txt)
    [--keyword-definitions PATH] default output/effects/keyword-definitions.json
    [--target-size N]            default 5000

python -m effects extract-keyword-definitions
    [--output PATH]              default output/effects/keyword-definitions.json

python -m sealed match-outcomes
    [--effect-records DIR]       no default; the instrumentation opt-in
    [cap/budget flags]           § Collection caps and budgets

python -m effects validate-corpus
    [--effect-records DIR]       default output/effects/records/
    [--limit N]                  default 0 (all); run it on the first minutes of a pass
    [--cards-folder PATH ...]    default output/cardsfolder/ + output/tokenscripts/;
                                 the sidecars the corpus's provenance keys join against
    [--min-keyed-rate F]         default 0.95
    [--max-duplicate-rate F]     default 0.02
    [--max-unpaired-link-rate F] default 0.02
    [--max-names-per-game N]     default 120
    [--min-cost-evidence N]      default 200
    [--turn-jump-tolerance N]    default 1
    [--max-duplicate-event-rate F]        default 0.02 (events inside one record)
    [--max-trigger-fired-share F]         default 0.65
    [--min-zone-change-from-zone-rate F]  unset; measured and watched, not judged
    [--min-attributed-rate F]             unset; measured and watched, not judged

python -m effects collect-coverage
    [--effect-records DIR]       default output/effects/records/
    [--cards-folder PATH ...]    default output/cardsfolder/ + output/tokenscripts/
    [--split-from PATH]          checkpoint whose held-out cards to exclude
    [--target-records N]         default 50
    [--decks-per-round N]        default 500
    [--no-progress-rounds N]     default 3
    [cap/budget flags]           § Collection caps and budgets

python -m effects collect-variants                    (stage four)
    [--effect-records DIR]       default output/effects/records/
    [--forge-cards-path PATH]    default ../forge/forge-gui/res/cardsfolder/
    [--variant-volume F]         default 0.2
    [--decks-per-round N]        default 500
    [cap/budget flags]           § Collection caps and budgets

python -m effects train-effect-model
    [--printings-path PATH]      default resources/AllPrintings.json
    [--split-from PATH]          inherit a checkpoint's split; required for variant runs
    [other flags]                § Training

python -m effects encode-abilities
    [--variant NAME]             default full; resolves --checkpoint and the output suffix
    [--checkpoint PATH]          default: the variant's latest.pt
    [--cards-folder PATH ...]    default output/cardsfolder/ + output/tokenscripts/
    [--variant-scripts PATH]     stage four
    [--vocab-path PATH]          default: the path the checkpoint trained with
    [--keyword-definitions PATH] default: the path the checkpoint trained with
    [--clean]

python -m effects evaluate-effect-model
    [--checkpoint PATH]          default models/effects/effect-model/latest.pt
    [--variant-checkpoint NAME=PATH ...]   repeatable
    [--records-dir DIR]          default output/effects/records/
    [--cards-folder PATH ...]    default output/cardsfolder/ + output/tokenscripts/
    [--variant-scripts PATH]     stage four
    [--vocab-path PATH]          default: the path the checkpoint trained with
    [--keyword-definitions PATH] default: the path the checkpoint trained with
```

Splits are never a flag: `evaluate-effect-model` reads the held-out card list and game ids from `--checkpoint`.
