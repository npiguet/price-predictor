# Feature Specification: Ability effect model

**Feature Branch**: `023-ability-effect-model`
**Created**: 2026-09-06
**Status**: Draft
**Input**: User description: "Ability effect model: pretrained per-ability embeddings and a state-conditional effect head, trained on game-effect records collected from instrumented Forge matches."

Derived from the root spec [`../2026-09-05-ability-effect-model.md`](../2026-09-05-ability-effect-model.md), which is
authoritative for every contract below. Rationale, rejected alternatives, corpus statistics, and the
Forge-source feasibility evidence live in
[`../../experiments/2026-09-04-ability-effect-model-design.md`](../../experiments/2026-09-04-ability-effect-model-design.md).

## User Scenarios & Testing *(mandatory)*

The four stories build on one another. Each widens the corpus without invalidating earlier
records, because the record schema is fixed before any collection happens. Each ends in a re-runnable
evaluation, so every story is independently demonstrable.

### User Story 1 - First embeddings without patching Forge (Priority: P1)

An operator wants to know whether the whole idea produces meaningful ability embeddings before taking
on a permanent maintenance burden. They convert the card corpus (which now also emits a provenance
sidecar per card and converts Forge's token scripts), extract keyword definitions, build the
effects-side vocabulary, then run ordinary sealed self-play with instrumentation switched on. The
workers attach to Forge's public event bus and bracket stack resolution, attributing every event to
the ability being resolved. No Forge source is modified. They train the model on the resulting
records, encode the ability cache, and run the evaluation battery.

**Why this priority**: This is the deliverable in miniature and the only story that needs nothing from
the sibling Forge checkout beyond what stock Forge already exposes. It produces the first `e` vectors,
the shipping gates, and the routing canary that decides whether the probe machinery is worth building at
all. If the embeddings are poor, that is learned here, before any patch exists to maintain.

**Independent Test**: Run `match-outcomes --effect-records` for a short session, train on the shard
directory, run `encode-abilities`, then `evaluate-effect-model`. Delivers a populated embedding cache
plus gate 1, gate 3, and per-keyword gate-2 verdicts — none of which any existing tool produces.

**Acceptance Scenarios**:

1. **Given** a stock (unpatched) sibling Forge checkout, **When** an instrumented worker starts,
   **Then** it detects the absent hooks, runs in degraded attribution, and every record it writes
   carries `mode = degraded`.
2. **Given** `match-outcomes` run with `--effect-records DIR`, **When** the run finishes, **Then** DIR
   holds `{run_id}.{worker}.jsonl` shards of `resolution` and `combat` records, and that command's own
   `match-outcomes.txt` and `cards-played.txt` outputs are unchanged in format and content by the
   flag's presence (their rows carry per-run timestamps, run ids, and durations, so equality is on
   what the flag affects, not byte equality between runs).
3. **Given** `match-outcomes` run without `--effect-records`, **When** the run finishes, **Then** no
   effect records are written anywhere.
4. **Given** a resolution that was countered, **When** its records are read, **Then** a cost record
   exists with `outcome = countered` and no `link_id`, and no effect half exists.
5. **Given** a first-strike combat, **When** its records are read, **Then** two `combat` records
   exist, one per damage step.
6. **Given** a trained checkpoint, **When** `encode-abilities` runs, **Then** `output/effects/abilities/`
   holds one file per converted card and token script, in a subtree named for its source tree, with
   rows aligned one-to-one with that source's sidecar lines.
7. **Given** a trained checkpoint and its `identity` variant, **When** `evaluate-effect-model` runs,
   **Then** it reports gate 1, gate 2 per damage-step keyword, and gate 3 with pass/fail against the
   pinned thresholds.
8. **Given** a corpus that contains no `continuous` records, **When** training runs, **Then** the
   sampling mixture renormalizes over the classes actually present and the absent kinds' fields
   contribute no loss.

---

### User Story 2 - Complete attribution and the corpus match play cannot reach (Priority: P2)

The operator applies the engine patch set to the sibling checkout and re-runs collection. Events now
carry their causing ability directly rather than by bracket, so mana abilities, replacement rewrites,
and sub-ability attribution all become recordable. Three more record kinds start flowing, each from
where the engine already knows the answer: playability from logging points at calls the AI already
makes, continuous effects read out of the per-card layer tables after a recompute, and trigger-fire
from the trigger handler's own condition evaluation. Separately, the operator runs
the coverage collector, which builds decks over the whole converted corpus rather than sealed-legal
sets, so the cards sealed pools can never contain finally reach a game. Reaching a game is the
precondition, not the whole fix: cards Forge still declines to cast fall to the residue that
interventional resolutions pick up.

**Why this priority**: Bracket attribution is exact only up to replacement and static
interactions inside one window, and it leaves the corpus's largest line kind (`static`, 29% of ability
lines) with no records at all. This story supplies the signal for statics, costs, timing, and the
evasion keywords, and it starts closing the coverage hole that sealed self-play cannot reach.

**Independent Test**: Apply the patches, re-run collection, and confirm records appear with
`mode = patched` and with the four new kinds present. Run `collect-coverage` to completion on a small
`--target-records` and confirm the reported residues plus the per-card record counts.

**Acceptance Scenarios**:

1. **Given** a patched Forge checkout, **When** an instrumented worker starts, **Then** it detects the
   hooks and writes `mode = patched` on its records.
2. **Given** Hardened Scales and Doubling Season both applying to one counter-placement event, **When**
   the rewrite records are read, **Then** two records exist, each carrying the incoming event it
   actually received and the outgoing event it produced.
3. **Given** a stable board under an anthem, **When** continuous records are read, **Then** one record
   exists per (game, static, board hash) rather than one per recompute, and the acting static's own
   layer-channel contributions are absent from that record's entity inputs.
4. **Given** a decision point the AI evaluated, **When** its `playability`/`decision` record is read,
   **Then** it holds only rules-level verdicts, and no policy judgment ("another time", "life in
   danger") appears anywhere in the corpus.
5. **Given** `collect-coverage --split-from CHECKPOINT`, **When** decks are built, **Then** no
   held-out card appears in any deck.
6. **Given** a card the castability consult judges uncastable, **When** decks are built, **Then** the
   card still receives deck slots, and at run end it is reported in the uncastable residue.
7. **Given** a card that gains no new qualifying record across `--no-progress-rounds` consecutive
   rounds, **When** the run continues, **Then** that card is retired and the run terminates once every
   card is satisfied or retired.
8. **Given** a completed coverage run, **When** the sealed corpora are inspected, **Then**
   `match-outcomes.txt` and `cards-played.txt` are unchanged by it.

---

### User Story 3 - Counterfactual records for what observation cannot reach (Priority: P3)

Observation only covers abilities Forge chose to use: a sweeper never resolves on an empty board, and
a card the AI declines to cast produces nothing. The operator enables interventional resolutions,
which fork the game and force-resolve a chosen ability with chosen targets. Where gate 2 reported that
a damage-step keyword's embeddings do not separate, the operator also enables that keyword's probe,
which forks at declare-blockers, strips the keyword, and resolves the damage step on the fork.

**Why this priority**: Forks are the only expensive collection mechanism — each costs a game copy that
re-parses every card from its script — and they carry fidelity and process-global-state hazards.
Everything cheaper comes first, and the probe half of this story may never be built at all, because
gate 2 decides per keyword whether it is needed.

**Independent Test**: Run collection with `--interventions-per-game` set and confirm interventional
records appear with the fork's own state, paired to their real counterparts by `mirror_of`. Then run
`evaluate-effect-model` and read the matched real-vs-fork agreement check.

**Acceptance Scenarios**:

1. **Given** an interventional resolution, **When** its record is read, **Then** it carries
   `interventional = true` and `fork = true`, holds the fork's own state, has no `link_id`, and no
   activation record was written for it.
2. **Given** a fork whose copy-score check disagrees with the live game before any perturbation,
   **When** the guard fires, **Then** a warning is logged, no record is written, and the fork still
   counts against the per-game budget.
3. **Given** `--probe-keywords` left at its default, **When** collection runs, **Then** no probe forks
   are taken.
4. **Given** probes enabled, **When** a probe runs, **Then** both branches run under an installed
   seeded random source restored in a `finally`, only one game runs concurrently in that JVM, and the
   fork branch is written as an ordinary `combat` record with `fork = true`.
5. **Given** a real resolution and its forks in one game, **When** the corpus is read, **Then** at most
   two forks name that resolution in `mirror_of`.
6. **Given** a trained checkpoint, **When** `evaluate-effect-model` runs on a corpus containing
   mirrored pairs, **Then** it reports matched real-vs-fork prediction agreement.

---

### User Story 4 - Encoding the mechanism instead of its description (Priority: P4)

The converted prose line is the script's hand-written description, not the mechanism. The operator
rebuilds the vocabulary over the sidecars' script lines, switches the encoder's primary surface to the
script with prose retained as a paired secondary, upgrades keyword definitions from reminder templates
to the implementations Forge generates, and runs `collect-variants` to collect engine ground truth for
perturbed scripts — texts that never existed on a real card.

**Why this priority**: The encoding-surface half changes what the model reads rather than what is
collected, so it depends on no earlier story. It is last because the vocabulary rebuild it forces must
not disturb the prose vocabulary a prose-surface checkpoint records, which is why the script
vocabulary gets a path of its own. `collect-variants` is the exception to the independence: it decks
and schedules variant cards exactly as the coverage collector does, so it reuses the coverage collector's machinery.

**Independent Test**: Run `build-vocab --surface script`, train with the script vocabulary, and confirm
a prose-surface checkpoint still loads and encodes against its own recorded vocabulary path.
Run `collect-variants` and confirm variant records are collected, marked, and excluded from the paired
loss.

**Acceptance Scenarios**:

1. **Given** `build-vocab --surface script`, **When** it runs, **Then** it writes
   `models/effects/vocab-script.txt` and leaves `models/effects/vocab.txt` untouched.
2. **Given** a compound selector `Creature.nonDragon+OppCtrl`, **When** the script tokenizer runs,
   **Then** it yields `Creature`, `nonDragon`, and `OppCtrl` as separate tokens.
3. **Given** a perturbed script, **When** its records are collected, **Then** they carry
   `synthetic = true` and `variant_of`, the variant is never converted to prose, and it contributes no
   paired-encoding loss term.
4. **Given** a card carrying a held-out text, **When** `collect-variants` runs, **Then** no variant
   is generated from it.
5. **Given** a completed `collect-variants` run, **When** the sealed corpora are inspected, **Then**
   `match-outcomes.txt` and `cards-played.txt` are unchanged by it.
6. **Given** a checkpoint trained against the prose vocabulary, **When** any inference command loads it after the
   script vocabulary exists, **Then** it resolves the prose vocabulary the checkpoint recorded and the
   vocabulary hash check passes.

---

### Edge Cases

- A worker JVM crashes mid-write, leaving a partial final line in a shard. Readers tolerate it and
  skip it rather than failing the load.
- A converted line was merged from several runtime traits, or a trait was deduplicated away entirely.
  The sidecar carries several provenance keys for the first case and no line for the second.
- An event attributes to a sub-ability link the sidecar does not map. Attribution falls back to the
  root line rather than being dropped.
- A converted token script and a converted card share a filename (Ajani's Pridemate is both). The two
  trees stay separate, and the provenance key and the cache subtree both carry the source tree.
- The vocabulary is rebuilt between training and encoding. `build-vocab` overwrites its target in
  place, so the embedding table would otherwise be silently re-indexed; the checkpoint's recorded
  vocabulary and keyword-definition hashes catch it and the inference command fails fast. The hashes
  cover those two files only — a reconversion that changes ability lines without changing the
  vocabulary passes the check, so corpus regeneration remains an operator-sequencing concern.
- The corpus grows between a training run and its evaluation, since shards are append-only.
  Evaluation scores only the `game_id`s recorded in the checkpoint's split and ignores the rest.
- A keyword has fewer than 200 qualifying combat records for gate 2. It is routed to the probe exactly
  as a keyword that failed the threshold would be.
- A game's board yields more than four distinct created-object groups. The overflow flag is set rather
  than silently truncating.
- An optional effect the AI declines produces a cost record with `outcome = declined` and no partner,
  which is signal rather than a gap.
- Copy-spell effects (Fork, Reverberate) carry only a copied flag with no original-ability
  back-reference, and resolve through the stack object's source card.
- Batch memory exceeds the 8 GB GPU budget under live context re-encoding. `--context-cache` switches
  to the stop-gradient momentum cache.

## Requirements *(mandatory)*

### Functional Requirements

#### Package layout and boundaries

- **FR-001**: A new Python package `src/effects/` MUST exist, laid out hexagonally
  (`domain` → `application` → `infrastructure`), exposing `python -m effects <subcommand>`.
- **FR-002**: `effects` MUST import only the following, and neither `price_predictor` nor `sealed` may
  import from `effects`:
  - from `price_predictor`: the tokenizer, `tokenizer_store` (the vocabulary file format both
    surfaces are written in and read back from), `build_vocabulary` and its target-size truncation,
    `forge_jvm` worker helpers, `torch_checkpoint`, `torch_training.clip_per_group`, `append_only`,
    `ridge_probes`, and `card_filenames.sanitize_card_name` (a variant script's filename is the path
    its provenance key resolves to, and the Java `CardFilenames` mirrors this one sanitizer — a third
    implementation would break the join without failing the build);
  - from `sealed.domain`: `manabase.compute_basic_lands`, `card_embedding_layout`;
  - from `sealed.infrastructure`: `ConvertedCardLocator`, `embedding_store`,
    `MatchWorkerConnector` (the effects collectors spawn the same Java worker main, so
    rebuilding its system-property mapping here would be a second place for it to drift);
  - from `sealed.application`: nothing. `train-scorer` Phase A is re-run as a subprocess, not
    imported, so the two application layers stay disjoint.
- **FR-003**: Java collectors, the `MatchWorkerMain` instrumentation behind `--effect-records`, and
  `KeywordDefinitionMain` MUST live in `forge-connector`.
- **FR-004**: The engine hooks MUST live as commits on a branch of the sibling `../forge` checkout,
  one commit per hook, carrying the attribution hooks, the missing cause at the `Destroyed` firing
  site, the trigger-fire and playability logging points, the mana-production point, the per-static
  layer accessors and their `…Without` recombinations, and the combat damage assignment. This
  repository MUST NOT carry an exported copy of them: a branch is the history, and a directory of
  patch files beside it is a second copy of that history maintained by hand.
- **FR-004a**: The hooks' inventory — every class, method, and the record channel it feeds — MUST
  live in code, in `PatchHooks.REQUIRED`, and MUST be what hook detection reads rather than a
  parallel list. It is the specification of what a patched Forge has to provide, and what a lapsed
  patch is reconstructed from.
- **FR-005**: Workers MUST detect hook presence at startup and run degraded (bracket-only attribution)
  against stock Forge rather than failing. A worker MUST also report which required hooks are absent,
  because a partly-applied patch still detects as `patched` while a channel stays empty.
- **FR-006**: Model artifacts MUST live under `models/effects/`; corpus and generated files under
  `output/effects/`.

#### Ability identity

- **FR-007**: `python -m price_predictor convert` MUST write a provenance sidecar next to every
  converted card as `<name>.provenance.json`, without altering the converted text.
- **FR-008**: The join key between runtime trait objects and converted lines MUST be printed
  provenance — (script file, face, trait kind, index within that kind's slice of the face's raw trait
  list). Converted-line ordinals MUST NOT be used as a key.
- **FR-009**: Per rendered line the sidecar MUST carry: the provenance key list; the sub-ability links
  the line covers as index paths below the trait; the trait's script API type, parameter-key list, and
  script line as text; and role spans over the prose tagged `cost` | `effect` | `trigger-condition` |
  `target-spec`.
- **FR-010**: Granted abilities MUST resolve through the grantor accessors to the donor card's printed
  line, copied abilities through the original-ability back-reference, and copy-spell effects through
  the stack object's source card.
- **FR-011**: `python -m price_predictor convert` MUST also convert Forge's token scripts from
  `../forge/forge-gui/res/tokenscripts/` into `output/tokenscripts/` with sidecars of their own,
  kept out of `output/cardsfolder/`.
- **FR-012**: A provenance key's script-file component MUST include its script tree (cardsfolder,
  tokenscripts, or variant-scripts).

#### Corpus

- **FR-013**: Records MUST be written as append-only
  `{run_id}.{worker}.jsonl` shards, one JSON record per line, to the directory named by
  `--effect-records` (`output/effects/records/` is the convention and the default everywhere the
  flag has one); readers MUST load every `*.jsonl` in the directory and tolerate a trailing partial
  line.
- **FR-014**: The record schema MUST be fixed before collection begins, so later work
  widen the corpus without invalidating earlier records.
- **FR-015**: Every record MUST carry the envelope fields defined in the root spec: `record_id`,
  `run_id`, `timestamp`, `game_id`, `kind`, `mode`, `interventional`, `fork`, `synthetic`,
  `actor_player`, `state`, and `payload`, plus `moment`, `subkind`, `link_id`, `mirror_of`,
  `variant_of`, and `ability` where each applies.
- **FR-016**: `record_id` MUST be `{run_id}.{worker}.{counter}`, unique across the run's shards, and
  `game_id` MUST be `{run_id}.{worker}.{game counter}`. Both carry the worker index because each
  worker counts independently, and `game_id` is the join key for a checkpoint's recorded split.
- **FR-017**: On a modal resolution, `ability` MUST be the provenance key of the chosen `option`
  line (the converted format's modal-option line type) rather than the parent `spell` line.
- **FR-018**: `mode`, `interventional`, `fork`, and `synthetic` MUST NOT reach the model as inputs.
- **FR-019**: State MUST be stored as data (names plus dynamic attributes), with the tensor
  representation derived at training time, so a representation change never requires re-collection.
- **FR-020**: Snapshot characteristics MUST be computed (post-layer), except that a `continuous`
  record's snapshot has the acting static's own contributions removed from every layer channel it
  wrote.
- **FR-021**: A snapshot MUST carry the block and field structure defined in the root spec's
  § State snapshot table: `global`; `players[]`; `entities[]` across identity, computed
  characteristics, board state, granted abilities, and stack extras; `refs`; and `pending_event`.
- **FR-022**: Snapshots MUST follow the four inclusion tiers in order — (1) referenced objects, each
  carried as an entity in whatever zone it sits with that zone recorded; (2) core: global, battlefield
  entities, and command-zone effect cards (player-scoped continuous effects with no permanent, carried
  as entities with the originating line's key); (3) unreferenced stack contents; (4) unreferenced hand
  and graveyard contents — with tiers 1–2 always present, tier 3 requiring the
  engine patch, and tier 4 what an interventional fork reads. Readers MUST treat an absent tier as uncollected rather than empty.
- **FR-023**: Perspective MUST NOT be stored: controllers are absolute ids and mine/opponent tags are
  derived at training time relative to `actor_player`.
- **FR-024**: Effect payloads MUST carry typed event lists of
  `{type, subjects, params, duration, attributed_to}`, over a vocabulary that is the union of Forge
  trigger types, bus events, and bracket diffs, with the canonical member list in
  `src/effects/domain/event_schema.py`, which also fixes the per-type field normalization.
- **FR-025**: A checked-in completeness test MUST map every Forge effect API class to a covered event
  type or an explicit exclusion.
- **FR-026**: Each record kind MUST carry the per-kind payload defined in the root spec's payload
  table, including one `combat` record per damage step.
- **FR-027**: Recorded playability verdicts MUST be rules-level only; the AI's policy judgments MUST
  NOT be recorded.
- **FR-028**: Collection caps and budgets MUST be exposed as flags on every collecting supervisor with
  the root spec's defaults: `--mana-cap` 2000, `--playability-rate` 0.1,
  `--interventions-per-game` 2, `--probes-per-game` 2, `--probe-keywords` empty (probes disabled).
- **FR-029**: `--playability-rate` MUST sample `decision`-subkind logging points only; `attackers` and
  `blockers` records are always logged. `--mana-cap` MUST count resolution records per unique
  mana-ability text, scoped per worker process. `continuous` records take no cap, because coalescing
  per stable board is itself the cap.

#### Collectors

- **FR-030**: Instrumentation MUST be opt-in per run. `python -m sealed match-outcomes
  --effect-records DIR` forwards the flag to the Java workers, and adding it MUST leave that command's
  own outputs unchanged.
- **FR-031**: `--effect-records` MUST have no default on `match-outcomes`, where it is the
  instrumentation opt-in, and MUST default to `output/effects/records/` on `collect-coverage` and
  `collect-variants`, where collection is the whole point of the command.
- **FR-032**: Collection without the engine patch MUST use only the public event bus plus a bracket around stack
  resolution, attributing events to the resolving ability by bracket, and MUST NOT require a patched
  Forge.
- **FR-033**: The patch MUST add exactly three attribution hooks: the trigger-handler cause channel,
  the shared replacement execution point (with the parameter map deep-copied before the call), and a
  threaded currently-resolving-sub-ability pointer.
- **FR-034**: State-based-action deaths MUST attribute to the bracket they follow, and combat damage
  to its damage-step bracket.
- **FR-035**: Continuous effects MUST be read from the per-card layer tables after a recompute; static
  id 0 entries (temporary pumps) MUST stay attributed to the resolution bracket.
- **FR-036**: Stat changes that fire no trigger MUST be captured by diffing computed stats inside the
  bracket when the stats-changed bus event fires, coalesced per bracket, with trigger-channel events
  taking precedence where both report.
- **FR-037**: Mana records MUST ride the inline-path cast/resolution triggers, exist only under
  `mode = patched`, and honor `--mana-cap`.
- **FR-038**: The trigger-fire hook MUST sit at the trigger handler's condition evaluation, and the playability hooks at the AI's candidate computation, combat-setup legality, and the legality and cost-adjustment checks.
- **FR-039**: The playability logger MUST snapshot defensively — a verdict may be abandoned
  mid-evaluation, and the legality check mutates the checked ability's targets, so the logger MUST NOT
  reuse a checked ability object.
- **FR-040**: Interventional resolutions MUST run Forge's game simulator on a fork, route unaffordable
  candidates through the play-without-paying-mana path, verify the located ability through the
  provenance key, store the fork's own state, write the effect half only with no `link_id`, and cap
  forks targeting one real resolution at two — a fixed constant, independent of
  `--interventions-per-game`. Force-resolving drains the fork's stack, so fork
  records attribute by bracket.
- **FR-041**: Probes MUST fork at declare-blockers after blocks lock, strip one keyword below the
  layer system with a keyword-cache refresh, resolve the damage step, run both branches under a seeded
  random source restored in a `finally`, allow only one concurrent game per JVM, and record the fork
  branch as an ordinary `combat` record with `fork = true` and `interventional = false`, which is what
  distinguishes a probe from an interventional record by flags alone.
- **FR-042**: The real-vs-fork diff MUST be computed only at evaluation time, never as a training
  target.
- **FR-043**: Every fork MUST be score-checked against the live game at creation before any
  perturbation; a mismatch MUST log a warning and discard the fork, which still counts against its
  budget.

#### Depleted collection

- **FR-130**: Training games and validation games MUST come from two `match-outcomes` runs writing
  into the same `--effect-records` directory, differing only in `--exclude-cards`. A depleted run
  opens pools with every held-out card removed; a full-strength run opens them normally and supplies
  the card-disjoint stratum.
  No record field marks which run a shard came from — the split is decided once by `build-corpus`
  (FR-137) and recorded in its manifest; the trainer reads it and derives nothing.
- **FR-131**: `python -m effects holdout-cards --out PATH` MUST write the depletion list: one Forge
  canonical card name per line, every card under `--cards-folder` carrying a held-out text. It MUST
  select texts by the same rule and the same flags as FR-088, and MUST report the card count and the
  share of the corpus it represents.
- **FR-132**: `python -m sealed match-outcomes --exclude-cards PATH` MUST omit the listed cards from
  every pool its workers open, redrawing within the same rarity slot so pool size and rarity
  structure are unchanged. It is the collection path: a match worker opens its own pool per match
  rather than reading a pools file, so the exclusion MUST reach the worker and not a pools
  directory. The flag MUST take a plain newline-delimited name list, so that `sealed` gains no
  import of `effects`.
- **FR-132a**: `python -m sealed generate-pools --exclude-cards PATH` MUST deplete a generated pools
  file the same way, for the consumers that read one (`build-decks`, `pick-decks`). The two paths
  MUST share one redraw implementation; a second copy is the copy that stops matching.
- **FR-132b**: `collect-coverage` and `collect-variants` MUST accept `--training-corpus DIR`,
  equivalent to `--effect-records DIR` with `--exclude-cards DIR/holdout-cards.txt`. A corpus and the
  list that depleted it belong together, so the command MUST refuse a directory holding no
  `holdout-cards.txt` rather than run undepleted, and MUST refuse the flag alongside either flag it
  implies. Coverage counts records per card, and discovery recurses, so the directory MUST be the
  depleted corpus rather than a tree also holding the full-strength one — counting validation records
  toward coverage makes a card whose only records are held out read as satisfied.
- **FR-133**: The full-strength run MUST be sized for the card-disjoint stratum alone and is far
  smaller than the depleted run. `collect-coverage` and `collect-variants` build their own decks
  rather than opening pools, so each takes the depletion list directly (FR-045, FR-057) and neither
  requires a trained checkpoint to know the holdout.
- **FR-134**: A checkpoint MUST record the holdout flags it trained under alongside its split, and
  `--split-from` MUST carry them. Changing the holdout invalidates a depleted corpus, whose games were
  composed against the old list, so the flags are pinned for the life of a corpus.

#### Coverage collector

- **FR-044**: `python -m effects collect-coverage` MUST work in rounds, each rebuilding weighted decks,
  playing `--decks-per-round` (default 500) of those decks as matches, and recounting coverage.
- **FR-045**: It MUST read the held-out card list from `--exclude-cards PATH` — the list
  `holdout-cards` writes — and exclude those cards from every deck, folding case on both sides.
  `--split-from PATH` MUST remain as the alternative, for adding coverage to a corpus an existing
  checkpoint trained on. Passing both MUST be refused rather than merged: the holdout would then have
  two spellings and a disagreement between them would be silent. Neither given means no exclusions.
- **FR-046**: Decks MUST be built over the whole converted card corpus rather than sealed-legal sets,
  drawing candidates from the `output/cardsfolder/` entry of `--cards-folder` alone, weighted toward
  cards with the fewest effect records, as 40-card decks of 23 nonlands plus basics from
  `compute_basic_lands`.
- **FR-047**: The castability consult MUST only rank slots; it MUST NOT drop a card from deck
  building, since being in a game is the precondition an intervention forks from.
- **FR-048**: The coverage unit MUST be drawn from the `output/cardsfolder/` entry of `--cards-folder`
  alone, like the deck candidates, since a token script is not a deckable card and could never be
  satisfied or retired.
- **FR-049**: A card MUST count as satisfied once `--target-records` (default 50) records across every
  shard have it as the acting line's host, an event subject, or a referenced ref; presence in a
  snapshot alone MUST NOT count.
- **FR-050**: A card gaining no new qualifying record across `--no-progress-rounds` (default 3)
  consecutive rounds MUST be retired, so the run always terminates.
- **FR-051**: The run MUST report two residues at the end — cards judged uncastable, and castable
  cards that never reached `--target-records` — each retired card counted under its consult verdict.
- **FR-052**: Coverage matches MUST write effect records only, never `match-outcomes.txt` or
  `cards-played.txt`. A per-run progress file under `--effect-records` is not a sealed corpus and is
  permitted: it is what bounds a round, and nothing downstream reads it.

#### Synthetic script variants

- **FR-053**: `python -m effects collect-variants` MUST perturb one parameter of one script per
  variant, reading source scripts from `--forge-cards-path` (default
  `../forge/forge-gui/res/cardsfolder/`) — a numeric parameter shifted by up to ±3 or doubled, floored at
  zero in either case, or a selector swapped from the checked-in whitelist in
  `src/effects/domain/script_variants.py`.
- **FR-054**: Variant cards MUST be decked and scheduled exactly as coverage decks are, over the
  variant set rather than the converted corpus, with the same rounds and `--decks-per-round`.
- **FR-055**: Perturbed scripts MUST be written to `output/effects/variant-scripts/` and loaded from
  there as custom cards, never entering `output/cardsfolder/` or Forge's own tree.
- **FR-056**: Variants MUST exist on the script surface only: never converted to prose, excluded from
  the paired-encoding loss, with a sidecar written per variant script.
- **FR-057**: Variant records MUST carry `synthetic = true` and `variant_of`. No variant MUST be
  generated from a card carrying a held-out text: a perturbation yields a text that is not itself
  held out, so the variant would reach training and teach the held-out card's mechanics under an edit
  the split cannot detect.
- **FR-058**: `--variant-volume` (default 0.2) MUST cap variant records as a fraction of the real
  records already in `--effect-records`.
- **FR-059**: Variant matches MUST write effect records only, never `match-outcomes.txt` or
  `cards-played.txt`. A per-run progress file under `--effect-records` is not a sealed corpus and is
  permitted: it is what bounds a round, and nothing downstream reads it.

#### Keyword definitions

- **FR-060**: `python -m effects extract-keyword-definitions` (Java `KeywordDefinitionMain`) MUST write
  keyword → reminder-text template for all keywords to `--output` (default
  `output/effects/keyword-definitions.json`), adding the generated implementation
  script captured as text at the keyword factory for the script-generated majority of keywords. The
  engine-coded minority generates no script and keeps its reminder template.

#### Model

- **FR-061**: The ability encoder MUST pool through a `[CLS]` token to the bottleneck `e` (`--e-dim`),
  with additive Gaussian noise (`--e-noise`) during training.
- **FR-062**: Prose MUST be the sole encoding surface until the script vocabulary exists, with the
  script-API classification auxiliary standing in for script structure; thereafter the Forge script line is
  primary and prose the paired secondary under an asymmetric loss with stop-gradient on the script
  side.
- **FR-063**: Training-only heads MUST be an MLM head over masked tokens (`--mlm-weight`,
  `--mlm-mask-prob`) and a script-API classification head from `e` predicting the trait's API type plus
  its parameter-key set (`--api-weight`). The paired-encoding loss MUST run over lines that have
  both surfaces; synthetic variants have only the script surface and contribute no pairing term.
- **FR-064**: `python -m effects build-vocab` MUST write `models/effects/vocab.txt` by default, scan
  converted cards, converted token scripts, and the keyword-definition file, and seed `[PAD]`,
  `[UNK]`, `cardname`, `[MASK]`, `[CLS]`. `--target-size` (default 5000) post-truncates the
  corpus-frequency vocabulary; `--cards-folder` is repeatable.
- **FR-065**: `build-vocab --surface script` MUST add the sidecars' script lines to the scan and write
  `models/effects/vocab-script.txt`, leaving the prose vocabulary untouched.
- **FR-066**: Tokenization MUST be whole-token only with no subword fallback; unknown keywords are
  covered by forced expansion, unknown subtypes and token names by the next vocabulary rebuild.
- **FR-067**: The script tokenizer MUST split compound selectors compositionally.
- **FR-068**: Numbers MUST use a monotone numeric embedding — a shared learned base vector plus
  log1p(n) times a learned direction.
- **FR-069**: Prose tokens MUST add a role embedding looked up from the sidecar's role spans.
- **FR-070**: Keyword-expansion dropout MUST replace a keyword token with its definition at
  probability `--keyword-expand-p`, always expand keywords unknown to the vocabulary, instantiate
  parameterized templates with the instance's own values (a keyword referenced without an instance,
  inside another definition, expands with the template's generic wording), never expand keywords whose body lives on
  the host card, and leave keywords inside an expansion as tokens. On the script surface the definition MUST
  be the captured script on the script surface and the reminder template on the prose surface, falling
  back to the template where no script exists.
- **FR-071**: Each record kind MUST enter the one shared input surface as the input variation defined
  in the root spec's record-kind table (§ Model, "Each record kind is an input variant of this one
  surface"), and supervise the heads that table names — including `[ACT]` left empty for `combat` and
  for the `attackers`/`blockers` subkinds, the anchored attacker carried via the source flag on
  `blockers`, the pending-event overlay on `rewrite` and `trigger`, and the acting static's own
  layer-channel contributions masked out of the entity inputs on `continuous`, structured features
  still included.
- **FR-072**: The effect head input MUST be one token per slot —
  `[GLOBAL] [ACT] [PLAYER]… [CARD] e e … [CARD] e …` — with position ids resetting at each `[CARD]`
  and each slot carrying the contents defined in the root spec.
- **FR-073**: An entity's ability tokens MUST be its printed and attachment-granted lines only;
  temporary grants ride the overlay.
- **FR-074**: Numeric overlay and player scalars MUST enter as raw values plus a log1p copy, with no
  binning.
- **FR-075**: Context ability tokens MUST be dropped at probability `--context-dropout` during
  training.
- **FR-076**: The saved artifact MUST keep the three output heads defined in the root spec's
  § Output heads, and MUST filter out the MLM, script-API, and pairing heads at save time.
- **FR-077**: The per-entity head MUST be one shared head mapped over every `[CARD]` and
  `[PLAYER]` output — an affected/unaffected gate, then conditional field groups:
  - permanents and stack entities: zone outcome (categorical: stayed, died, exiled, to hand, library
    top, library bottom, transformed, face up/down, phased out, blinked, countered), tap state, damage
    taken, counters delta by type, P/T delta with duration, type/color delta with duration, keywords
    gained/lost, control change, attached-to;
  - players: life delta, cards drawn/discarded/milled, library events (scry, surveil, tutor, reveal,
    reorder), mana delta;
  - legality bits: target-legal relative to `[ACT]`; attacker-legal; blocker-legal, with
    `min_blockers` read at the anchored attacker.
- **FR-078**: The created-objects head MUST sit at `[GLOBAL]` with K = 4 group slots, each
  `{present, scripted flag, token-script id or characteristic fields (P/T, type flags, keyword flags),
  count}`, in a canonical order — sorted by token-script id, characteristics-only groups last by
  descending count — with an overflow flag covering more than 4 distinct groups. The token-script id
  MUST be the token's script stem (`c_a_food_sac`), falling back to the printed name only for a token
  whose script cannot be named; a shard collected before 2026-09-17 carries the printed name in that
  field regardless of which token it names.
- **FR-079**: The verdict head MUST sit at `[ACT]`, carrying playability verdict bits (can-play,
  affordable, has-legal-target), predicted cost paid, and the trigger-fired bit.
- **FR-080**: Counts and open-ended magnitudes (damage, counters, life, draws, mana) MUST use
  Poisson-family count regression, signed delta fields MUST decompose into a categorical direction
  (negative, zero, positive) plus a nonnegative magnitude under the count loss, closed vocabularies
  (zone outcome, token-script id) MUST be categorical, and gates and bits MUST be binary. Gate 1's
  deviance MUST be computed over the magnitudes.
- **FR-081**: Loss MUST be normalized per record over entity count; the affected gate trains on every
  entity — players included, since the per-entity head is mapped over `[PLAYER]` outputs too — and
  conditional fields only where the gate's target fires. Count losses MUST use the full Poisson
  negative log-likelihood, Stirling term included, so every loss term is nonnegative at its optimum.
- **FR-082**: The sparse field group — keywords gained/lost, control change, attached-to, the duration
  on every delta that carries one, type/color delta, and counters delta by type — MUST enable at the
  first step of `--curriculum-epoch` (default 3). Every other field trains from step zero, as do all
  three heads' remaining outputs. Validation MUST score the field set the epoch trained with, and the
  early stopper MUST reset at the curriculum boundary.

#### Curated corpus

- **FR-135**: `python -m effects build-corpus` MUST read the raw shard corpus under `--records-dir`
  once and write a curated dataset under `--output` (default `output/effects/corpus/`) holding
  `training/`, `validation/card-disjoint/`, `validation/game-disjoint/`, `validation/gate-one/`
  (resolution records of card-disjoint games whose acting text is held out),
  `validation/samples/card-disjoint.jsonl.gz` and `validation/samples/game-disjoint.jsonl.gz` (the
  trainer's per-epoch validation set, drawing its card-disjoint resolution classes from
  `validation/gate-one/`), and `manifest.json`. Each shard directory MUST hold shards in the corpus
  shard format (FR-027), so every existing reader loads them unchanged, and MUST be repacked so that
  each shard holds **at least** `--shard-records` (default 2000) records and closes at the first game
  boundary thereafter; the final shard of a stratum MAY be shorter, and a game MUST never be split
  across shards. `--shard-records 0` keeps one output shard per source shard.
- **FR-136**: Training records MUST be selected per record and both validation strata per **game**.
  Whole-game selection is what keeps intra-game joins intact: `evaluate-effect-model` resolves a probe
  record's `mirror_of` to a `record_id` in the same stratum, and a stratum holding one half of a pair
  scores nothing and reports the keyword as under-sampled rather than as broken (FR-105).
- **FR-137**: The split MUST be decided once, by the FR-088 rule under the same
  `--holdout-permille` and `--holdout-max-carriers`, against the corpus as it stands at build time. A
  game with a record naming a held-out card MUST go to the card-disjoint stratum;
  `--game-disjoint-games` (default 1000) of the remaining games MUST go to the game-disjoint stratum;
  the rest are training candidates.
- **FR-138**: The training corpus MUST hold at most `--text-cap` (default 200) records per unique
  ability text, dropping the excess at random under `--seed`. The cap MUST be a ceiling and never a
  floor: a text below it keeps every record it has, so the tail `collect-coverage` exists to fill
  survives curation whole.
- **FR-139**: `--class-mix` MUST set the on-disk proportions over the eight sampling classes
  (FR-085), defaulting to the training mixture. A class the raw corpus cannot supply at its share MUST
  be written in full and its shortfall recorded in the manifest, never silently under-filled. What a
  class can supply MUST be measured against the corpus-wide per-text cap rather than per shard, and
  the manifest MUST record the proportions actually delivered beside the requested ones.
- **FR-140**: `--training-records`, when non-zero, MUST subsample the capped result within each class
  to that total. It defaults to zero, meaning no ceiling beyond `--text-cap`.
- **FR-141**: The manifest MUST record effective games per unique ability text counted over the whole
  raw corpus, and the encoding surface those text keys were built on. This is the corpus-wide table
  FR-086 cannot compute; a `--corpus` training run MUST read it rather than counting the resident
  shard, and MUST refuse a `--vocab-path` whose surface is not the manifest's — every lookup would
  miss and weighting would fall back to the resident shard with nothing reported. Coverage and variant shards are built dense in scarce
  texts, so a per-shard count reads those texts as common and down-weights exactly the records those
  collectors were run to obtain.
- **FR-142**: The card-disjoint stratum MUST hold at most `--card-disjoint-text-cap` (default 50)
  games per held-out ability text, so each held-out text weighs comparably in gate 1's per-stratum
  averages (FR-103) and in best-checkpoint selection (FR-092). Games are the selection unit (FR-136),
  so the cap governs which games are admitted rather than which records survive within one: a
  held-out game MUST be admitted only while every held-out text it carries is under the cap, and a
  held-out game carrying no held-out text MUST be dropped rather than trained on. Texts the holdout
  does not name MUST NOT be tallied against the cap.
- **FR-143**: The dataset MUST be a function of the raw corpus, the flags and `--seed` alone. The
  manifest MUST record all three, plus the source shard names and byte sizes, the per-class and
  per-stratum counts, and the unique-ability-text count of each output.
- **FR-144**: A raw corpus that has grown MUST be rebuilt whole rather than extended in place, since
  the split and the rarity table are corpus-wide quantities. A build MUST clear the three output
  directories before writing them, so no shard of an earlier build survives into a dataset the
  manifest does not describe. `--verify` MUST report the difference between an existing manifest's
  source list and the corpus on disk, and write nothing.
- **FR-145**: The run MUST report, per class and per stratum, the records read, the records kept and
  the records the cap dropped, the unique ability texts each output holds, and the delivered class
  proportions against the requested ones. The text counts are what say whether curation preserved the
  tail, which no record count can show.
- **FR-145a**: `build-corpus` MUST refuse to write a dataset gate 1 cannot be scored on, reporting the
  condition and exiting non-zero. An empty holdout MUST be refused before the survey pass, naming each
  `--cards-folder` path tried and whether it existed; an empty card-disjoint stratum MUST be refused
  after it and before anything is written.
- **FR-146**: `--corpus DIR` MUST be the trainer's only input and MUST be required: `train-effect-model`
  reads a curated dataset rather than a raw shard corpus, taking the split, the holdout rule and the
  rarity table from its manifest rather than computing any of them. `--records-dir`, `--reserved-shards`,
  `--holdout-permille` and `--holdout-max-carriers` are no longer flags of the trainer's — each named a
  decision the manifest now carries instead. `--split-from` MAY still be passed on a variant run, but
  only for compatibility with an older invocation: it inherits nothing beyond what the manifest already
  fixes, and the split it would otherwise compute is never read. The curated corpus is still read one
  training shard at a time, never whole.
- **FR-147**: A checkpoint trained with `--corpus` MUST record the manifest's path and digest
  alongside its split (FR-090), and `evaluate-effect-model` MUST fail fast when the dataset it reads
  hashes differently. A rebuilt dataset is a different split, so scoring the gates against it would
  score them partly on games the model trained on.
- **FR-148**: `build-corpus` MUST refuse a record for exactly two reasons — a resolution record
  with no acting ability, and any record carrying more than `--max-events-per-record` events
  (default 64) — in both passes, and MUST record the count per reason in the manifest as
  `quality_dropped`. An event attributed to `unresolved` MUST NOT be a refusal reason. It MUST
  instead be counted: `build-corpus` MUST record in the manifest, as `unattributed_records`, the
  number of records it **kept** carrying at least one such event, and MUST log that count and its
  share of kept records. `attributed_to` names the sub-ability clause that produced an event, and
  `unresolved` means the collector walked the acting chain and no clause claimed it — a failure to
  attribute, not a failure to belong, because the bracket collector records the events that happened
  inside the ability's own resolution. On the full-strength corpus 12.3% of resolution records with
  an acting ability have every event stamped `unresolved` with none exceeding ten events, and
  another 1% carry one unattributed side effect (a state-based-action `zone_change` death, a
  `choice_made`, a `tapped`); a `combat` record has no acting chain at all, so 98.9% of that class
  is stamped `unresolved` by design with the cause in the event's `cause`. Refusing on it discards
  legitimate resolution outcomes, the "died" zone outcome among them, while the whole-game dumps it
  was reaching for are caught by the no-ability and event-flood rules. `build-corpus` MUST also
  refuse to write the dataset when the quality rules refuse more than half of any one sampling
  class's records, naming the class and its counts: a whole class refused means an unpatched
  (`degraded`) checkout or a rule that does not fit that kind, and written silently the dataset is
  simply missing a sampling class.
- **FR-149**: `build-corpus` MUST record, per top-level directory under `--records-dir`, the games
  it read and the games naming a held-out card (`games_by_source`, `held_out_games_by_source`),
  and MUST warn when a directory routes more than none and fewer than 5% of its games.
- **FR-150**: The collector MUST resolve a token host with no `PaperToken` through its image key
  against the token database before falling back to a `cardsfolder` path
  (`ProvenanceKey.tokenScriptStemOf`; verified by `CopiedTokenProvenanceTest`).
- **FR-151**: `build-corpus` MUST remap a provenance key whose script file the converted card tree
  does not hold and whose filename stem names a token script to `tokenscripts/<stem>.txt`, resolving
  the stem by name, then by the carrying entity's colours, types and P/T, then by its printed-key
  counts per trait kind — counting one more `spell` key than the script declares, for the
  permanent's own cast spell — and MUST leave a key untouched when more than one script remains. A
  key under `state.entities` (`printed`, `granted_attached`, `granted_temporary`) MUST resolve per
  entity, against that entity's own characteristics; every other key site (the acting `ability`,
  playability candidates, `responsible_static`, `replaced_by`) MUST be rewritten only when every
  entity in the record that carried the same old key agrees on one stem, and MUST otherwise resolve
  by name alone or stay untouched. It MUST record `token_keys_remapped`, `token_keys_ambiguous` and
  `forge_tokenscripts` in the manifest. `--no-remap-token-keys` turns the step off.

#### Training

- **FR-083**: `python -m effects train-effect-model` MUST train both transformers jointly, end to end,
  from random initialization.
- **FR-084**: Batches MUST mix several games, group each game's records together, and encode each
  unique ability text once; `--context-cache` MUST switch context gradient to a stop-gradient momentum
  cache refreshed every `--cache-refresh` batches. The cache is specified but not implemented: the
  trainer's batcher has none, so `--context-cache` is accepted, warned about at startup and otherwise
  ignored, and context abilities are re-encoded live.
- **FR-085**: The per-batch sampling mixture MUST default to the root spec's eight-class shares and be
  renormalized over the classes present in the corpus. Fields whose record kinds are absent from the
  corpus MUST contribute no loss, so a three-class corpus trains the same heads without them.
- **FR-086**: Within a class, records MUST weight ∝ effective_games^(−0.5), capped at 20× the weight of
  the most-observed ability text, where effective games counts distinct games contributing a record of
  that unique text. These rarity weights MUST apply inside a weighted shuffle without replacement over
  the resident shard rather than a per-batch class draw, since the trainer never holds more than one
  shard at once. Effective games are counted over the resident shard rather than the whole corpus for
  the same reason; a `--corpus` run reads the corpus-wide table from the curated manifest instead
  (FR-141).
- **FR-087**: Records with no acting ability text MUST bypass rarity weighting: `combat` and
  `playability-legality` sample uniformly within their class, and a `decision` record's per-candidate
  examples key on the candidate's text.
- **FR-088**: The card-disjoint validation split MUST hold out ability texts, not cards. A text is
  eligible when at most `--holdout-max-carriers` (default 8) cards under `output/cardsfolder/` carry
  it; an eligible text is held out when `crc32` of its normalized script text modulo 1000 is below
  `--holdout-permille` (default 20). Membership MUST depend on the text's own bytes alone, so adding
  cards never reassigns an existing text. A card is held-out when any of its lines carries a held-out
  text. Training MUST exclude every game with a record naming a held-out card — decided once, by
  `build-corpus`, over the whole raw corpus, rather than computed live by the trainer. Game-disjoint
  validation MUST be the fixed sample `validation/samples/game-disjoint.jsonl.gz`, which `build-corpus`
  draws once from the game-disjoint stratum (FR-135) and the trainer reads whole.
- **FR-088a**: The hash MUST be stable across processes, machines and interpreter versions. The
  salted built-in `hash()` over `str` MUST NOT be used.
- **FR-088b**: `train-effect-model` MUST report, before its first epoch, the held-out text count, the
  share of `output/cardsfolder/` cards those texts remove, and each stratum's record count. It MUST
  fail rather than train when the card-disjoint stratum is empty, and MUST warn when the curated
  manifest's `per_stratum["gate-one"]` count is below 2000 resolution records whose acting text
  appears on no training card. The failure message MUST name the depleted and full-strength
  collection runs (FR-130) as the remedy.
- **FR-088c**: Gate-1 margins MUST additionally be reported split by whether a held-out text's first
  printing falls in the newest sets, so recency is a breakdown of the card-disjoint stratum rather
  than a second holdout.
- **FR-089**: The best checkpoint MUST be selected by card-disjoint validation loss.
- **FR-090**: A checkpoint MUST record the split it trained against — the holdout flags (FR-134), the
  held-out card list they produced, and the `game_id` set across both strata — plus the vocabulary and keyword-definition paths, their content
  hashes, and the keyword withheld from training (or null). The evaluator reads the withheld keyword
  from the checkpoint rather than from a flag, for the same reason it reads the split from there: the
  zero-shot check must measure the model that was trained, not a keyword an operator remembers
  choosing.
- **FR-091**: A variant run MUST see the same games as the full run it baselines. Training both against
  the same `--corpus` is how that requirement is met: the manifest enumerates the split, so two runs
  reading one curated dataset train on the same games by construction (FR-146). `--split-from PATH` is
  a compatibility spelling accepted on a variant run — it inherits nothing, neither the split nor the
  vocabulary nor the keyword-definition paths are read from the path, and the trainer MUST warn at
  startup when it is passed.
- **FR-092**: Inference commands MUST hash the vocabulary and keyword-definition files they actually
  use and fail fast on a mismatch with the checkpoint's recorded hashes.
- **FR-093**: Checkpoints MUST be saved under `--model-output`, defaulting to
  `models/effects/effect-model/` for `--variant full` and `models/effects/effect-model/{variant}/`
  otherwise, as `{timestamp}.pt` plus `latest.pt`, so a variant run never overwrites the shipping
  checkpoint.
- **FR-094**: Four evaluation-baseline variants MUST train on the identical pipeline via `--variant`,
  each defined as: `identity` — a free embedding per unique text replaces the encoder; `state-only` —
  all `e` inputs zeroed; `no-state` — every input zeroed except `[ACT]` and the record-kind flag, with
  the entity ability `e` tokens zeroed too (this is the average-effect control); `taxonomy` — `e`
  replaced by an embedding of the sidecar's API type and parameter keys.
- **FR-095**: The trainer MUST expose the root spec's flag table with its stated defaults, and MUST
  hardcode the stated constants (encoder d_model 256 / 4 layers / 4 heads; effect-head trunk d_model
  256 / 6 layers / 4 heads; `ff_dim` 4 × d_model; dropout 0.1; AdamW; lr 1e-4 constant after warmup;
  warmup over the first 5% of scheduled steps; per-parameter-group gradient clip 1.0; seed 42).
  Per-parameter-group clipping MUST use two groups, `encoder` and `head`, plus `identity` for that
  variant.

#### CLI surface

- **FR-096**: Every command MUST expose the flags and defaults of the root spec's § CLI summary,
  not only the trainer's flag table. In particular: `--cards-folder` is repeatable and defaults to
  `output/cardsfolder/` plus `output/tokenscripts/` on `build-vocab`, `collect-coverage`,
  `encode-abilities`, and `evaluate-effect-model`; `--surface` defaults to `prose`;
  `--probe-keywords` is comma-separated; `--variant-scripts` names the perturbed-script tree; and
  `--records-dir` / `--effect-records` default to `output/effects/records/` everywhere except
  `match-outcomes`.
- **FR-097**: `--vocab-path` and `--keyword-definitions` MUST default, on every inference command,
  to the paths the loaded checkpoint recorded from its training run, and an explicit value MUST
  override that default while still being hash-checked.

#### Embedding cache

- **FR-098**: `python -m effects encode-abilities` MUST write one file per converted card, token
  script, and variant script under `output/effects/abilities/`, in a subtree named for
  its source tree and mirroring that tree's layout, holding a float32 array of shape
  `(n_lines, e_dim)` row-aligned with the source's sidecar — rendered lines for a converted tree, script lines for the
  variant tree.
- **FR-099**: The cache MUST live outside `output/cardsfolder/`, because the sealed pipeline's
  `encode-cards --clean` deletes every `.npz` under that tree.
- **FR-100**: `encode-abilities` MUST be idempotent, and `--clean` MUST remove only files this command
  wrote.
- **FR-101**: `--variant` MUST select checkpoint and output suffix together — `full` reads the shipping
  checkpoint and writes `<name>.npz`; any other variant reads that variant's checkpoint and writes
  `<name>.{variant}.npz` beside it. `--checkpoint` overrides the resolved default.
- **FR-102**: The `taxonomy` variant MUST emit its `e` construction into the same row layout despite
  having no encoder, so every `e`-geometry check reads one file shape.
- **FR-103**: Cache-time keyword handling MUST match inference: known keywords stay tokens, unknown
  keywords are always expanded.
- **FR-104**: The cached vector MUST be the primary surface's `e`, which the checkpoint's recorded
  `--vocab-path` decides.
- **FR-105**: The downstream card representation MUST be a `[CARD]` token carrying structured features
  followed by the card's ability `e` rows, with position ids resetting at each `[CARD]` and faces
  separated by an `[ALTERNATE]` token tagged with the converted `layout:` line's value (the authority
  on the layout vocabulary). Positions continue across faces; only `[CARD]` resets them.

#### Evaluation

- **FR-106**: `python -m effects evaluate-effect-model` MUST run the battery over the trained variants
  and report per record kind and per held-out stratum. The four strata are computed from the
  provenance sidecar's `script_api_type`, `script_param_keys`, and `script_text`, which every converted
  tree carries from the first collected record:
  - **unique-text** — the line's text appears on no training card;
  - **shared-text** — the line's text also appears on a training card;
  - **novel combination** — every sub-ability API type in the line appears in training, but their
    ordered sequence for this line does not;
  - **numeric extrapolation** — the line's (API type, parameter-key set) pair appears in training, and
    at least one of its numeric parameter values falls outside the range observed for that pair in
    training.

  A line may qualify for more than one stratum and is reported under each.
- **FR-107**: Splits MUST come from `--checkpoint`'s recorded split and never be recomputed; the
  command MUST fail fast when a `--variant-checkpoint` records a different split, or different
  vocabulary or keyword-definition hashes, than `--checkpoint`.
- **FR-108**: Every reported check MUST score only the recorded `game_id`s, ignoring games appended
  after the training run.
- **FR-109**: Metrics MUST condition on affected entities, be reported per field and class-balanced
  within each categorical field, and every kind MUST report the `state-only` variant as its floor.
- **FR-110**: The command MUST report the root spec's twelve checks: nearest-neighbor inspection and
  UMAP colored by effect category; the ward canary; zero-shot keyword; role-polarity probe; scaling
  calibration; matched real-vs-fork agreement; the `identity` variant on the game-disjoint split; the
  linear-decodability battery against the sealed encoder at `models/sealed/encoder/latest.pt`; the
  pooled-`e` scorer smoke test; the `taxonomy` comparison; the average-effect control (the `no-state`
  variant); and the probe-diff re-check.
- **FR-111**: Checks whose records do not yet exist MUST be skipped rather than failing: matched
  real-vs-fork agreement and the probe-diff re-check need forks, and the
  role-polarity probe needs mana records and so the engine patch. Every other check runs against any
  corpus.
- **FR-112**: The ward canary MUST compare ward's `e` against a checked-in list of functional-twin
  ability texts in `src/effects/domain/ward_twins.py`, passing when ward is closer in cosine distance
  to each twin than the median of ward's cosine distances to all bare single-keyword `e` vectors.
- **FR-113**: Checks over `e` geometry (gate 3, the decodability battery, the ward canary, the scorer
  smoke test) read the `output/effects/abilities/` caches, so `encode-abilities` MUST have run for
  every variant those checks cover before `evaluate-effect-model` runs.
- **FR-114**: Pooled per-card `e` MUST be the concatenation of the mean and the max over the card's
  ability rows. The linear-decodability battery MUST run that pooled vector through the ridge harness
  of [`experiments/2026-08-28-encoder-preferences.md`](../../experiments/2026-08-28-encoder-preferences.md),
  reported side by side with the sealed encoder on the same feature table.
- **FR-115**: The pooled-`e` scorer smoke test MUST concatenate that pooled `e` with the sealed
  encoder's vector rather than replacing it, write the result into a scratch copy of the cards folder
  — never `output/cardsfolder/` — and re-run `train-scorer` Phase A against it. It is informational
  and gates nothing.
- **FR-116**: The remaining checks MUST be: the role-polarity probe, comparing the predicted
  mana-pool sign for `{R}` in cost position against effect position; scaling calibration, predicted
  sweeper deaths as a function of board size; the `identity` variant on the game-disjoint split as
  the in-distribution memorization ceiling; the average-effect control, comparing the `no-state`
  variant on the decodability battery, the scorer smoke test, the ward canary, and scaling
  calibration; and the probe-diff re-check, gate 2's canary re-run over real-and-fork combat pairs
  joined by `mirror_of`, for probed keywords only.
- **FR-117**: The trainer MUST support withholding one implemented keyword's token from training so
  the zero-shot keyword check can run; that keyword's occurrences are always expanded.
- **FR-118**: Gate 1 (identity baseline) MUST require all three of: ≥ 0.05 absolute affected-gate F1
  improvement, ≥ 0.05 absolute zone-outcome accuracy improvement on affected entities, and ≥ 5%
  relative reduction in mean Poisson deviance over count-valued fields — on the card-disjoint split's
  unique-text stratum over resolution records. It MUST block shipping the model or cache.
- **FR-119**: Gate 2 (damage-step keyword canary) MUST run per keyword over first strike, double
  strike, deathtouch, lifelink, trample, indestructible, wither, and infect; require at least 200
  qualifying records; pass when the affected fields move in the keyword's rules direction in ≥ 70% of
  them; and route a failing or under-sampled keyword to the damage-step probe. It MUST block nothing.
- **FR-120**: A qualifying record for gate 2 MUST be a combat record from the game-disjoint validation
  split in which the keyword's presence changes the damage-step outcome.
- **FR-121**: Gate 2's perturbation MUST be model-side, not a game fork: remove the keyword from the
  carrying participant's model input — its ability token where the keyword is printed or
  attachment-granted, and the overlay's temporarily-granted-keywords channel where it is not. Gate 2
  therefore runs before any probe exists, which is what lets it decide whether one is needed.
- **FR-122**: The gate-2 qualifying predicate, rules direction, and affected fields MUST live as one
  row per keyword in a checked-in table (`src/effects/domain/damage_step_keywords.py`).
- **FR-123**: Gate 3 (collapse canaries) MUST require mean pairwise cosine similarity ≤ 0.5 over 10,000
  random pairs and a top principal component explaining ≤ 30% of total variance, over one vector per
  unique ability text. It MUST block shipping the model or cache.
- **FR-124**: Run results MUST be recorded in the design record's Outcome section, never in the root
  spec.

#### Reading the corpus

- **FR-125 (withdrawn)**: The trainer no longer reserves shards for the game-disjoint stratum or
  sweeps the raw corpus at startup; `--corpus` naming a curated dataset (FR-135) MUST be required on
  every run, and the split, both validation strata and the rarity table MUST come from its manifest
  instead.
- **FR-126**: An epoch MUST read `--shards-per-epoch` (default 256) training shards, dividing
  `--steps-per-epoch` evenly among them, and MUST draw them at random from the whole training shard
  list rather than as a contiguous run of it. The shard list is in path order, so a contiguous block
  holds one collection run's consecutive worker lifetimes and never leaves the family that sorts
  first; an epoch's draw MUST be able to reach every part of the list. The draw MUST be a function of
  `--seed` and the epoch number alone, so an epoch's composition does not depend on how many batches
  the epochs before it planned.
- **FR-126a**: `--seed` MUST seed weight initialization, batch planning and the shard draw. Omitted,
  it MUST be drawn from the OS rather than defaulting to a fixed value, and the drawn value MUST be
  logged at startup and recorded on the checkpoint, so runs differ by default and any one of them
  repeats by passing its seed back.
- **FR-127**: Per-epoch validation MUST run on records captured once from the reserved shards, capped
  per stratum, and MUST reuse those same records every epoch — an early-stopping rule reading a
  freshly drawn sample each epoch would measure which records got drawn rather than whether the model
  improved.
- **FR-127a**: The captured sample MUST follow the **training** sampling mixture, not the proportions
  the stratum holds on disk. `build-corpus` mixes the training stratum and leaves both validation
  strata as collected, and the two differ by more than a factor of five on the largest class, so a
  loss read off the natural proportions measures a different objective than the one training
  descends. A class the stratum cannot supply at its share MUST be reported as short rather than
  silently under-filled.
- **FR-127b**: The shards the sample is drawn from MUST be drawn across the stratum rather than taken
  from the head of its shard list, which is one collection run's. Reading MAY stop as soon as the
  sample is full, and MUST NOT when the split is derived from the shards rather than inherited from a
  manifest — a shard skipped there is a game the checkpoint cannot enumerate.
- **FR-127b-i**: The sweep MUST read shards across `--workers` processes, and a worker MUST return
  the digest of its shards — the games it routed to each stratum, its class histogram, and the
  records its share of the mixture kept — rather than the records it read. Returning whole shards
  would spend more on pickling them than parallelism saves, and the routing is decidable in the
  worker because a game never spans two shards. Threads MUST NOT be used for it: the cost is
  `json.loads` and record construction, both of which hold the GIL.
- **FR-127c**: Validation MUST batch at `--batch-size` over the shuffled sample rather than one game
  per batch. A batch's class composition decides which fields carry loss at all, so a whole-game
  batch scores a different field set than a training batch and the two numbers are not comparable;
  a stratum's loss is also the mean over its batches, and whole-game batches make that a mean over
  as many numbers as the stratum has games. Validation MUST report the loss by field.
- **FR-128**: The trainer MUST log the corpus size before reading anything and MUST log one line per
  shard as it goes, naming the shard, its record counts, and its timings. That line MUST also carry
  the shard's mean training loss, the loss decomposed by field, the pre-clip gradient norm of each
  **module**, the learning rate and the step rate. Reading a loss term back is a device
  synchronization, so exactly one step per shard MUST do it — the shard's last. The norms are per
  module rather than per optimizer group because the optimizer holds one group, and a single number
  covering the encoder and the head together cannot say which produced the gradient; measuring them
  MUST NOT change how gradients are clipped.
- **FR-129**: A provenance key naming a script file with no sidecar under its tree MUST NOT stop a run.
  The ability contributes no text, and the reader MUST count the occurrence per script file so the
  share is visible — an unconfigured *tree* still raises, being a misconfigured run rather than a gap
  in the corpus.

### Key Entities

- **Effect record**: one observed game event or decision point, carrying its envelope (identity, kind,
  collection-mode metadata), a pre-event state snapshot, and a per-kind payload. The unit of training.
- **Provenance key**: (script file, face, trait kind, index within kind) — the stable identity of a
  printed ability line, joining runtime trait objects to converted lines and to cache rows.
- **Provenance sidecar**: the per-card file mapping each rendered converted line to its provenance
  keys, sub-ability links, script API type and text, and prose role spans.
- **State snapshot**: global block, player blocks, and a variable-length entity set with identity,
  computed characteristics, board state, granted abilities, and stack extras, plus refs and any
  pending event. Stored as data, tensorized at training time.
- **Ability embedding `e`**: the fixed-width bottleneck vector for one ability line, cached offline
  per unique text. The feature's primary shipped artifact.
- **Effect head**: the state-conditional model consuming `e` plus a state snapshot and emitting
  per-entity, created-object, and verdict predictions. The feature's second shipped artifact.
- **Keyword definition**: a keyword's reminder-text template and, on the script surface, its captured
  generated implementation script — the expansion target for keyword-expansion dropout.
- **Variant script**: a perturbed copy of a Forge card script, existing on the script surface only,
  used for engine-ground-truth records of texts that never existed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A model trained on the collected corpus beats a per-text free-embedding baseline on
  never-before-seen card text by all three of gate 1's margins, demonstrating that the encoder reads
  text rather than memorizing identities.
- **SC-002**: The embedding cache is not collapsed: mean pairwise cosine similarity stays at or below
  0.5 over 10,000 random pairs, and no single principal component explains more than 30% of variance.
- **SC-003**: Every one of the eight damage-step keywords receives a canary verdict, and each failing
  or under-sampled keyword is routed to a probe, so no gated decision is left to post-hoc judgment.
- **SC-004**: Ward's embedding sits closer to each of its checked-in functional twins than the median
  distance from ward to all bare single-keyword embeddings, showing that behavioral records pull
  together texts whose surfaces barely overlap.
- **SC-005**: An operator running against stock Forge obtains a populated cache and a full set of gate verdicts without
  modifying a single line of Forge source.
- **SC-006**: Every card in the converted corpus is either satisfied at `--target-records` or reported
  in a named residue after a coverage run, so coverage is accounted for rather than estimated.
- **SC-007**: Enabling instrumentation on `match-outcomes` leaves that command's own two output files
  unchanged, and neither the coverage collector nor the variant collector writes to them at all, so
  the sealed corpora cannot be contaminated from any collection path.
- **SC-008**: Training completes within the 8 GB GPU budget, falling back to the stop-gradient context
  cache when live re-encoding does not fit.
- **SC-009**: A checkpoint evaluated after the corpus has grown scores on exactly the games it trained
  against, so no gate is ever computed partly on trained-on data.

## Assumptions

- The four stories are implemented in order. The first is the minimum shippable slice; each later one
  widens the corpus and re-runs the same evaluation.
- The Forge feasibility findings the design record verified against the sibling checkout hold for the
  version this branch builds against (2.0.15-SNAPSHOT). The three hooks are still required, and the
  `Destroyed` firing site still drops its cause.
- The patch set is maintained by hand and re-applied on Forge upgrades. Degraded mode exists so that a
  lapsed patch downgrades attribution quality rather than stopping collection.
- Regenerating the converted corpus (`price_predictor convert`) and re-running the sealed pipeline's
  `encode-cards` are operator steps outside this feature; `effects build-vocab` is inside it, but
  when it runs relative to training is likewise the operator's call. The recorded hashes catch a vocabulary or keyword-definition rebuild between training and
  inference; they do not cover the converted text itself, so a reconversion mid-run is the operator's
  responsibility to avoid.
- Gate 2's 200-record minimum and 70% direction threshold, gate 1's three margins, and gate 3's two
  thresholds are pinned in the root spec and are not re-negotiated after seeing results.
- Consuming `e` in the scorer, picker, or draft agent is out of scope. The pooled-`e` scorer smoke test
  is informational only and gates nothing.
