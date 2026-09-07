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

The four stories are the root spec's four stages. Each widens the corpus without invalidating earlier
records, because the record schema is fixed before any collection happens. Each ends in a re-runnable
evaluation, so every stage is independently demonstrable.

### User Story 1 - First embeddings without patching Forge (Priority: P1)

An operator wants to know whether the whole idea produces meaningful ability embeddings before taking
on a permanent maintenance burden. They convert the card corpus (which now also emits a provenance
sidecar per card and converts Forge's token scripts), extract keyword definitions, build the
effects-side vocabulary, then run ordinary sealed self-play with instrumentation switched on. The
workers attach to Forge's public event bus and bracket stack resolution, attributing every event to
the ability being resolved. No Forge source is modified. They train the model on the resulting
records, encode the ability cache, and run the evaluation battery.

**Why this priority**: This is the deliverable in miniature and the only stage that needs nothing from
the sibling Forge checkout beyond what stock Forge already exposes. It produces the first `e` vectors,
the shipping gates, and the routing canary that decides whether stage three needs to build probes at
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
stage three's interventions pick up.

**Why this priority**: Stage one's bracket attribution is exact only up to replacement and static
interactions inside one window, and it leaves the corpus's largest line kind (`static`, 29% of ability
lines) with no records at all. This stage supplies the signal for statics, costs, timing, and the
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
Everything cheaper comes first, and the probe half of this stage may never be built at all, because
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
collected, so it depends on no earlier stage. It is last because the vocabulary rebuild it forces must
not disturb the prose vocabulary that stage-one-to-three checkpoints record, which is why the script
vocabulary gets a path of its own. `collect-variants` is the exception to the independence: it decks
and schedules variant cards exactly as the coverage collector does, so it reuses stage two.

**Independent Test**: Run `build-vocab --surface script`, train with the script vocabulary, and confirm
the stage-one-to-three checkpoints still load and encode against their own recorded vocabulary path.
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
4. **Given** a variant derived from a held-out card, **When** the split is applied, **Then** the
   variant is held out with it.
5. **Given** a completed `collect-variants` run, **When** the sealed corpora are inspected, **Then**
   `match-outcomes.txt` and `cards-played.txt` are unchanged by it.
6. **Given** a checkpoint trained at stage two, **When** any inference command loads it after the
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
    and `ridge_probes`;
  - from `sealed.domain`: `manabase.compute_basic_lands`, `card_embedding_layout`;
  - from `sealed.infrastructure`: `ConvertedCardLocator`, `embedding_store`,
    `MatchWorkerConnector` (the effects collectors spawn the same Java worker main, so
    rebuilding its system-property mapping here would be a second place for it to drift);
  - from `sealed.application`: nothing. `train-scorer` Phase A is re-run as a subprocess, not
    imported, so the two application layers stay disjoint.
- **FR-003**: Java collectors, the `MatchWorkerMain` instrumentation behind `--effect-records`, and
  `KeywordDefinitionMain` MUST live in `forge-connector`.
- **FR-004**: The engine patch set MUST live under `forge-connector/patches/`, carrying the three
  attribution hooks, the missing cause at the `Destroyed` firing site, and the trigger-fire and
  playability logging points, applied to the sibling `../forge` checkout.
- **FR-005**: Workers MUST detect hook presence at startup and run degraded (bracket-only attribution)
  against stock Forge rather than failing.
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
- **FR-014**: The record schema MUST be fixed before stage one collection begins, so later stages
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
  and graveyard contents — with tiers 1–2 present from stage one, tier 3 from stage two, and tier 4
  from stage three. Readers MUST treat an absent tier as uncollected rather than empty.
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
- **FR-032**: Stage one collection MUST use only the public event bus plus a bracket around stack
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

#### Coverage collector

- **FR-044**: `python -m effects collect-coverage` MUST work in rounds, each rebuilding weighted decks,
  playing `--decks-per-round` (default 500) of those decks as matches, and recounting coverage.
- **FR-045**: It MUST read the held-out card list from `--split-from PATH` when given and exclude
  those cards from every deck.
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
  `cards-played.txt`.

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
- **FR-057**: Variant records MUST carry `synthetic = true` and `variant_of`, and a variant of a
  held-out card MUST be held out with it.
- **FR-058**: `--variant-volume` (default 0.2) MUST cap variant records as a fraction of the real
  records already in `--effect-records`.
- **FR-059**: Variant matches MUST write effect records only, never `match-outcomes.txt` or
  `cards-played.txt`.

#### Keyword definitions

- **FR-060**: `python -m effects extract-keyword-definitions` (Java `KeywordDefinitionMain`) MUST write
  keyword → reminder-text template for all keywords to `--output` (default
  `output/effects/keyword-definitions.json`), adding, from stage four, the generated implementation
  script captured as text at the keyword factory for the script-generated majority of keywords. The
  engine-coded minority generates no script and keeps its reminder template.

#### Model

- **FR-061**: The ability encoder MUST pool through a `[CLS]` token to the bottleneck `e` (`--e-dim`),
  with additive Gaussian noise (`--e-noise`) during training.
- **FR-062**: Prose MUST be the sole encoding surface through stage three, with the script-API
  classification auxiliary standing in for script structure; from stage four the Forge script line is
  primary and prose the paired secondary under an asymmetric loss with stop-gradient on the script
  side.
- **FR-063**: Training-only heads MUST be an MLM head over masked tokens (`--mlm-weight`,
  `--mlm-mask-prob`) and a script-API classification head from `e` predicting the trait's API type plus
  its parameter-key set (`--api-weight`). Stage four adds the paired-encoding loss over lines that have
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
  the host card, and leave keywords inside an expansion as tokens. From stage four the definition MUST
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
  descending count — with an overflow flag covering more than 4 distinct groups.
- **FR-079**: The verdict head MUST sit at `[ACT]`, carrying playability verdict bits (can-play,
  affordable, has-legal-target), predicted cost paid, and the trigger-fired bit.
- **FR-080**: Counts and open-ended magnitudes (damage, counters, life, draws, mana) MUST use
  Poisson-family count regression, signed delta fields MUST decompose into a categorical direction
  (negative, zero, positive) plus a nonnegative magnitude under the count loss, closed vocabularies
  (zone outcome, token-script id) MUST be categorical, and gates and bits MUST be binary. Gate 1's
  deviance MUST be computed over the magnitudes.
- **FR-081**: Loss MUST be normalized per record over entity count; the affected gate trains on every
  entity — players included, since the per-entity head is mapped over `[PLAYER]` outputs too — and
  conditional fields only where the gate's target fires.
- **FR-082**: The sparse field group — keywords gained/lost, control change, attached-to, the duration
  on every delta that carries one, type/color delta, and counters delta by type — MUST enable at
  `--curriculum-step` steps. Every other field trains from step zero, as do all three heads' remaining
  outputs.

#### Training

- **FR-083**: `python -m effects train-effect-model` MUST train both transformers jointly, end to end,
  from random initialization.
- **FR-084**: Batches MUST mix several games, group each game's records together, and encode each
  unique ability text once; `--context-cache` MUST switch context gradient to a stop-gradient momentum
  cache refreshed every `--cache-refresh` batches.
- **FR-085**: The per-batch sampling mixture MUST default to the root spec's eight-class shares and be
  renormalized over the classes present in the corpus. Fields whose record kinds are absent from the
  corpus MUST contribute no loss, so a stage-one corpus trains the same heads without them.
- **FR-086**: Within a class, records MUST weight ∝ effective_games^(−0.5), capped at 20× the weight of
  the most-observed ability text, where effective games counts distinct games contributing a record of
  that unique text.
- **FR-087**: Records with no acting ability text MUST bypass rarity weighting: `combat` and
  `playability-legality` sample uniformly within their class, and a `decision` record's per-candidate
  examples key on the candidate's text.
- **FR-088**: The card-disjoint validation split MUST take cards first printed in the newest sets,
  newest-first, until they cover at least 8% of the cards under `output/cardsfolder/`, and MUST
  exclude from training every game with a record naming a held-out card. Game-disjoint validation
  takes 10% of the remaining games.
- **FR-089**: The best checkpoint MUST be selected by card-disjoint validation loss.
- **FR-090**: A checkpoint MUST record the split it trained against — the held-out card list and the
  `game_id` set across both strata — plus the vocabulary and keyword-definition paths, their content
  hashes, and the keyword withheld from training (or null). The evaluator reads the withheld keyword
  from the checkpoint rather than from a flag, for the same reason it reads the split from there: the
  zero-shot check must measure the model that was trained, not a keyword an operator remembers
  choosing.
- **FR-091**: `--split-from PATH` MUST make a run inherit another checkpoint's split, vocabulary, and
  keyword-definition paths, and every variant run MUST inherit from the `full` run it baselines.
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

#### CLI surface

- **FR-096**: Every command MUST expose the flags and defaults of the root spec's § CLI summary,
  not only the trainer's flag table. In particular: `--cards-folder` is repeatable and defaults to
  `output/cardsfolder/` plus `output/tokenscripts/` on `build-vocab`, `collect-coverage`,
  `encode-abilities`, and `evaluate-effect-model`; `--surface` defaults to `prose`;
  `--probe-keywords` is comma-separated; `--variant-scripts` appears from stage four; and
  `--records-dir` / `--effect-records` default to `output/effects/records/` everywhere except
  `match-outcomes`.
- **FR-097**: `--vocab-path` and `--keyword-definitions` MUST default, on every inference command,
  to the paths the loaded checkpoint recorded from its training run, and an explicit value MUST
  override that default while still being hash-checked.

#### Embedding cache

- **FR-098**: `python -m effects encode-abilities` MUST write one file per converted card, token
  script, and (stage four) variant script under `output/effects/abilities/`, in a subtree named for
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
- **FR-104**: The cached vector MUST be the primary surface's `e` — prose through stage three, script
  from stage four.
- **FR-105**: The downstream card representation MUST be a `[CARD]` token carrying structured features
  followed by the card's ability `e` rows, with position ids resetting at each `[CARD]` and faces
  separated by an `[ALTERNATE]` token tagged with the converted `layout:` line's value (the authority
  on the layout vocabulary). Positions continue across faces; only `[CARD]` resets them.

#### Evaluation

- **FR-106**: `python -m effects evaluate-effect-model` MUST run the battery over the trained variants
  and report per record kind and per held-out stratum. The four strata are computed from the
  provenance sidecar's `script_api_type`, `script_param_keys`, and `script_text`, which every converted
  tree carries from stage one:
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
  real-vs-fork agreement and the probe-diff re-check are available from stage three, and the
  role-polarity probe from stage two (its effect-position half needs mana records). Every other check
  runs from stage one.
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
  them; and route a failing or under-sampled keyword to the stage-three probe. It MUST block nothing.
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
- **Keyword definition**: a keyword's reminder-text template and, from stage four, its captured
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
- **SC-005**: A stage-one operator obtains a populated cache and a full set of gate verdicts without
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

- The four stages are implemented in order. Stage one is the minimum shippable slice; each later stage
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
