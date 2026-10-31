# Feature Specification: Ability effect model — generation 3

**Feature Branch**: `025-ability-effect-model-gen3`
**Created**: 2026-10-31
**Status**: Draft
**Input**: User description: "A new feature based on specs/2026-10-30-ability-effect-model-gen3.md"

Derived from the root spec [`../2026-10-30-ability-effect-model-gen3.md`](../2026-10-30-ability-effect-model-gen3.md)
(the root spec), which is authoritative for every contract below and is cited as "root § N.M".
That root spec amends the base spec [`../2026-09-05-ability-effect-model.md`](../2026-09-05-ability-effect-model.md)
(implemented by [`../023-ability-effect-model/`](../023-ability-effect-model/spec.md)), the gen-2 spec
[`../2026-10-06-ability-effect-model-gen2.md`](../2026-10-06-ability-effect-model-gen2.md) (implemented by
[`../024-ability-effect-model-gen2/`](../024-ability-effect-model-gen2/spec.md)), and the record-schema
contract [`../023-ability-effect-model/contracts/record-schema.md`](../023-ability-effect-model/contracts/record-schema.md).
Everything this feature does not mention stays as features 023 and 024 built it. Where they
disagree, this feature governs.

Rationale, evidence and the run plan live in
[`../../experiments/2026-10-30-effect-model-gen3-design.md`](../../experiments/2026-10-30-effect-model-gen3-design.md)
(the gen-3 record). The rules catalogue lives in [`../../rules/`](../../rules/README.md); its file
formats are specified in `rules/README.md`, and `rules/gen3-fields.json` (rendered as
`rules/GEN3-FIELDS.md`) is the list of record for every new input, output, extension, planned item
and fix this feature names by id (`in.*`, `out.*`, `ext.*`, `planned.*`, `fix.*`). Where that file
disagrees with the root spec, the root spec governs.

Generations:

| gen | what |
|---|---|
| gen-1 | the checkpoint of run `09.17f`, trained under feature 023 |
| gen-2 | the arms of the gen-2 sweep, trained under feature 024 |
| gen-3 | this feature: a corpus collected from scratch under a new record-schema version, indexed and balanced rule by rule, and a 256-wide model with a relation-aware trunk, selected by probes and rule tests |

State on master when this feature starts (the gen-3 record's Result):

- The catalogue's split, triage, scenarios and their verification against the rules text and card
  scripts are done for all 410 units (root § 4.1–§ 4.3). No scenario has been checked against Forge.
- The heads pass (`rules/heads/`), `rules/MODEL-FIELDS.md` and the merged `rules/gen3-fields.json`
  exist (root § 7). The map has not been rerun against master's current targets.
- Gen-2's collector, curation, target and probe defects are fixed on master.
- No detector, directed fork, gen-3 record, gen-3 corpus or gen-3 checkpoint exists.

## Clarifications

### Session 2026-10-31

- Q: Root § 6 says an operation is available once it is built. Which of the seven fork operations
  does this feature build? → A: All seven (FR-011): change power or toughness, add or remove a
  keyword, change a colour or type, add or remove counters, tap or untap, change a life total,
  replace one card with another, the damage-step probe's keyword removal reused. Every classified
  variant then has its operation, and `match-outcomes` refuses only a rule with no detector or no
  classified variant (FR-014).
- Q: Root § 6 says a directed fork "plays on". How far does it play before it stops? → A: To the
  end of the current turn, then the fork is discarded, as interventional forks play the turn on
  (FR-010).
- Q: How does a rule's share divide between its `fires` records and its near misses? → A:
  Equally. Each polarity draws half the rule's share, a short side repeats up to the reuse cap, the
  manifest names a rule left one-sided, and `other` records count toward the rule without a half
  of their own (FR-054a).
- Q: What happens to a rule whose confirmed Forge departure is accepted? → A: The scenario's
  expected outcome is rewritten to Forge's behaviour, and the rule returns to the rule tests,
  scored against Forge's outcome. A patched departure returns the rule unchanged. Only a confirmed
  departure that is neither patched nor accepted excludes the rule (FR-005, FR-071).
- Q: What does a stratum score do when one of its four components has nothing to score? → A: That
  never happens by design: the validation strata are curated so every component of both strata
  has something to score, and curation fails when it cannot. Training does not check for it on
  each run; if it meets an empty component anyway, it fails (FR-068a).

## User Scenarios & Testing *(mandatory)*

The stories follow the gen-3 record's Future plans. Story 1 runs on gen-2's corpus and needs no
new collection, so it starts first and its counts steer stories 3 and 6. Story 2 changes what is
decided at collection time, so it is complete before the first gen-3 game is collected: the record
schema cannot be applied to records afterwards. Story 4 is built and tested before gen-3's first
training run and needs no gen-3 data. Stories 5 to 8 consume the gen-3 corpus.

### User Story 1 - Count the records that show each rule (Priority: P1)

An operator runs `python -m effects rule-counts --records-dir <gen-2 records>`. Every scenario of
the catalogue has a detector, or a declared reason it has none. The command tags each record with
the rules it shows and prints, per rule, how many records show the rule's effect happening and how
many show its near miss, split between random-seat games and games between two Forge AIs. A
record whose outcome contradicts its scenario's expected outcome is listed for review against
Forge, and a confirmed Forge departure is recorded on its scenario.

**Why this priority**: The per-rule counts decide which rules self-play already covers, which need
directed forks or coverage rounds, and whether the random seat needs a larger share. The same
detectors later drive rule budgets, the rule mask and the rule tests, so nothing rule-balanced can
be built before them.

**Independent Test**: Run `rule-counts` on gen-2's records. Check that every scenario's detector
fires on at least one record of its `fires` polarity wherever the corpus holds one, rejects a
hand-built near-miss record, and that the counts file and the printed table agree.

**Acceptance Scenarios**:

1. **Given** gen-2's corpus and scenario `704.5f-1` (a creature with toughness 0 dies), **When**
   `rule-counts` runs, **Then** a resolution record whose effect half moves a 0-toughness creature
   to the graveyard during the state-based check is tagged `704.5f` with polarity `fires`, a
   resolution that leaves a creature at toughness 1 is tagged with polarity `does_not_fire`, and the
   rule's line reports both counts split by random-seat and Forge-AI games.
2. **Given** rule 702.2 (deathtouch), **When** `rule-counts` runs, **Then** combat records with a
   deathtouch creature and the probe forks that strip deathtouch are tagged 702.2 through the combat
   family and the probe keyword.
3. **Given** rule 613 (layer order), whose detector is declared `none`, **When** `rule-counts`
   runs, **Then** the rule is listed with no counts and its declared reason, and no record is tagged
   with it.
4. **Given** a scenario whose detector reads a gen-3-only field (for example `failed_checks`),
   **When** `rule-counts` runs on gen-2 records, **Then** the rule is reported as not countable on
   that generation rather than as zero.
5. **Given** a record its detector tags with a polarity whose observed outcome differs from the
   scenario's expected outcome, **When** `rule-counts` runs, **Then** the record is listed for
   review, by record id and scenario id.
6. **Given** a scenario carrying a confirmed Forge departure, neither patched nor accepted,
   **When** rule tests are frozen (story 8), **Then** that rule has no rule-test records.
7. **Given** a scenario whose confirmed departure is accepted, **When** rule tests are frozen,
   **Then** its expected outcome reads Forge's behaviour and the rule has rule-test records scored
   against it.

---

### User Story 2 - Collect a gen-3 corpus that records what decides each rule (Priority: P2)

An operator rebuilds the fat JAR and runs `match-outcomes --effect-records` (and the coverage and
variant collectors) as before. Every record now carries `generation: 3`. A permanent's card slot
lists every live line, its keyword lines included, each static flagged `active`. The snapshot
carries the new facts of `rules/gen3-fields.json`. Combat records snapshot the board before any
damage. A resolution's held effect half takes every death of the state-based check that follows
it. Playability verdicts carry `timing` and `failed_checks` for both seats at every priority stop.
Turn-based actions get records, attackers records are anchored to one attacker, and the collector
can count every event it drops.

**Why this priority**: Every new input, output and rule-test record depends on fields that exist
only if the collector writes them. The corpus is collected from scratch and cannot be retrofitted.

**Independent Test**: Run a short pilot with `--count-dropped-events`. Run `validate-corpus` and
`field-coverage` over the shards. Check each acceptance scenario below on records of the pilot.

**Acceptance Scenarios**:

1. **Given** a creature with a static ability, a triggered ability and a replacement effect,
   **When** any record snapshots it, **Then** its card slot lists all three lines with their
   provenance keys, beside its spell and activated abilities, and the static carries `active`.
2. **Given** a creature that lost all abilities, **When** it is snapshotted, **Then** it lists no
   lines.
3. **Given** a permanent that gains flying from another permanent's static, **When** it is
   snapshotted, **Then** the granted line is listed on it.
4. **Given** a creature with ward {2} and one with annihilator 2, **When** they are snapshotted,
   **Then** each carries a keyword line built from its own instance's fragment, value included.
5. **Given** a creature with protection from Goblins, **When** it is snapshotted, **Then** its
   protection line names Goblins, read from the instance.
6. **Given** a combat with first strike, **When** the combat record is written, **Then** its
   snapshot shows no damage marked from that combat, and its shape and assignment are filled.
7. **Given** a burn spell that kills a creature, which triggers a dies ability, **When** the check
   ends, **Then** the death sits in the spell's effect half, the trigger opens its own bracket, and
   the trigger's record holds no death of that check.
8. **Given** a sorcery in hand during the opponent's turn, **When** the collector evaluates
   candidates at a priority stop, **Then** the candidate carries `timing = false`, `can_play =
   false`, and a `casting_speed` entry in `failed_checks`, and both players' candidates are
   evaluated, the player without priority included.
9. **Given** a spell whose `Condition$` segment did not hold, **When** it resolves, **Then** the
   resolution payload's `conditions_met` carries that segment's index with `met = false`.
10. **Given** a draw step, **When** the active player draws, **Then** a `turn_action` record with
    subkind `draw` holds the draw event.
11. **Given** two attackers declared, **When** attackers records are written, **Then** each record
    is anchored to one attacker and lists its legal defenders and whether it must attack.
12. **Given** `--count-dropped-events`, **When** the run ends, **Then** the dropped events are
    printed by event type and engine phase, and the pilot report gives the number of
    `mana_produced` events with an empty `mana_by_color`.

---

### User Story 3 - Directed forks fill the rules self-play rarely shows (Priority: P3)

An operator names rules in `--directed-fork-rules`. During live games, at each phase boundary,
the collector runs each named rule's detector against the board. When the board meets the
scenario's base variant, it forks, applies the one-fact change of a contrasting variant through a
fork operation, plays the fork on and records it. The real record and the fork form a minimal
pair from a real game.

**Why this priority**: The per-rule counts of story 1 will show rules self-play rarely or never
reaches. Directed forks are the preferred second source after the corpus itself (root § 5.3).

**Independent Test**: Enable one rule with a built operation, play a few games, and check that
each directed-fork record differs from its real twin by the one fact the operation changed, is
score-checked before the change, and counts against `--directed-forks-per-game`.

**Acceptance Scenarios**:

1. **Given** `--directed-fork-rules 704.5f` and a live board with a 2/2 creature about to be
   dealt 2 damage, **When** the phase boundary is reached, **Then** the collector forks, raises
   the creature's toughness to 3 through "change a creature's power or toughness", plays the fork
   to the end of the turn, writes the fork's records and discards the fork.
2. **Given** a rule in `--directed-fork-rules` whose every contrasting variant is marked
   unclassifiable, **When** `match-outcomes` starts, **Then** it exits 1 naming the rule.
3. **Given** one scenario variant per fork operation, **When** each is enabled in turn, **Then**
   each of the seven operations produces a fork that differs from its real twin by that one fact.
4. **Given** a fork whose copy already disagrees with the live game's score, **When** it is
   checked, **Then** it is discarded and still counts against the per-game budget.
5. **Given** directed-fork records placed in a validation game, **When** rule tests are frozen,
   **Then** none of them is a rule-test record.

---

### User Story 4 - A relation-aware trunk reads typed links between slots (Priority: P4)

The trunk's attention layer takes, beside the slots, a set of (from slot, to slot, type) triples.
Each relation type adds a learned key vector and a learned value vector, per layer and head,
wherever the two slots meet in attention. With both tables at zero the layer computes exactly what
`nn.TransformerEncoderLayer(norm_first=True)` computes.

**Why this priority**: Every gen-3 checkpoint trains with this trunk, and the inputs of story 5
place most links as relations. It needs no gen-3 data, so its tests are written and pass before
the first training run, and its step time and peak memory are measured on a fixed shard.

**Independent Test**: Run the four layer-property tests of root § 10.5, and time a training step
and measure peak memory on a fixed shard against the 8 GB GPU.

**Acceptance Scenarios**:

1. **Given** a layer with both relation tables at zero and weights copied from an
   `nn.TransformerEncoderLayer`, **When** both run on the same input in fp32, **Then** their
   outputs and gradients match.
2. **Given** a permutation of the slots applied to the slots and the relations together, **When**
   the layer runs, **Then** its outputs are permuted the same way.
3. **Given** a single relation from slot i to slot j, **When** the layer runs, **Then** the output
   at i changes against the relation-free run and the output at j does not.
4. **Given** the toy task of reading a feature of a partner among identical-looking slots, **When**
   a small model trains, **Then** it learns with the relation and stays at chance with the tables
   held at zero.
5. **Given** `--relation-values off`, **When** training runs, **Then** the value tables stay zero
   and receive no update.

---

### User Story 5 - The model reads and predicts what decides each rule (Priority: P5)

The model reads gen-3's new inputs: line flags, command-zone slots for floating effects, delayed
triggers and dungeons, library and exiled-card slots, the 33 new inputs of
`rules/gen3-fields.json`, turn, priority and ownership inputs, the revised keyword flags, all at
width 256. It predicts the 22 new outputs, the extended existing ones, legality outputs supervised
on every candidate entity, and the verdict head's `timing` and failed-check columns. After each
input or output change, the rules-to-fields map is rerun and reports the rules the change made
learnable.

**Why this priority**: A record that shows a rule teaches nothing unless the model reads what
decides it and an output is trained on its outcome (root § 7).

**Independent Test**: Encode a gen-3 pilot's records into model inputs and targets; check each new
input and output on a hand-picked record; rerun the map and check its learnable count against the
previous run.

**Acceptance Scenarios**:

1. **Given** an Aura attached to a creature, **When** the record is encoded, **Then** an
   `attached to` relation runs from the Aura's slot to the creature's, and the gen-2 "card is
   attached" feature is gone.
2. **Given** "creatures can't block this turn" resolved earlier, **When** a later record is
   encoded, **Then** a command-zone card slot carries the effect, its ability slots read the `e` of
   the line that created it, an `affects` relation runs to each affected creature, and its duration
   is one of the three duration values.
3. **Given** a decision record whose candidate targets "target creature", **When** its targets are
   built, **Then** `target_legal` is 1 on every creature in `legal_targets` and 0 on every other
   card and player in the record, and the affected gate is not supervised by the record.
4. **Given** a candidate with `timing = false`, **When** its verdict targets are built, **Then** the
   `timing` column is 0 and the `casting_speed` column is 1.
5. **Given** an Ensoul Artifact resolution, **When** targets are built, **Then** `base_pt_delta`
   carries the change of base power and toughness against the previous base.
6. **Given** a game a player lost, **When** its final record is encoded, **Then** `game_lost` is 1
   for that player and 0 for the other.
7. **Given** a gen-3 change to the input layout, **When** the map reruns, **Then** it reports the
   rules newly learnable and writes `rules/heads/` lines for the new layout.

---

### User Story 6 - Curate a corpus balanced rule by rule (Priority: P6)

An operator runs `build-corpus` over the gen-3 records. Within each sampling class, each rule
tagged on the class's records gets a share, half for the records where it fires and half for its
near misses, and the build adds copies of a rule's records until each half is reached, within the
reuse cap and the text cap. Each added copy carries the rule it was drawn for. Turn and step
conditions are balanced met against unmet. Rare protection qualities get examples from
rule-labelled copies, variant scripts and interventional forks. The manifest records per-rule
counts, budgets, shortfalls per polarity, and one-sided conditions and rules.

**Why this priority**: The rule balance is gen-3's goal, and the trainer refuses a corpus without
per-rule counts.

**Independent Test**: Build a corpus from a pilot's records and read the manifest: every tagged
rule has its share, written count and shortfall per class; added copies carry a rule; natural
copies carry none.

**Acceptance Scenarios**:

1. **Given** a class whose records carry 40 tagged rules and a class budget of 400,000, **When**
   `build-corpus` runs, **Then** each rule's share is 10,000 records in that class, and a rule with
   2,000 natural records, 1,000 of each polarity, and a reuse cap of 4 is written up to 8,000 times
   with a shortfall of 2,000.
2. **Given** a record tagged with two rules, **When** an added copy is drawn, **Then** it is drawn
   for the rule further below its share and carries that rule.
3. **Given** a record acting through a line with "activate only during your turn", **When**
   signatures are built, **Then** whether the condition held is part of its outcome signature, met
   and unmet draw equal budgets, and a condition seen on one side only is named in the manifest.
4. **Given** a permanent spell's resolution record, **When** `build-corpus` runs, **Then** the
   record survives, keyed to the permanent's card.
5. **Given** a decision record whose candidate key is unjoinable, **When** `build-corpus` runs,
   **Then** it is refused as `unjoinable-key`.
6. **Given** a records set holding a gen-2 record, **When** `build-corpus` starts, **Then** it
   exits 1.
7. **Given** the class of scenario 1 and a rule with 1,600 `fires` and 400 `does_not_fire` records
   and a reuse cap of 4, **When** `build-corpus` runs, **Then** each polarity's half is 5,000,
   `fires` is written 5,000 times, `does_not_fire` 1,600 times, the shortfall is 3,400, and the
   manifest names the rule as one-sided in that class.

---

### User Story 7 - Train with the rule mask, one loss scale and averaged weights (Priority: P7)

An operator runs `train-effect-model --corpus <gen-3 dataset>`. An added copy trains only the
outputs the map names for its rule (hard mask), or every output with the rest down-weighted (soft
mask). Every field's loss is divided by its constant-prediction loss, floored at
`--deviance-floor`. Counts and amounts use a negative binomial. Validation, stopping and saving
read an exponential moving average of the weights.

**Why this priority**: Without the mask, an added copy drawn for one rule repeats every other
output it carries and distorts them; without one scale, a few large fields dominate the loss.

**Independent Test**: Train a few epochs on a small curated gen-3 corpus under each mask mode and
check the mask, the scaled losses, the averaged weights and resume behaviour on a fixture.

**Acceptance Scenarios**:

1. **Given** `--rule-mask hard` and an added copy drawn for 702.2, **When** its loss is computed,
   **Then** only the outputs the map names for 702.2, plus the affected gate on the rule's own
   entities, contribute.
2. **Given** `--rule-mask soft`, a record with natural share 1 written 4 times, **When** its
   copies' losses are computed, **Then** outputs other than those of its drawn rules weigh 1/4 on
   each copy, and its drawn rules' outputs weigh 1 on every copy.
3. **Given** a field whose constant-prediction loss is 0.01 nats, **When** its loss is scaled,
   **Then** it is divided by 0.04.
4. **Given** an epoch that ends before warmup does, **When** it is validated, **Then** it uses the
   current weights, is not a best candidate and does not count toward `--patience`.
5. **Given** a resume with a different `--rule-mask`, **When** it starts, **Then** it is refused.

---

### User Story 8 - Select checkpoints by probes and rule tests (Priority: P8)

Each epoch, each validation stratum gets a score from four equally weighted components: the
knowledge probes, the three grids, the rule tests and the mean explained deviance over fields. The
control score mixes the two strata by `--stratum-weight`. The run keeps the epoch with the best
control score, and patience counts epochs without a new best. Probe sets and rule tests are split
by game into a selection half and a reporting half. New probes cover turn and step conditions,
anthems, multi-blocker combats, every protection quality, activated "any target" damage, two
ablations and target legality.

**Why this priority**: Gen-2's validation loss was too noisy to stop on and disagreed with the
probes about the best epoch. Gen-3's selection reads what the model is meant to know.

**Independent Test**: Run a short training on a fixture corpus with frozen probe and rule-test
sets; check the per-stratum components printed each epoch, the rescaling, the best-epoch choice
and that the reporting half never enters the control score.

**Acceptance Scenarios**:

1. **Given** probe AUCs, grid errors, rule-test AUCs and explained deviances for both strata,
   **When** the epoch closes, **Then** each component is rescaled (AUC → 2·AUC − 1, R² and
   explained deviance as is, grid → 1 − error ratio), clipped below at −1, the stratum score is
   their mean, and the control score is `W · card-disjoint + (1 − W) · game-disjoint`.
2. **Given** a field whose constant-prediction loss is at or below the floor, **When** the deviance
   component is computed, **Then** the field is left out of the mean.
3. **Given** a rule whose test records in a stratum all have polarity `fires`, **When** the
   stratum's rule-test component is computed, **Then** that rule is left out.
4. **Given** a new best control score, **When** the epoch closes, **Then** encoder, trunk and heads
   are saved together from that epoch and `resume.pt` is copied to `resume-best.pt`.
5. **Given** a probe that selects training records, **When** it is scored, **Then** repeated rows
   of one record are dropped, the probe is reported, and it feeds no stratum score.
6. **Given** the selected checkpoint, **When** the scorecard is written, **Then** probes and rule
   tests are reported per half.
7. **Given** a curated corpus whose game-disjoint selection half holds no rule with test records of
   both polarities, **When** the probe set and rule tests are frozen, **Then** the freeze exits 1
   naming the stratum, the half and the rule-test component.

---

### Edge Cases

- A record shows several rules. It is tagged with each; it counts toward each rule's budget; an
  added copy is drawn for one of them (FR-055).
- A scenario's contrasting variant has polarity `other`. Its records are counted but are not
  rule-test records (FR-072).
- A rule's detector exists but finds no record in either validation stratum. The rule contributes
  nothing to the rule-test component; it is reported as untested.
- A detector reads only gen-3 fields and `rule-counts` runs over gen-2 records. The rule is
  reported as not countable on gen-2 (FR-007).
- A static's condition is reevaluated mid-action. `active` is Forge's answer at snapshot time.
- A face-down permanent. Its card slot carries its real face's lines flagged inactive and face
  state "face down"; nothing of the real face reaches an opponent-facing input other than through
  those inactive lines.
- A trigger reaches the stack during the state-based check and itself causes a death later. That
  later death is outside the check and joins the trigger's own resolution record, not the held
  half.
- A state-based death with no held resolution and no open damage step (for example, a creature
  dying to a -1/-1 counter from an upkeep cost). It reaches no record and the dropped-event count
  counts it.
- A candidate whose failed checks Forge evaluates together (a script restriction bundling phase and
  turn). It reports `script_restriction`, not the individual checks.
- A spell with two targeting clauses. `target_legal` labels the union of the clauses' legal
  targets; per-clause legality is out of scope.
- A permanent with several protection qualities. Each is its own keyword line naming its quality.
- A mutated permanent. Each component card gets a card slot tied by `merged with`, and its lines
  reach the permanent through `line of`.
- An entry of Forge's remembered list that does not outlive its resolution. It is not a relation;
  the rule that separates lasting entries is settled in planning.
- A banding combat. `in band with` exists as a relation type, but no banding output is trained
  (banding is out of scope).
- An added copy whose rule names an output the record does not carry. Under the hard mask that
  copy trains only the affected gate on the rule's entities where the record supervises it, and
  nothing else.
- A resume across the warmup boundary. The averaged weights start at the warmup's end, and
  `resume.pt` carries the step they started from.
- A directed fork rule whose precondition never occurs in a game. No fork is taken; the budget is
  unused.

## Requirements *(mandatory)*

### Functional Requirements

#### Rules catalogue and review against Forge (root § 4)

- **FR-001**: The catalogue's units, triage and scenarios under `rules/` MUST be reused as they
  stand, in the formats `rules/README.md` specifies, read against the 2026-09-25 edition of the
  Comprehensive Rules. Chapters 8 and 9 stay out of scope.
- **FR-002**: Every scenario MUST declare its detector: the record kind and observable outcome the
  detector looks for, and the near miss its contrasting variant gives; or `none`, with a reason,
  where no record field or event shows the outcome. `rules_catalogue.py validate` MUST fail on a
  scenario with neither.
- **FR-003**: Every non-base variant's `delta` MUST be classified into exactly one fork operation of
  root § 6, or marked unclassifiable with a reason. `validate` MUST check the classification's
  enum.
- **FR-004**: A record whose detector-assigned polarity's observed outcome differs from its
  scenario's `expected` MUST be listed for review, by record id and scenario id. The review
  rereads the rule, the card scripts and the record, and corrects the scenario where it was wrong.
- **FR-005**: A disagreement that survives review MUST be recordable on the scenario as a suspected
  Forge departure, and a confirmed one as a confirmed departure, accepted or awaiting a patch on the
  `effect-record-hooks` branch. A rule with a confirmed, unpatched, unaccepted departure MUST have
  no rule-test records (FR-071). Accepting a departure MUST rewrite the scenario's expected outcome
  to Forge's behaviour, keeping the departure recorded on the scenario as accepted; the rule then
  returns to the rule tests, scored against Forge's outcome. A patched departure returns the rule
  with its expected outcome unchanged.

#### Detectors and per-rule counts (root § 5)

- **FR-006**: A detector MUST take one record and return the rules it shows, each with a polarity
  (`fires`, `does_not_fire`, `other`) and the entities the rule's outcome names. The same tags MUST
  feed the per-rule counts, the rule budgets (FR-054) and the rule mask (FR-061).
- **FR-007**: Detectors MUST run over gen-2 and gen-3 records. A detector that needs a field the
  record's generation lacks MUST report the rule as not countable on that generation, never as
  zero.
- **FR-008**: `python -m effects rule-counts --records-dir DIR [--output PATH]` MUST count, per
  rule, the records of each polarity, split between random-seat games and games between two Forge
  AIs, write them to `--output` (default `output/effects/rule-counts.json`) and print them. It
  reads gen-2 and gen-3 records.
- **FR-009**: Records from directed forks and from collection aimed at a rule MUST be identifiable
  as such in every downstream reader, so they never become rule-test records wherever their game is
  placed.

#### Directed forks (root § 6)

- **FR-010**: A directed fork MUST wait for a live board that meets an enabled rule's base variant,
  as the rule's detector judges it on the board, at a phase boundary; fork the game; apply one
  contrasting variant's change through its classified operation; and play the fork on with the
  decisions the game made, to the end of the current turn, recording it. The fork is then
  discarded and the live game continues unchanged.
- **FR-011**: The fork operations are: change a creature's power or toughness; add or remove a
  keyword; change a colour or type; add or remove counters; tap or untap a permanent; change a life
  total; replace one card with another. The existing damage-step probe is the "remove a keyword"
  operation. This feature MUST build all seven operations, the existing one reused, so every
  classified variant has its operation; a variant marked unclassifiable (FR-003) has none.
- **FR-012**: `match-outcomes`, and `collect-variants`, MUST take `--directed-fork-rules LIST`
  (default none) and `--directed-forks-per-game N` (default 2), a budget of its own beside the
  existing fork budgets. A discarded fork still counts against it.
- **FR-013**: Every directed fork MUST be score-checked against the live game before any change, as
  the base spec's forks are.
- **FR-014**: `match-outcomes` MUST exit 1 at startup when a rule in `--directed-fork-rules` has no
  detector, or none of its contrasting variants is classified into an operation.

#### Rules-to-fields map (root § 7)

- **FR-015**: The map MUST give every mechanical and legality rule one output verdict (`trained`,
  `recorded`, `missing`, `planned`) and list the deciding facts the model does not read with the
  gen-3 fields that close each gap, in `rules/heads/<unit>.jsonl` as `rules/README.md` specifies.
- **FR-016**: The map MUST be rerun against master's current targets before the first input or
  output change, and after each change to the input layout or output heads. Each rerun MUST report
  the rules that change made learnable, and `gen3-fields` MUST recompute `learnable_after_gen3`.
- **FR-017**: `rules/MODEL-FIELDS.md` MUST be updated with each layout change so the map reads the
  layout as the code builds it.

#### Collection: schema version and envelope (root § 8.1, § 8.13)

- **FR-018**: Every gen-3 record MUST carry the envelope field `generation` with the value `3`.
  `generation` is collection metadata and never reaches the model. A record without it is gen-1 or
  gen-2, told apart as feature 024 specifies.
- **FR-019**: The frozen-schema rule MUST hold within gen-3 only. A field may change meaning or go
  away relative to gen-2. The schema-compatibility tests MUST be re-pinned to the gen-3 schema, and
  the record-schema contract MUST gain a gen-3 section listing the changes of root § 8.13.
- **FR-020**: `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the knowledge
  probes MUST exit 1 on a records set holding a gen-1 or gen-2 record. `rule-counts` MUST accept
  both.

#### Collection: lines of a permanent (root § 8.2, § 8.3)

- **FR-021**: A permanent's card slot MUST list its live triggers, static abilities and
  replacement effects beside its spell and activated abilities, each with its provenance key; so
  MUST the lines an attachment grants, the lines another permanent's static grants it, spliced
  lines, an emblem's lines, and lines as text-changing effects leave them. The lists MUST be
  Forge's live ones, so a creature that lost its abilities lists none.
- **FR-022**: Each listed static MUST carry `active`, Forge's answer to whether its condition holds
  at snapshot time.
- **FR-023**: Every keyword whose fragment Forge's keyword factory produces MUST get a line in its
  card slot from the instance's own fragment, with its value, whether or not it is printed. A trait
  marked `Secondary$ True` MUST be kept, keyed to the keyword line the card prints so the sidecar
  joins it. A protection line MUST be read from the instance's fragment, which names its quality.

#### Collection: snapshot, links and records (root § 8.4–§ 8.12)

- **FR-024**: The snapshot MUST carry the facts `rules/gen3-fields.json` marks `new`, in the
  shapes it gives: per player speed, city's blessing, rad counters, monarch, initiative and the
  per-turn counts of `in.history`; global day, night or neither; per entity `owner`; per card
  designations, X paid, chosen values, costs paid, history bits, order rank, a face-down card's
  real face, printed loyalty and defense, mana cost symbols; floating effects and pending delayed
  triggers as command-zone objects with their source lines, affected or remembered objects,
  duration and the amount left on a counting-down shield; each player's dungeon and current room;
  each library's top card and every library card the record's events name; the cards exiled with a
  permanent; a mutated permanent's components; the pending event's subject as it last existed; the
  resolving ability's die results and coin flips; and, per static line, the permanents it affects.
- **FR-025**: The snapshot MUST report a creature's set base power and toughness as its base, and
  its boosts apart from it.
- **FR-026**: Each relation type of root § 10.3 MUST be collected from the Forge source its table
  row names. A link made by an ability MUST be recorded from that ability's line, not from its card.
- **FR-027**: A combat record's snapshot MUST be taken at the start of the damage step, before any
  damage is dealt. Its shape MUST be read at the first damage event, and its assignment at any point
  of the step.
- **FR-028**: State-based outcomes MUST join the record of the action that left the board in that
  state, replacing the base spec's bracket attribution. A resolution's held effect half MUST take
  every death while the view tracker is frozen, every pass included. A trigger reaching the stack
  during the check MUST open its own bracket without closing the held half, which is written at the
  first action after the check ends. The trigger's own record MUST hold none of the check's deaths.
  A death in a held half stays marked as no clause's doing. A combat damage step's record holds its
  own check.
- **FR-029**: A state-based death with no held resolution and no open damage step MUST reach no
  record and be counted by the dropped-event count.
- **FR-030**: The AI's playability hook MUST pass each candidate before the AI evaluates it, ahead
  of its `canPlay` filter. The collector's `PlayabilityVerdict` MUST compute `can_play`,
  `affordable` and `has_legal_target` for both seats from the rules alone: costs without the AI's
  cost preferences, each targeting clause's candidates against the clause's minimum, a charm by how
  many offered modes have reachable targets. The random seat MUST filter its own plays on the same
  verdict.
- **FR-031**: `legal_targets` MUST list every card and player the rules allow as a target of the
  candidate. Each candidate MUST carry `timing` (`canCastTiming`); `can_play` MUST be `canPlay` and
  `canCastTiming` together.
- **FR-032**: Each candidate MUST carry `failed_checks`, a list of `{check, responsible_static}`
  over the identifiers `split_second`, `casting_speed`, `whose_turn`, `phase`, `combat`,
  `activation_limit`, `script_condition`, `zone`, `static_prohibition`, `cost`,
  `script_restriction`. A check Forge evaluates apart from the others MUST be reported under its own
  identifier; `script_restriction` only where Forge evaluates several together. A candidate whose
  `timing` is false MUST carry a `casting_speed` entry. `responsible_static` is set where a static
  decided the check; the candidate's own `responsible_static` keeps its gen-2 meaning.
- **FR-033**: The collector MUST evaluate both players' candidates at every priority stop, the
  player without priority included.
- **FR-034**: The resolution payload MUST carry `conditions_met`: for each chain segment with a
  `Condition$`, `{segment, met}`, keyed by the chain index `attributed_to` uses.
- **FR-035**: A prevention that prevented nothing MUST write no note. A probe fork's damage events
  MUST carry their source, recovered from the payload's `assignment_choices`.
- **FR-036**: Turn-based actions MUST get a record of kind `turn_action` with `subkind` one of
  `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`, `lore_counter`, `attack_tap`, its
  payload the action's events.
- **FR-037**: Attackers records MUST be anchored to one attacker, as blockers records are, with
  `anchor_attacker`, `legal_defenders` (the players, planeswalkers and battles it may attack) and
  `must_attack`. Blockers records MUST carry `must_block` per listed creature.
- **FR-038**: `collect-coverage --wording-cards PATH` MUST run a coverage round over the listed
  cards, those with an activated "any target" damage ability.
- **FR-039**: A charm mode cast through Flashback MUST be keyed to a sidecar line. An ability a
  clone gains from the card it copies MUST be keyed to that card's line. A variant record's key
  MUST name its file under `variant-scripts/`.
- **FR-040**: The random seat MUST NOT repeat a free activation whose resolution changes nothing.
- **FR-041**: Under `--count-dropped-events` (`match-outcomes`, `collect-coverage`,
  `collect-variants`; default off), the collector MUST count each event that arrives while no
  record is open, by event type and engine phase, and print the counts when the run ends.
- **FR-042**: `validate-corpus` and `field-coverage` MUST cover every new field of root § 8.13,
  each new number published watched first.

#### Model input (root § 9)

- **FR-043**: The ability encoder, `e` and the trunk MUST each be 256 wide by default.
- **FR-044**: The model input MUST follow root § 9.2–§ 9.6: line flags (active, granted
  temporarily, cast with with a multikicker count, chosen) on every ability slot; a temporarily
  granted keyword as a granted line read through its reminder text, replacing the gen-2 count and
  keyword multi-hot; a face that is not up as ability slots flagged inactive; command-zone card
  slots for floating effects, delayed triggers and dungeons; card slots for the pending event's
  last known subject, the library cards of FR-024, cards exiled with a permanent and a mutated
  permanent's components; the 33 inputs of `rules/gen3-fields.json` in the places root § 9.4 gives;
  the six turn, priority and control inputs of root § 9.5 beside gen-2's acting-player inputs;
  amounts as raw plus log1p, with thermometers where root § 9.4 names them; subtypes as one learned
  embedding per subtype, shared with chosen creature types; face state one-hot over front, back,
  flipped, melded, face down, replacing the face-down flag.
- **FR-044a**: Keyword flags MUST follow root § 9.6: a flag is kept where a selector in Forge's card
  scripts or a check in its engine names the keyword, and for protection; every other scripted
  keyword loses its flag and is read through its line; a keyword with neither flag nor fragment
  expands to its reminder text; `split_second` gains a flag; a keyword counter reads as a granted
  keyword line and every other named counter gets its own scalar input. The kept-flag list MUST be
  derived by a scan of Forge's card scripts and engine code and checked in.
- **FR-045**: A checkpoint MUST record `training_settings["input_layout"] = "gen-3"`, and every
  loader MUST feed the layout it records.

#### Relation-aware trunk (root § 10)

- **FR-046**: Each record MUST carry its relations as a set of (from slot, to slot, type) triples.
  A relation adds no feature to either slot and acts only inside attention, by the score and output
  formulas of root § 10.1. Only relations from i to j enter i's reading of j. Attention dropout MUST
  apply to `a(i, j)` before it multiplies the values, and the relation value term MUST read the same
  dropped-out weights.
- **FR-047**: Each layer and head MUST have its own key table and value table, one head-wide vector
  per relation type, initialized to zero, in the `head` parameter group. Under `--relation-values
  off` the value tables MUST stay zero and untrained.
- **FR-048**: The layer MUST never build a vector per pair of slots: the key term is computed per
  slot and type and looked up per pair; the value term sums attention weights per slot and type and
  multiplies the value table. Attention MUST be computed explicitly, not through a fused kernel that
  does not return the weights.
- **FR-049**: The layer MUST replace `nn.TransformerEncoderLayer` in the trunk, keep gen-2's
  pre-norm, feed-forward block and dropout, and satisfy the four properties of root § 10.5, each
  checked by a test written before the layer.
- **FR-050**: The relation types MUST be those of root § 10.3's table, each name being two types,
  one per direction; `targets` MUST have one type per target index. A card's lines MUST be tied to
  it by `line of`, the lines of one card to each other by `line of the same card`, a charm's modes
  to their root by `mode of`. Cost history MUST be a count on the card, not a relation. Same-name
  relations MUST be built from name matches, with no learned name embedding. A remembered-list entry
  MUST be a relation only when it outlives its resolution.
- **FR-051**: Each gen-2 link feature MUST gain its relation and be kept or replaced as root § 10.4's
  table says.

#### Outputs (root § 11)

- **FR-052**: The model MUST predict the 22 new outputs of root § 11.2 with the slots and shapes
  given there, each supervised only by records that carry it. Per-card and per-player outputs join
  the per-entity head; outputs on the acting ability join the verdict head; a new global head beside
  the created-objects head carries the turn-structure and day/night outputs. The resolution outcome
  MUST be an output only, and the gen-2 resolution-outcome input on the acting-ability slot MUST be
  removed. The replacement result MUST supervise `rewrite` records.
- **FR-053**: The existing outputs MUST be extended as root § 11.3 says: `zone_outcome`'s new
  values, `library_bottom` set, phasing in and out distinct; `library_events` setting `scry`,
  `surveil` and `tutor` only for their own events, plus `shuffle`; each created-objects group's
  other-controller binary, subtypes, counters per type and group kind; `keywords_gained` and
  `keywords_lost` over the closed list `build-corpus` records; a signed-delta output for every named
  counter other than a keyword counter.
- **FR-053a**: Legality outputs MUST follow root § 11.4: `target_legal` supervised on every card and
  player of a decision record (1 inside `legal_targets`, 0 outside), unsupervised for a candidate
  with no targeting clause, the union for several clauses; `blocker_legal` and `attacker_legal`
  supervised on every listed creature; attack target legal on every player, planeswalker and battle
  of an anchored attackers record; the attack/block requirement on the anchor. All train
  independently of the affected gate, and playability records MUST NOT supervise the affected gate.
- **FR-053b**: The verdict head MUST gain a `timing` column and one column per `failed_checks`
  identifier, `script_restriction` included.

#### Curation (root § 12)

- **FR-054**: `build-corpus` MUST take a rule's share within each sampling class as the class budget
  divided by the number of rules tagged on the class's records, and add copies of a rule's records
  beyond their natural share, the copies gen-2's class, family and signature budgets give, until the
  rule's written records reach its share, through the gen-2 water-filling allocator, bounded by the
  reuse cap and `--text-cap`. Added copies come on top of the class budget.
- **FR-054a**: Within a rule's share, records of polarity `fires` and records of polarity
  `does_not_fire` MUST each draw half. A short side repeats up to the reuse cap, and the rest of
  its half stays unfilled as the rule's shortfall; it is not given to the other side. Records of
  polarity `other` count toward the rule's written records and draw no half of their own. Every
  rule whose two sides stay unequal in a class MUST be named in the manifest.
- **FR-055**: A record tagged with several rules MUST count toward each; each added copy MUST be
  drawn for the rule furthest below its share and carry that rule. A copy within the natural share
  MUST carry none. A record with no tag keeps its natural share.
- **FR-056**: For a record whose acting line, candidate or listed static carries a turn or step
  condition, the outcome signature MUST gain whether the condition held, read from `failed_checks`,
  `conditions_met` or `active`; met and unmet MUST draw equal budgets, a short side repeating up to
  the reuse cap; every condition that stays one-sided MUST be named in the manifest.
- **FR-057**: Interventional forks MUST resolve a text whose branch is seen on one player's turn only
  at a phase boundary of the other player's turn, under the existing fork rules and budgets.
- **FR-058**: Rare protection qualities (from a creature type, a card type, a condition, everything)
  MUST get examples from rule-labelled copies (a copy of a real record granting the quality and
  relabelling the outcomes by rule 702.16), from `collect-variants` swapping a protection keyword's
  quality as a whitelisted selector, and from interventional forks granting protection mid-game, in
  amounts per-rule rebalancing sets.
- **FR-059**: The validation strata MUST keep only games between two Forge AIs. A permanent spell's
  resolution record MUST survive `build-corpus`, keyed to the permanent's card. A decision record
  whose candidate's key is unjoinable MUST be refused as `unjoinable-key`.
- **FR-060**: The manifest MUST add the per-rule counts (FR-008's shape), per class and rule the
  records available, share, written and shortfall, each split by polarity, the one-sided conditions
  and one-sided rules (FR-054a), and the closed subtype
  and keyword lists of FR-052/FR-053.

#### Training (root § 13)

- **FR-061**: With n a record's natural share and d its copies written: under `--rule-mask hard`
  (default) an added copy MUST train only the outputs the map names for its drawn rule and the
  affected gate on the rule's own entities where the record supervises the gate, every other output
  masked, while a natural copy trains every output; under `--rule-mask soft` every copy trains every
  output, outputs other than those of the record's drawn rules weighing n / d on each copy. Under
  both, the outputs of every rule an added copy of the record was drawn for MUST train at full
  weight on each copy. A gen-2 lethality copy trains every output at full weight.
- **FR-062**: Each output field's loss MUST be divided by the loss of its constant prediction on the
  same records under the same distribution, clamped below at `--deviance-floor` (default 0.04 nats).
- **FR-063**: Count and amount fields MUST be scored by a negative binomial likelihood in place of
  the Poisson. Gate 1's deviance margin MUST read the negative binomial deviance; its other margins
  are unchanged.
- **FR-064**: Training MUST keep an exponential moving average of the weights over `--ema-epochs`
  (default 1) epochs, each step moving it toward the current weights by 1 / (`--steps-per-epoch` ×
  `--ema-epochs`), starting from the weights at the end of warmup. Validation, stopping and saved
  checkpoints MUST use it. An epoch ending before warmup does is validated on the current weights
  and is neither a best candidate nor counted by `--patience`.
- **FR-065**: `resume.pt` MUST carry the averaged weights and the step they started from. A resume
  MUST refuse a change to `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight` or
  `--relation-values`.
- **FR-066**: `train-effect-model` MUST exit 1 when the corpus manifest carries no per-rule counts.
- **FR-067**: Each training shard's line MUST carry the pre-clip gradient norm of the key tables and
  of the value tables beside the parameter groups' norms, read on the shard's last step; the `head`
  group's norm still includes the tables.

#### Checkpoint selection and evaluation (root § 14)

- **FR-068**: Each epoch, on the averaged weights and from forward passes only, each validation
  stratum MUST get a stratum score: the mean of four components, each rescaled so 0 is its baseline
  and 1 is perfect and clipped below at −1: the knowledge probes at rung 3, each probe type one
  score, averaged; the damage, combat and evasion grids, each scored against the rule's outcome in
  every cell, averaged; the rule tests, each rule counting equally; the mean explained deviance over
  fields, leaving out any field whose constant-prediction loss is at or below `--deviance-floor`.
  Rescaling: AUC as 2 × AUC − 1; R² and explained deviance as is; a grid as one minus the ratio of
  its mean absolute error to the error of predicting the grid's mean outcome in every cell.
- **FR-068a**: Every component MUST have something to score in each half of both validation
  strata: at least one probe type with records, at least one grid with cells, at least one rule
  whose test records hold both polarities, and at least one supervised output field. The strata
  MUST be curated to meet this. `build-corpus`, and the freeze of the probe set and rule tests
  (FR-071), MUST exit 1 when the corpus they produce does not meet it, naming the stratum, half
  and component. Training runs no separate check for it. If an epoch nevertheless meets a
  component with nothing to score, including a deviance component left empty because every field
  falls at or below `--deviance-floor`, `train-effect-model` MUST exit 1 naming it.
- **FR-069**: The control score MUST be the card-disjoint stratum score × `--stratum-weight`
  (default 0.5) plus the game-disjoint stratum score × (1 − it). A run MUST keep one best
  checkpoint, encoder, trunk and heads together, from the epoch with the best control score,
  replacing selection by card-disjoint validation loss. `--patience` (default 5) MUST count epochs
  without a new best control score, and a new best MUST copy `resume.pt` to `resume-best.pt`.
  Calibration stays a reported number.
- **FR-070**: Each probe set and the rule tests MUST be split in two by game: the selection half
  feeds the control score, the reporting half scores the selected checkpoint. Each half of a probe
  set MUST be at least as large as gen-2's whole probe set. A probe that selects training records
  MUST drop repeated rows of one record, is reported, and never feeds a stratum score.
- **FR-071**: A rule's test records MUST come from games in the validation strata, found there by
  its detector. Rule tests MUST be frozen per curated corpus with the knowledge-probe set, with a
  content digest, and exclude every rule with a confirmed Forge departure that is neither patched
  nor accepted (FR-005).
- **FR-072**: Records from probe forks, interventional resolutions, directed forks and collection
  aimed at a rule, and records matching a variant of polarity `other`, MUST NOT be rule-test
  records.
- **FR-073**: For each output the map names for a rule, a rule test MUST read the model's prediction
  at the entity the scenario's outcome names, or at the record for an acting-ability output: the
  probability of the scenario's outcome for a binary or categorical output, the expected value for a
  count or delta. Its score MUST be the AUC ranking `fires` records above `does_not_fire` records,
  averaged over the rule's outputs. A rule whose test records in a stratum hold one polarity only
  MUST be left out of that stratum's component.
- **FR-074**: The knowledge-probe suite MUST gain the probes of root § 14.5: turn and step
  conditions (with casting-speed and split-second subsets), anthem, the combat grid with several
  blockers, the evasion grid's protection rows over every protection quality, the toughness sweep on
  activated damage read with "any target" and "target creature", the board-abilities ablation and
  the ability-slot noise ablation on gen-3 checkpoints, and target legality reading `target_legal`.
  Scorecards MUST add these probes and the rule tests, per half.
- **FR-075**: Each training epoch line MUST print each stratum's score and its four components, and
  the control score.

#### Operator surface (root § 15)

- **FR-076**: The commands and flags of root § 15 MUST exist with the defaults given there; defaults
  live only in the operator surface. Startup validation: every record carries `generation: 3`
  (`build-corpus`, `train-effect-model`, `evaluate-effect-model`, the knowledge probes); the corpus
  manifest carries per-rule counts (`train-effect-model`); every directed-fork rule has a detector
  and a built operation (`match-outcomes`). A failed startup check MUST exit 1. Gates still exit 0
  after any verdict.

#### Records and artifacts (root § 16)

- **FR-077**: A checkpoint MUST add the relation tables, the averaged weights, `--rule-mask`,
  `--deviance-floor`, `--ema-epochs`, `--stratum-weight` and `--relation-values`. The frozen rule
  tests MUST be written with the knowledge-probe set, holding the rule tests' records and their
  halves with a content digest.

#### Boundaries

- **FR-078**: The import-direction test of feature 023 MUST keep passing. The new detector and
  rule-test code lives in `effects`; the catalogue scripts stay under `scripts/`.
- **FR-079**: Everything root § 2 lists as reused unchanged MUST stay as features 023 and 024 built
  it: the collectors, hooks, fork budgets and shard format except where FR-018 to FR-042 change
  them; the gen-2 encoding text, script tokenization and keyword expansion; the masked-template
  holdout and its coverage round; lethality copies; `build-corpus`'s class budgets, allocator, reuse
  cap and repacking; the random seat and its share; the knowledge-probe suite and scorecards beyond
  FR-074.
- **FR-080**: Nothing root § 2 lists as out of scope is built: a controller drawing records for weak
  fields during training, thresholded or focal masks, puzzle-style starting states, a larger or
  softer random seat, combat lethality copies, other rule-labelled copies, layer attribution,
  subgames and restarts, casual mechanics, banding outputs, engine procedure mid-action, derived
  characteristics, per-clause target legality.

### Key Entities

- **Rule**: a numbered rule of the Comprehensive Rules, triaged mechanical or legality, grouped in a
  unit (a three-digit section, or one keyword in 701/702).
- **Scenario**: a minimal pair of boards for one or more rules, with decisions, at least two
  variants each with a polarity, its detector declaration, and any recorded Forge departure.
- **Detector**: a function from one record to the rules it shows, each with polarity and the
  entities the outcome names. Its tags drive counts, budgets, the mask and rule tests.
- **Fork operation**: one of seven one-fact changes a directed fork applies to a live board.
- **Directed fork**: a fork taken when a live board meets a rule's base variant, applying one
  variant's change; its records never become rule-test records.
- **Rule tag**: the rule an added copy was drawn for; absent on natural copies.
- **Natural share / added copy**: the copies gen-2's budgets give a record, and the copies beyond
  them that rule budgets add.
- **Relation**: a directed, typed (from slot, to slot, type) triple acting only inside attention,
  through per-layer, per-head key and value tables.
- **Command-zone slot**: a card slot for a floating effect, delayed trigger or dungeon, read like a
  permanent through the `e` of its source lines.
- **Stratum score / control score**: the per-stratum mean of four rescaled components, and their
  weighted mix that selects the best epoch.
- **Rule test**: per rule, the AUC with which the model's prediction separates `fires` from
  `does_not_fire` records in a validation stratum.
- **Selection half / reporting half**: the game-level split of each probe set and the rule tests.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every scenario in the catalogue declares a detector or a reason for none, and every
  non-base variant is classified into a fork operation or marked unclassifiable; `validate` passes.
- **SC-002**: `rule-counts` over gen-2's corpus reports, for every rule with a detector, its counts
  per polarity split by random-seat and Forge-AI games, or "not countable on gen-2".
- **SC-003**: In a gen-3 pilot, every record carries `generation: 3`, every field of root § 8.13
  appears with a non-constant value in `field-coverage` wherever the pilot's games can produce it,
  and the dropped-event report is printed.
- **SC-004**: In a gen-3 pilot, no trigger record holds a death of the state-based check that its
  trigger reached the stack during.
- **SC-005**: The trunk layer passes all four property tests, and one training step at width 256 on
  a fixed shard fits the 8 GB GPU; its step time and peak memory are reported.
- **SC-006**: After all input and output changes, the map's rerun reports a learnable count, and
  every rule it reports learnable after gen-3 lists an output and no missing input.
- **SC-007**: A gen-3 curated corpus's manifest gives, for every tagged rule in every class, its
  share, written count and shortfall per polarity, and every one-sided condition and rule.
- **SC-008**: A training run selects its checkpoint by control score, and its scorecard reports
  every probe and the rule tests on the reporting half, which never entered selection.
- **SC-009**: A gen-3 checkpoint loads in `encode-abilities`, `evaluate-effect-model` and the
  knowledge probes without restating any flag, and `encode-abilities` writes the gen-3 `e` cache.

## Assumptions

- Story 2 is complete before the first gen-3 collection; the gen-3 corpus is collected from scratch
  and no gen-2 record is migrated into it or mixed with it.
- Detectors are written from the scenario declarations the catalogue holds, in the same
  agent-assisted way as the triage and heads passes, and checked against gen-2's records before
  gen-3 collection.
- The review against Forge (FR-004) starts from the suspected departures listed in
  `rules/FINDINGS.md`.
- Which rules to enable for directed forks, how many coverage rounds to run, the random-seat share,
  and whether to try the soft mask are run-time decisions recorded in the gen-3 record, not
  defaults this feature fixes. The run plan trains with the hard mask first.
- The rule that tells a lasting remembered-list entry from a transient one is settled in planning
  (root § 2).
- Target-index relation types are capped at a fixed number of indices set in planning, recorded in
  the checkpoint.
- Consuming `e` in the scorer, picker, draft agent or game agent is out of scope.
- Run results land in the gen-3 record's Result section, never in this spec or the root spec.
