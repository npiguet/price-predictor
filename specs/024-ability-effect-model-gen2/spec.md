# Feature Specification: Ability effect model — generation 2

**Feature Branch**: `024-ability-effect-model-gen2`
**Created**: 2026-10-06
**Status**: Draft
**Input**: User description: "let's start a new specification from specs/2026-10-06-ability-effect-model-gen2.md"

Derived from the root spec [`../2026-10-06-ability-effect-model-gen2.md`](../2026-10-06-ability-effect-model-gen2.md),
which is authoritative for every contract below. That root spec amends the base spec
[`../2026-09-05-ability-effect-model.md`](../2026-09-05-ability-effect-model.md), implemented by
[`../023-ability-effect-model/`](../023-ability-effect-model/spec.md). Everything this feature does not
mention stays as feature 023 built it. Where the two disagree, this feature governs.

Rationale, evidence and the run order live in
[`../../experiments/2026-09-18-effect-model-gen2-improvements-design.md`](../../experiments/2026-09-18-effect-model-gen2-improvements-design.md)
(the gen-2 record) and
[`../../experiments/2026-09-19-effect-knowledge-probes-design.md`](../../experiments/2026-09-19-effect-knowledge-probes-design.md)
(the probes record). The probes record's family table and its section on assembled labels are the
label specification for FR-079.

Generations:

| gen | what |
|---|---|
| gen-1 | the checkpoint of run `09.17f` (`models/effects/runs/2026-09-17-full-textless-corpus/latest.pt`), trained under feature 023 |
| gen-2 | this feature: the arms of a sweep, all trained on one curated dataset built from a newly collected corpus; arms differ in the width of `e` (`--e-dim`) or the size of the encoder (`--encoder-layers`, `--encoder-d-model`) |

## Clarifications

### Session 2026-10-06

- Q: How does the random seat draw attacks, blocks and targets, whose legal options are combinations? → A: Element by element (FR-022b): each possible attacker attacks with probability ½; each potential blocker picks uniformly from no block plus the attackers it may legally block; a targeted spell or ability draws its target count uniformly between its minimum and maximum, then that many distinct legal targets uniformly; a declaration Forge rejects is redrawn, and after a fixed number of failed draws the seat takes the Forge AI's choice.
- Q: Does `--text-cap` count repeated copies, or only distinct records? → A: Copies (FR-049): within a cell a text supplies at most `min(--reuse-cap × n, --text-cap)` written records, repeats included, so no text is replayed past the cap; a family that stays short goes on the manifest's shortfall list for collection rather than being filled by repeating a few texts.
- Q: A spell's script line carries no `Cost$` (its mana cost sits on the card's `ManaCost` line), so where do the value head's mana-cost targets come from on spell lines? → A: From `Cost$` only, and never from the card's mana cost (FR-058a): the card cost covers the whole card, so no line's encoding text gains it, and spell lines mask their mana-cost targets even when an additional cost puts mana in `Cost$`; the trunk relates the card cost to its abilities. Colours stay unmasked in the holdout template (FR-039).
- Q: How are charms exposed to the model and recorded? → A: As separate abilities (FR-001, FR-005a, FR-063a, FR-029a–f): the root line encodes only its own script line (choices, count, repeatability) and no mode; each `option` line encodes its mode's chain; option rows carry a learned option-kind embedding and follow their root row; a patched worker records each chosen mode, Pawprint and repeated modes included, as its own effect half acting through its `option` line, sharing the root cost record's `link_id`.
- Q: SVar labels are free author names (3,488 distinct in Forge's card scripts, 2,355 used once), so texts differing only in a label get different templates and most labels fall out of the vocabulary. How are they handled? → A: Renamed by position inside `script_text` itself (FR-002a): each line's chain labels become `SV1`, `SV2`, … in order of first appearance, at the reference and at the segment opening alike, so the encoding text, the rarity table and the masked template never see an author's label name.
- Q: Playability payloads carry verdict bits and `responsible_static` keys but no "reason" field, so how is a playability record's rule family derived? → A: From the responsible static's `Mode$` where one is named, otherwise from the first failing verdict bit (FR-047a): a `decision` candidate with no static takes `cannot-play`, `unaffordable` or `no-legal-target`, and `none` only when every bit is true; a legality record takes the sorted set of distinct static modes across `forbidden`, or `none`.
- Q: Is an outcome signature a multiset (counts matter) or a set? → A: A set (FR-049 step 2): the distinct (zone outcome, changed or not) pairs over the affected entities, so mass effects are not split into thin cells by board size.
- Q: Are mana abilities candidates for the random seat's uniform play draw? → A: No (FR-022a): mana abilities are excluded, and a seat whose only playable abilities are mana abilities counts as having nothing playable and passes.
- Q: What happens with `--random-seat-share` above 0 but no `--effect-records`, where a random-seat match would write nothing? → A: `match-outcomes` refuses before any game (FR-026).
- Q: A randomly drawn play may be one the Forge AI declined, so the AI may have no targets, X, modes or cost choices for it. How are its choices made? → A: All at random (FR-022c): every choice made while the play is cast or activated (targets, modes, X, additional-cost choices) is drawn uniformly from the legal options with no `P` gate, and only mana payment stays with the Forge AI; a play whose draws keep failing is abandoned and the play draw repeats over the remaining candidates, passing when none remains.
- Q: The `attackers`, `blockers` and `decision` records are written from hooks inside the Forge AI's decision code, so a random draw that skips the AI writes none. How does a random seat's decision get its records? → A: The `P` draw comes first; on success the random seat writes the records itself, from the legal-option lists it draws from, through the same emitters the hooks call, and stamps its attack and block declarations `what_if = false` explicitly; on failure the Forge AI runs and its hooks fire as for a Forge seat (FR-022d).
- Q: The seat receives priority many times per turn. Is every priority a play decision point with its own `P` draw? → A: Yes (FR-022a): every priority the seat receives is a play decision point with its own `P` draw, and the pilot tunes `P` knowing the per-turn rate of random plays is well above `P`; most priorities have nothing legal to play anyway.
- Q: Which digits does the holdout's masked template mask: standalone numbers only, or also digits inside identifiers (`GE3`, `P1P1`, `w_1_1_soldier`, `Main1`, `SV2`)? → A: Every run of digits wherever it sits, so `GE3` and `GE4` share a template, except in the chain labels FR-002a renames, which keep their numbers (FR-039). The mask keys the holdout only; the encoder reads the unmasked text.
- Q: Underscore is a word character in the prose grammar, so `TokenScript$ w_1_1_soldier` tokenizes to `w_`, `1`, `_`, `1`, `_soldier`. How are token script names tokenized on the script surface? → A: The value of `TokenScript$` also splits on `_`, giving `w 1 1 soldier` (FR-009a); `_` stays a word character everywhere else.
- Q: Should the camel-case split apply to script parameter keys, turning `ConditionCompare$` into `condition compare $`? → A: No (FR-009): a parameter key and its `$` stay one lowercased token (`conditioncompare$`); the camel-case split applies to values only. Keys outside the vocabulary fall to `[UNK]` through the ordinary `--target-size` cut, unseeded.

## User Scenarios & Testing *(mandatory)*

The stories follow the gen-2 record's run plan. Stories 1 to 3 change what is decided at
collection time, so all three are complete before the first gen-2 game is collected. The holdout
and the encoding text key every record, and the random seat and the two envelope fields cannot be
added to a corpus afterwards. Stories 5 and 7 also run on gen-1's artifacts (run plan stage 0), so
they can be built and exercised while stories 1 to 3 are in progress.

### User Story 1 - The encoder reads the whole ability, not its first line (Priority: P1)

An operator re-runs `extract-keyword-definitions`, `convert` and `build-vocab --surface script`.
Each sidecar line's script text now holds the whole script chain the ability owns: a trigger's
executed ability, every sub-ability, a charm's modes, a replacement's replacing ability. The
segments are joined by a separator token, and chain labels are renamed by position (`SV1`,
`SV2`, …) so no author-chosen SVar name reaches the text. Script text is tokenized by the prose grammar, with
camel-case compounds split and chain labels kept whole. Keyword lines are recognised by their
display name and expand to a filled, specifier-free reminder template when expansion is drawn.
Validation, evaluation and the cache leave known keywords as tokens.

**Why this priority**: The holdout key, the rarity table and every record's encoding are computed
from `script_text`. Changing `script_text` after collection invalidates the holdout. This is the
first thing the run plan does, and nothing downstream can start before it.

**Independent Test**: Run `extract-keyword-definitions`, `convert` and `build-vocab --surface
script` on the current Forge checkout. Inspect the sidecars of a trigger card, a sub-ability chain,
a charm and a replacement effect. Read the build log's length distribution, unknown-token rate and
camel-case count. Expand every definition and check that none contains `%` or `[UNK]`.

**Acceptance Scenarios**:

1. **Given** a card whose trigger names `Execute$ TrigDraw` and whose `TrigDraw` names
   `SubAbility$ DBChange`, **When** `convert` runs, **Then** the trigger line's `script_text` is the
   trigger line reading `Execute$ SV1`, then `[SEG]`, then `SV1:` followed by `TrigDraw`'s script
   reading `SubAbility$ SV2`, then `[SEG]`, then `SV2:` followed by `DBChange`'s script, and the
   converted `.txt` is byte-identical to the one written before the change.
2. **Given** a charm line `Choices$ DBExile,DBDraw,DBCounters`, **When** `convert` runs, **Then**
   its `script_text` is its own script line alone, with no mode inlined, and each mode's chain
   sits on that mode's `option` line (scenario 11).
3. **Given** a replacement effect with `ReplaceWith$ DBDraw` whose `DBDraw` has a sub-ability,
   **When** `convert` runs, **Then** the `DBDraw` segment and then its sub-ability's segment
   follow the root.
4. **Given** the script vocabulary is loaded, **When** `ValidTgts$ Creature.nonDragon+YouCtrl` is
   tokenized, **Then** the key yields one token `validtgts$`, `nonDragon` yields `non` and
   `dragon` and `YouCtrl` yields `you` and `ctrl`, while the value of `SubAbility$ SV2` stays one
   token `sv2` and `CounterType$ P1P1` stays one token `p1p1`. **When** `ConditionCompare$ GE3` is
   tokenized, **Then** it yields `conditioncompare$`, `ge` and `3`. **When** `TokenScript$ w_1_1_soldier` is tokenized, **Then** its value yields `w`,
   `1`, `1` and `soldier`, with no token holding `_`.
5. **Given** the prose vocabulary is loaded, **When** the same text is tokenized, **Then** no
   camel-case split is applied.
6. **Given** a keyword line `Ward:2`, **When** it expands, **Then** the whole display name `Ward`
   is replaced by the ward template with `2` filled in Forge's formatting, and the result holds no
   `%` token.
7. **Given** a definition whose template contains `’`, **When** the definitions load, **Then** it
   reads `'`.
8. **Given** `build-vocab` with a small `--target-size`, **When** it builds, **Then** every word of
   every template is still in the vocabulary; **When** a definition would expand to `[UNK]`,
   **Then** the build fails.
9. **Given** a line longer than 512 tokens, **When** any encoding command runs, **Then** it reports
   that line by provenance key as truncated.
10. **Given** `encode-abilities` or a validation pass, **When** a known keyword is met, **Then** it
    stays a token; **When** an unknown keyword is met, **Then** it expands.
11. **Given** the charm of scenario 2, **When** `convert` runs, **Then** each of its three `option`
    lines carries a provenance key and a `script_text` holding its own mode's chain, opened by the
    mode's label renamed `SV1:` and followed by that mode's sub-abilities, and the converted `.txt`
    is unchanged. The root line's `Choices$` reads `SV1,SV2,SV3`.
12. **Given** two cards whose chains are identical except that one names its sub-ability `DBDraw`
    and the other `DBDrawCard`, **When** `convert` runs, **Then** both lines carry the same
    `script_text`.

---

### User Story 2 - Hold out whole templates and cover every held-out text (Priority: P2)

The operator runs `holdout-cards` with the template key. Numbers, `CARDNAME` and description
values are masked out of each chained script text, and rare templates are held out by a hash of the
template. Every card that carries a held-out text is removed from the training pools. After the
full-strength collection, a held-out coverage round builds decks from the held-out cards alone
until every held-out text is recorded in enough distinct games.

**Why this priority**: The holdout decides which pools the training collection may use. Under the
template key, a text that differs from a trained one only in a number no longer counts as unseen.
Like the encoding text, the holdout is fixed before any game is collected.

**Independent Test**: Run `holdout-cards --holdout-unit template` and read its report: held-out
templates, texts, cards and the share of converted cards depleted. Check that two texts differing
only in a number share a template and land on the same side. Run `collect-coverage --only-cards`
for a short session and read its report of texts under the floor.

**Acceptance Scenarios**:

1. **Given** two texts that differ only in `NumDmg$ 2` versus `NumDmg$ 3`, or only in
   `ConditionCompare$ GE3` versus `GE4`, **When** the holdout is computed, **Then** both have the
   same masked template and are both held out or both kept. **Given** two texts whose chains differ
   only in which segment a reference names (`SV1` versus `SV2`), **Then** their templates differ.
2. **Given** a template carried by more cards than `--holdout-max-carriers`, **When** the holdout
   is computed, **Then** it is never held out, whatever its hash.
3. **Given** `holdout-cards`, `build-corpus` and `train-effect-model` all run with
   `--holdout-unit template`, **When** each computes the holdout, **Then** they agree exactly, and
   the manifest and checkpoint record the unit.
4. **Given** a corpus collected under the base spec's rule, **When** `build-corpus --holdout-unit
   text` runs, **Then** it reproduces feature 023's holdout.
5. **Given** `collect-coverage --only-cards PATH`, **When** decks are built, **Then** they contain
   only cards from PATH plus basics, and the records are full strength.
6. **Given** the coverage round in progress, **When** every held-out text with a castable carrier
   has acted in at least `--min-text-games` distinct games or been retired, **Then** the run ends
   and reports the texts left under the floor.
7. **Given** `--only-cards` beside `--exclude-cards`, `--training-corpus` or `--split-from`,
   **When** `collect-coverage` starts, **Then** it refuses before any game.

---

### User Story 3 - Off-policy outcomes and real legality decisions (Priority: P3)

The operator collects with `--random-seat-share 0.125 --random-seat-probability P`. In one match
in eight, one seat sometimes plays a random legal spell or ability, picks random targets, and
declares random attacks and blocks. Its records are marked `random_seat = true`. Legality records
are marked as real decisions or what-if queries. Every real decision is written, and only what-ifs
are sampled. Each chosen mode of a charm is recorded as its own effect half, acting through that
mode's `option` line.

**Why this priority**: Every gen-1 outcome is one Forge chose, so the model never saw what happens
when an ability is aimed at a target Forge would not pick. Neither field, and no per-mode split,
can be added to records afterwards, so all three are in place before collection.

**Independent Test**: Run a pilot of a few hundred games of `match-outcomes --effect-records` with
the random seat on. Run `validate-corpus` and `field-coverage` over it. Check that both new fields
vary, that the random-seat matches added no row to `match-outcomes.txt` or `cards-played.txt`, and
that every real legality decision appears. Find a resolved charm and check that it has one effect
half per chosen mode, each acting through an `option` line.

**Acceptance Scenarios**:

1. **Given** `--random-seat-share 0.125`, **When** many matches run, **Then** close to one match in
   eight has a random seat, its seat chosen at random per match, and its opponent is the standard
   Forge AI.
2. **Given** a random seat at a play, target, attack or block decision, **When** the draw with
   probability `P` succeeds, **Then** it draws from the legal options as FR-022a and FR-022b
   define; otherwise it takes the Forge AI's choice. On a board where nothing restricts attacks,
   each of its creatures that can attack does so in close to half of the random attack draws.
3. **Given** a random seat whose Forge AI would pass priority while a spell is playable, **When**
   the draw with probability `P` succeeds, **Then** the seat plays a playable spell or ability
   picked uniformly; **When** it fails, **Then** the seat passes. A random draw never passes while
   anything is playable, and never activates a mana ability; a seat with only mana abilities
   playable passes.
4. **Given** a random seat, **When** it plays a land, pays mana or mulligans, **Then** the Forge AI
   decides.
5. **Given** a match with a random seat, **When** it finishes, **Then** it has written effect
   records only, and no row to `match-outcomes.txt` or `cards-played.txt`.
6. **Given** a record whose `actor_player` is the random seat, **When** read, **Then**
   `random_seat = true`; every other record carries `random_seat = false`.
7. **Given** an `attackers` record whose snapshot phase is `combat_declare_attackers` and whose
   actor is the active player, **When** read, **Then** `what_if = false`; an `attackers` record
   anywhere else carries `what_if = true`.
8. **Given** a `blockers` record in phase `combat_declare_blockers` whose anchored attacker is
   attacking, **When** read, **Then** `what_if = false`; otherwise `what_if = true`.
9. **Given** `--legality-rate 0.1`, **When** collection runs, **Then** what-if records are sampled
   at that rate and every real decision is written.
10. **Given** two legality records with identical subkind and payload but different snapshots,
   **When** de-duplication runs, **Then** both are kept.
11. **Given** `--random-seat-share` above 0 with no `--random-seat-probability` or no
    `--effect-records`, or either value outside 0 to 1, **When** `match-outcomes` starts, **Then**
    it refuses before any game.
12. **Given** a patched worker and a "choose two" charm cast with modes 1 and 3, **When** it
    resolves, **Then** one cost record acts through the charm's root line, and two effect halves
    act through the `option` lines of modes 1 and 3. Each effect half carries only its own mode's
    events and a snapshot taken just before that mode resolved, and all three records share one
    `link_id`.
13. **Given** a Pawprint charm, or a charm whose modes may repeat, with one mode chosen twice,
    **When** it resolves, **Then** that mode's `option` line acts in two effect halves, one per
    resolution of the mode.
14. **Given** a degraded worker and the same charm, **When** it resolves, **Then** one effect half
    acts through the charm's root line and carries the events of every chosen mode.
15. **Given** a play the random draw picked, **When** it is cast, **Then** its targets, modes, X
    and additional-cost choices are each drawn at random whatever the outcome of a `P` draw, and
    only its mana payment is the Forge AI's. **Given** a randomly drawn spell with X in its cost and
    five mana available, **When** X is drawn, **Then** it falls uniformly between the smallest legal
    X and the largest those five mana can pay. **Given** a randomly drawn play whose draws Forge
    rejects up to the retry limit, **When** the limit is reached, **Then** the play is abandoned and
    another playable candidate is drawn, or the seat passes when none remains.
16. **Given** a random seat at its attack declaration, **When** the `P` draw succeeds, **Then** the
    Forge AI's attack code does not run, and the seat writes an `attackers` record per defender
    from the legal attackers it drew from, stamped `what_if = false` and written whatever
    `--legality-rate` is. **When** the draw fails, **Then** the Forge AI declares, and its hook
    writes the record as for a Forge seat.

---

### User Story 4 - A curated corpus balanced across rule families and outcomes (Priority: P4)

The operator runs `build-corpus` over the full gen-2 collection. Each class budget is split evenly
across rule families, and within a family across outcome signatures. Short families are repeated
up to a cap, and their unused budget passes to the rest. Real legality decisions fill the legality
class first. The game-disjoint stratum is chosen by a hash of each game, with a higher threshold
for games that hold a rare-keyword combat, so a rebuild over a grown corpus keeps every game it
placed before. The manifest reports every shortfall, which the operator sends back to the coverage
and variant collectors before the final rebuild.

**Why this priority**: Every arm trains on this one dataset. Without family balancing, rare rules
train least. Without a hash-stable stratum, a rebuild moves game-disjoint games into training.

**Independent Test**: Run `build-corpus` over the pilot shards into a scratch directory. Read the
manifest's per-family shares, repeats and shortfalls, its records per outcome signature, its on-
and off-policy counts, its real and what-if counts, and its held-out texts under the floor.
Rebuild over grown shards and check that every previously placed game-disjoint game is still
placed.

**Acceptance Scenarios**:

1. **Given** a class with four families and a budget of 4,000, **When** one family's capacity is
   300, **Then** it is written at capacity, its records repeated up to `--reuse-cap` times and no
   text past `--text-cap` copies, and the rest of its share is split evenly across the families
   that still have capacity.
2. **Given** the `resolution-effect` class, **When** a family's share is filled, **Then** it is
   split across outcome signatures the same way.
3. **Given** the `playability-legality` class, **When** it is filled, **Then** real decisions fill
   up to half the budget first, what-if records fill the rest, and family balancing applies within
   each half.
4. **Given** a cell defined by class, family and signature, with `--text-cap 200` and
   `--reuse-cap 4`, **When** a short family's texts hold 10, 150 and 1,000 distinct records,
   **Then** they contribute 40, 200 and 200 written records: the first repeated four times, the
   second's 200 spread over its 150 records, the third a random 200 under `--seed`, unrepeated.
5. **Given** a game naming no held-out card, **When** `crc32(game_id) mod 10⁶ / 10⁶` is below
   `--game-disjoint-share`, **Then** it enters the game-disjoint stratum; **When** it holds a
   qualifying combat for a listed keyword, **Then** the threshold is
   `--game-disjoint-keyword-share` instead.
6. **Given** a build, then a rebuild over more shards, **When** the stratum is computed, **Then**
   every game placed the first time is placed again.
7. **Given** a record whose acting line is a keyword line, **When** its family is assigned,
   **Then** it belongs to that keyword's family whatever its kind.
8. **Given** a display-name match that expands a text whose sidecar line's `script_api_type` is
   not `Keyword`, **When** `build-corpus` runs, **Then** it fails.
9. **Given** the card-disjoint validation sample, **When** its resolution slots are filled,
   **Then** they are filled round-robin over held-out texts, a fixed number per text by smallest
   record hash, until the quota is met.
10. **Given** `--game-disjoint-games` on the command line, **When** `build-corpus` parses its
    arguments, **Then** the flag is rejected: `--game-disjoint-share` replaces it.
11. **Given** a `decision` record with one candidate unaffordable and no responsible static, and
    another forbidden by a `CantBeCast` static, **When** families are assigned, **Then** the first
    candidate's example is in `unaffordable` and the second's in `CantBeCast`.

---

### User Story 5 - Training that reaches the tail and keeps amounts in `e` (Priority: P5)

The operator trains each sweep arm on the curated dataset. The rarity weight's ceiling is set
against the 99th-percentile text. Each epoch line shows what share of trained records came from
each rarity bucket and family. Noise on `e` is scaled to the spread of the `e` vectors themselves
and ramps in over the first epoch. A training-only value head reads amounts and costs from `e`.
The encoder's depth and width are flags recorded in the checkpoint.

**Why this priority**: These are the training changes the sweep compares. The noise ratio is fixed
by a pilot on the gen-1 corpus (stage 0) before the sweep starts.

**Independent Test**: Run three short noise-pilot trainings on the gen-1 corpus at
`--e-noise 0.05`, `0.1` and `0.2` with the value head on, and read the epoch lines. Then train one
arm at a non-default `--encoder-layers` and `--encoder-d-model`, and load the checkpoint with
`encode-abilities` and `evaluate-effect-model` without passing those flags again.

**Acceptance Scenarios**:

1. **Given** the record weights in a class, **When** computed, **Then** each is proportional to
   effective_games^(−0.5) and none exceeds 20 times the weight of the text at the 99th percentile
   of effective games.
2. **Given** an epoch finishes, **When** its line is logged, **Then** it reports the share of
   trained records per rarity bucket (1 game, 2–4, 5–19, 20 or more) and per rule family.
3. **Given** training, **When** a batch carries `e` vectors from any variant, **Then** each gets
   Gaussian noise with covariance `r² · Σ`. `Σ` is the running covariance, seeded from the first
   batch and updated with decay 0.99, with no gradient through it. `r` rises linearly from 0 to
   `--e-noise` over the first `--steps-per-epoch` steps.
4. **Given** validation, evaluation or encoding, **When** `e` is computed, **Then** no noise is
   added.
5. **Given** a line whose script states `NumDmg$ 3` in one segment and `NumDmg$ 2` in another,
   **When** the value head's target is built, **Then** the damage target is 5; **When** a script
   states a variable amount (`NumDmg$ X`), **Then** that target is masked.
6. **Given** a trained checkpoint, **When** it is saved, **Then** the value head is filtered out
   like the MLM and script-API heads, and the checkpoint records the holdout unit,
   `--encoder-layers`, `--encoder-d-model`, `--e-noise` and `--value-weight`.
7. **Given** an `--encoder-d-model` not divisible by the encoder's head count, **When**
   `train-effect-model` starts, **Then** it refuses before training.
8. **Given** the gen-1 checkpoint, which records no encoder size, **When** any command loads it,
   **Then** it builds the encoder at 4 layers and width 256.

---

### User Story 6 - Evaluation that reports per text, per family and per policy (Priority: P6)

The operator evaluates each arm. Beyond the base spec's gates, the report gives card-disjoint
results per text, broken down by rarity bucket and rule family. It also reports the memorization
gap per family, a separate off-policy slice, a separate real-decision legality slice, and the
zero-shot result for the withheld keyword. The embedding-probe scripts run against any checkpoint
and cache. A new command runs the pooled-`e` scorer smoke test end to end.

**Why this priority**: The arms are compared by hand, and these reports are most of what is
compared. Nothing earlier depends on them.

**Independent Test**: Run `evaluate-effect-model` on a gen-2 arm with no identity variant and read
each new section and the "gate 1 skipped" line. Run each embedding-probe script with that arm's
`--checkpoint` and `--abilities-root`. Run `scorer-smoke-test --scratch-dir` and check that
nothing under `output/cardsfolder/` changed.

**Acceptance Scenarios**:

1. **Given** a gen-2 checkpoint, **When** `evaluate-effect-model` runs, **Then** card-disjoint
   results appear per record and as a per-text mean, by rarity bucket and by rule family, with the
   family results also averaged over families.
2. **Given** both validation strata, **When** the report is written, **Then** per family it gives
   the game-disjoint result minus the card-disjoint result on the same fields.
3. **Given** records with `random_seat = true`, **When** the report is written, **Then** every
   field is reported separately on them.
4. **Given** legality records with `what_if = false`, **When** the report is written, **Then** the
   legality fields are reported separately on them.
5. **Given** a checkpoint with a withheld keyword, **When** the report is written, **Then** it
   gives per-field results on records where that keyword is on the acting line or carried by an
   entity, beside the same fields on records carrying trained keywords.
6. **Given** no `--variant-checkpoint identity`, **When** evaluation runs, **Then** gate 1 is
   reported as skipped.
7. **Given** `--win-rates PATH`, **When** the decodability battery runs, **Then** it reads that
   file.
8. **Given** `scorer-smoke-test`, **When** it runs, **Then** it writes the pooled `e` of
   `--checkpoint` concatenated with the vectors of `--sealed-encoder-checkpoint` under
   `--scratch-dir`, and runs `python -m sealed train-scorer` Phase A on that directory as a
   subprocess.
9. **Given** any embedding-probe script and an arm's `--checkpoint` and `--abilities-root`,
   **When** it runs, **Then** it reads that checkpoint's vocabulary, cache and width of `e`, and
   omits its `taxonomy` columns if that cache is absent.
10. **Given** `pca_directions.py`, **When** it runs, **Then** it reports the participation ratio
    (Σλ)² / Σλ².

---

### User Story 7 - A knowledge-probe suite that compares any checkpoints (Priority: P7)

The operator freezes a probe set once per curated corpus, then probes each checkpoint. The suite
fits linear and MLP probes at each rung of a read-out ladder, from the raw board up to the model's
own prediction, over ten families of game knowledge. It reports the share of the model's knowledge
that sits in `e`, edits single inputs of real records in four board sweeps, ablates `e`, and writes
a JSON scorecard. A compare command ranks checkpoints side by side.

**Why this priority**: The suite's first run is on gen-1, and it finishes before any gen-2 arm
needs it. It is analysis tooling and gates nothing, so it ranks last in dependency, though it can be
built in parallel with stories 1 to 3.

**Independent Test**: Run `run.py --checkpoint <gen-1> --freeze-probe-set` against gen-1's corpus
and sidecars, then `run.py --checkpoint <gen-1>`. Read the scorecard: rungs and shares per family,
the sweeps, the ablation, and the recorded digest. Run `compare.py` over two scorecards.

**Acceptance Scenarios**:

1. **Given** `--freeze-probe-set`, **When** run against a curated corpus, **Then** it writes the
   probe items keyed by provenance, their labels, the probe games of both validation strata and
   the board-sweep records, with a content digest.
2. **Given** a frozen set, **When** any checkpoint trained on that corpus is probed, **Then** the
   scorecard records that set's digest.
3. **Given** a board-dependent target, **When** probed, **Then** rungs 0, 1, 1w, 1o, 2 and 3 are
   each reported, with linear and MLP scores at rungs 0 to 2, as AUC for yes/no targets and R² for
   amounts.
4. **Given** any probe's cross-validation folds, **When** assigned, **Then** no ability text falls
   on both sides, and items that pool a card's lines are grouped by card.
5. **Given** rung 3 exceeds rung 0 by less than the minimum gap, **When** the share is computed,
   **Then** it is not reported.
6. **Given** the toughness sweep, **When** run, **Then** one target creature's toughness is set to
   each value from 1 to 8 and that creature's predicted death is read at each value.
7. **Given** the ablation, **When** run, **Then** for each output field it reports the loss
   increase under each of the three replacements of `e`, each applied to every slot, to `[ACT]`
   alone, and to the card slots alone.
8. **Given** two or more scorecards, **When** `compare.py` runs, **Then** it sets them side by side
   per family and per rung and ranks checkpoints on rung 1 minus rung 1w, per probe type.

---

### Edge Cases

- A chain references an SVar already emitted earlier in the same chain, for example a
  `RepeatSubAbility` that loops back. The SVar is emitted once, at its first reach, so the chain
  terminates.
- A chain references an SVar the script does not define. `convert` reports the line and the
  missing label, and emits the chain without that segment.
- A modal resolution acts through the charm's root line today. Feature 023's sidecars give
  `option` lines no provenance and no `script_text`, and `ProvenanceKey` resolves every
  sub-ability, a chosen mode included, to its root line. Under gen-2 each `option` line carries
  its mode's key and chain (FR-005a), and a patched worker records each chosen mode through its
  own `option` line (FR-029a).
- Two modes of one charm name the same sub-ability. Each mode's chain carries it, because chains
  are built per line and an SVar is de-duplicated only within one chain.
- A card carries two modal abilities, such as a modal spell and a modal triggered ability. Each
  root row is followed by its own `option` rows, so the option flag plus adjacency still names
  each option's root.
- A chosen mode is countered or fizzles on its own, because its targets became illegal. Its effect
  half follows feature 023's outcome rules, and the other modes' effect halves are unaffected.
- A keyword line's display name matches no definition. The line stays as tokens and never
  expands.
- A keyword word appears inside a non-keyword line. It expands through the token path, not the
  display-name match.
- A keyword is in `HOST_BODIED_KEYWORDS`. It is compared by display name and never expands.
- A template has more specifiers than the line has instance values. The unfilled specifiers are
  removed. `%1$s` used twice repeats its one value.
- A held-out text has no castable carrier. The coverage round cannot satisfy it, and it is
  reported in the list under the floor rather than blocking the run.
- A family has no records in a class. It takes no share, and the budget splits across the
  families that have records.
- A family stays short even at capacity, for example one made of a few texts each already at
  `--text-cap`. The manifest reports its shortfall, and the unused budget passes to the families
  that still have capacity.
- The legality class has fewer real decisions than half its budget. What-if records fill the rest
  of the budget.
- A game names a held-out card. It is card-disjoint and never enters the game-disjoint stratum,
  whatever its hash.
- A random seat faces a decision with one legal option. It takes that option.
- A randomly drawn play cannot be completed, for example a sacrifice cost with no legal object
  left once its other draws are made. Its draws are retried, then the play is abandoned and the
  play draw repeats over the rest (FR-022c), so the seat never stalls on a play it cannot finish.
- A randomly drawn play's resolution asks its controller a question, such as which card to search
  for or whether to use a "you may". The Forge AI answers it: only casting choices are random.
- A random-seat match's JVM crashes mid-game. Its shard follows the existing crash-tolerance rules,
  and there is no match-outcome row to clean up.
- Gen-1 sidecars carry the root line only. The noise pilot's value head reads root-line labels
  from a one-segment chain. The knowledge probes on gen-1 read a copy of gen-1's
  `output/cardsfolder/` and `output/tokenscripts/` kept aside before stage 1 reconverts.
- A gen-1 checkpoint records no `--e-noise` ratio. Noise is training-only, so loading for
  inference is unaffected.
- The `taxonomy` cache is absent for a gen-2 arm. The embedding-probe columns that read it are
  omitted rather than failing the script.
- An arm at a larger encoder size does not fit the 8 GB GPU at the default batch size. The
  operator reduces the batch through the existing flags. This feature adds no automatic fallback.

## Requirements *(mandatory)*

### Functional Requirements

#### Encoding text (root spec § 4.1)

- **FR-001**: `convert` MUST set each sidecar line's `script_text` to the concatenation of every
  script line its trait owns, in this order: the root line; for a trigger, the ability its
  `Execute$` names; each sub-ability in chain order, including a `RepeatSubAbility`; for a
  replacement effect, the ability its `ReplaceWith$` names followed by that ability's own chain. A
  charm's root line carries its own script line alone (its `Choices$`, the number of modes to
  choose, and whether modes may repeat), and its modes are encoded on their `option` lines
  (FR-005a).
- **FR-002**: Segments MUST be separated by the special token `[SEG]`, seeded in the script
  vocabulary beside `[PAD]` and `[CLS]`. Every segment after the root MUST open with the SVar label
  its parent referenced it by, in its FR-002a form, followed by a colon (`SV2:`).
- **FR-002a**: `convert` MUST rename every chain label in a line's `script_text` by position: the
  values of `Execute$`, `SubAbility$`, `RepeatSubAbility$` and `ReplaceWith$`, each comma-separated
  item of `Choices$`, and the label that opens each segment. Labels are numbered `SV1`, `SV2`, … in
  order of first appearance reading the line's `script_text` from the start, and one label MUST keep
  its number at every occurrence within the line, so a reference and the segment it names carry the
  same number. Numbering restarts on every line, `option` lines included: an `option` line's opening
  label is `SV1`, and a charm root line's `Choices$` reads `SV1,SV2,…` in `Choices$` order. A
  referenced label the script does not define (FR-003) is numbered like any other. No other SVar
  reference (an amount SVar such as `NumDmg$ X`, a `Count$` definition) is renamed. `build-vocab
  --surface script` MUST seed `sv1` up to the longest chain's label count, as it seeds the special
  tokens, so `--target-size` cannot drop them.
- **FR-003**: Each SVar MUST appear at most once in a chain, at its first reach. A referenced SVar
  the script does not define MUST be reported by `convert` and left out of the chain.
- **FR-004**: The rendered prose and the converted `.txt` files MUST NOT change. The sidecar's
  other fields keep the meaning feature 023 gives them.
- **FR-005**: The prose surface MUST be the encoding text only for a line with no script: a
  keyword-derived line or a synthetic land mana line.
- **FR-005a**: Each `option` line `convert` emits for a charm mode MUST carry, in its sidecar entry,
  the provenance key of the mode's trait and a `script_text` holding that mode's own chain: the
  mode's script line opened by its label, then its sub-abilities by the FR-001 order. Option lines
  follow their root line in the sidecar, in `Choices$` order, as the converter already emits them.
  The root line states no amounts, so its value-head amount targets are empty by construction
  (FR-058).
- **FR-006**: The encoder MUST truncate at 512 tokens. `build-vocab --surface script` MUST report
  the distribution of script-line lengths in tokens. Every command that encodes MUST report each
  line it truncates, by provenance key.

#### Script tokenization (root spec § 4.2)

- **FR-007**: The script surface MUST use the prose grammar, `tokenize`.
  `AbilityTokenizer.tokenize_script` MUST be removed, with its tests.
- **FR-008**: On the script surface, the grammar MUST also end a word where a lowercase letter is
  followed by an uppercase one (`nonDragon` → `non dragon`, `YouCtrl` → `you ctrl`). The split
  applies to parameter values, never to a parameter key (FR-009).
- **FR-009**: On the script surface, each of these MUST stay one lowercased token: every parameter
  key together with its `$` (`ConditionCompare$` → `conditioncompare$`, `SP$`, `Count$`), with no
  camel-case split and no separate `$` token; the values of
  `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, and each comma-separated item of `Choices$`; the SVar
  label that opens a segment, all in their FR-002a form (`sv2`); the value of `CounterType$`.
- **FR-009a**: On the script surface, the value of `TokenScript$` MUST also split at every `_`, which
  is dropped (`w_1_1_soldier` → `w 1 1 soldier`). Everywhere else `_` stays a word character. A
  letter run followed by a digit run is already two tokens under the grammar (`GE3` → `ge 3`), and
  only the FR-009 values stay whole.
- **FR-010**: `tokenize` MUST apply the script-surface rules exactly when the loaded vocabulary is a
  script vocabulary, as `surface_of` determines from its path. `MtgTokenizer` in `price_predictor`
  MUST NOT change.
- **FR-011**: `build-vocab --surface script` MUST apply the same rules to the script lines it stages
  for the shared builder. It MUST report the unknown-token rate over script parameters with every
  `*Description$` value removed, and the number of distinct parts the camel-case split produces.

#### Keyword expansion (root spec § 5)

- **FR-012**: One constant MUST set the keyword-expansion probability for validation, evaluation and
  the embedding cache, read by `SurfaceBatcher`'s scoring path and by `encode-abilities`. Its value
  MUST be 0. Training MUST keep `--keyword-expand-p`. Unknown keywords MUST expand at every
  probability, including 0.
- **FR-013**: A keyword line's display name MUST be its script text before the first colon. It
  MUST match the definitions table case-insensitively. The whole name MUST be replaced however many
  tokens it would split into. The text after the colon MUST supply the instance values.
- **FR-014**: A keyword word inside a non-keyword line MUST expand through the token path.
- **FR-015**: `build-corpus` MUST fail when a text the display-name match expands belongs to a
  sidecar line whose `script_api_type` is not `Keyword`.
- **FR-016**: `HOST_BODIED_KEYWORDS` MUST be compared by display name, so its keywords never
  expand.
- **FR-017**: Template filling MUST fill instance values into the placeholders in order, with
  `%1$s` repeating its value at each use. Each value MUST be formatted as Forge formats it before
  filling. Every specifier left unfilled (`%s`, `%d`, and the positional forms) MUST be removed, so
  no expansion contains a `%` token. The typographic apostrophe `’` MUST be replaced by `'` when the
  definitions load.
- **FR-018**: `extract-keyword-definitions` MUST record each keyword's value formatting in the
  definitions file, and expansion MUST apply that record.
- **FR-019**: `build-vocab` MUST seed every word of every template, after specifier removal, as it
  seeds the special tokens, so `--target-size` cannot drop them. The build MUST fail if any
  definition expands to a text containing `[UNK]`.
- **FR-020**: The definition MUST be the reminder template on both surfaces. `build-vocab` MUST NOT
  scan the generated keyword scripts the definitions file records.

#### Random seat (root spec § 6.1)

- **FR-021**: `python -m sealed match-outcomes` MUST accept `--random-seat-share F` (default 0) and
  `--random-seat-probability P`. In a share `F` of matches, one seat, chosen at random per match,
  MUST be a random seat. The other seat MUST be the standard Forge AI.
- **FR-022**: The random seat MUST override four decision points: the spell or ability to play, its
  targets, the attackers to declare, and the blocks to assign. At each, with probability `P`, it
  MUST draw uniformly from the legal options; otherwise it MUST take the Forge AI's choice. The
  targets of a play the random draw picked are drawn without the `P` gate (FR-022c).
- **FR-022a**: Every priority the random seat receives MUST be a spell-or-ability decision point
  with its own `P` draw; no per-step or per-turn cap applies. At that decision the random draw MUST
  never pass. When the Forge AI
  would play something, the random draw picks uniformly among the playable spells and abilities
  instead. When the Forge AI would pass and at least one spell or ability is playable, the random
  draw plays one picked uniformly among them. When nothing is playable, the seat passes. Mana
  abilities MUST NOT be candidates for the random draw: a seat whose only playable abilities are
  mana abilities counts as having nothing playable and passes.
- **FR-022b**: At the target, attack and block decision points the random draw MUST proceed element
  by element:
  - attackers: each creature that can attack attacks independently with probability ½;
  - blockers: each creature that can block picks uniformly from no block plus the attackers it may
    legally block;
  - targets: the number of targets is drawn uniformly between the ability's minimum and maximum,
    then that many distinct legal targets are picked uniformly;
  - a declaration or target set Forge rejects as illegal (attack or block requirements and
    restrictions, target constraints across slots) MUST be redrawn, and after a retry limit set in
    code the seat MUST take the Forge AI's choice for that decision.
- **FR-022c**: Every choice made while a play the random draw picked is cast or activated MUST be
  drawn at random, with no `P` gate. Mana payment MUST stay the Forge AI's. The draws:
  - targets: by FR-022b;
  - modes of a modal spell or ability: a mode count drawn uniformly between the charm's minimum and
    maximum, then that many modes picked uniformly from those Forge allows, repeating a mode only
    where the charm permits it; a Pawprint charm draws one mode at a time, uniformly among the
    modes whose pawprint cost still fits, until the budget is spent;
  - X: an integer drawn uniformly from the smallest legal X to the largest the seat's available
    mana can pay;
  - additional-cost choices, such as what to sacrifice, discard, exile or tap: that many distinct
    legal objects picked uniformly.
  A draw Forge rejects as illegal or unpayable MUST be redrawn. After the retry limit of FR-022b the
  play MUST be abandoned and the play draw repeated over the remaining candidates; the seat passes
  when none remains. Choices made while the play resolves stay the Forge AI's. A play the Forge AI
  chose keeps its own choices, with only its targets under the `P` gate of FR-022.
- **FR-022d**: At each decision point the `P` draw MUST come before the Forge AI's computation.
  When it fails, the Forge AI decides and its hooks write records as for a Forge seat. When it
  succeeds, the Forge AI's decision code does not run, and the random seat MUST write the records
  that code's hooks would have written, through the same emitters, from the lists it draws from:
  - the play decision: one `decision` record over the candidates it evaluated, each with the
    verdict bits and payload fields the `onCandidate` path writes;
  - the attack declaration: an `attackers` record per defender, from the creatures that could attack
    and those legal against it;
  - the block declaration: a `blockers` record per attacker, from the creatures that could block and
    those legal against it, with its minimum blocker count.
  The random seat MUST stamp its own `attackers` and `blockers` records `what_if = false`
  explicitly. It writes no what-if records. These records pass through the de-duplication of
  FR-028 and are exempt from `--legality-rate` like every real decision (FR-029).
- **FR-023**: Land drops, mana payment and mulligans MUST always be the Forge AI's.
- **FR-024**: The random seat MUST be a `PlayerControllerAi` subclass the match worker installs on
  one seat, in `forge-connector`. No engine hook may be added or changed, and the
  `effect-record-hooks` branch MUST NOT change.
- **FR-025**: A match with a random seat MUST write effect records only, never a row of
  `match-outcomes.txt` or `cards-played.txt`.
- **FR-026**: `match-outcomes` MUST refuse a `--random-seat-share` above 0 without
  `--random-seat-probability` or without `--effect-records`, and either value outside 0 to 1,
  before any game.

#### Legality records (root spec § 6.2)

- **FR-027**: An `attackers` record MUST be classed a real decision when its snapshot phase is
  `combat_declare_attackers` and its actor is the active player. A `blockers` record MUST be
  classed a real decision when its snapshot phase is `combat_declare_blockers` and its anchored
  attacker is attacking. Every other legality record MUST be classed a what-if.
- **FR-028**: The legality de-duplication key MUST be the subkind, the payload and the snapshot.
- **FR-029**: `--legality-rate` MUST sample what-if records only. Every real decision MUST be
  written.

#### Modal resolutions (root spec § 6.5)

- **FR-029a**: A patched worker MUST record a modal resolution, a charm or a Pawprint charm, as one
  cost record acting through the root line plus one effect half per chosen mode, each acting
  through that mode's `option` line. Forge chains each chosen mode as a cloned sub-ability
  (`CharmEffect.chainAbilities`) and resolves them in turn. The collector MUST map each clone to
  its mode through the spell's chosen list, which is in the same order, and attribute to it the
  events of its own clause, through the per-clause hook.
- **FR-029b**: Each mode's effect half MUST carry a snapshot taken just before that mode resolves,
  so a later mode's snapshot shows the earlier modes' effects.
- **FR-029c**: A mode chosen more than once MUST produce one effect half per resolution of the mode.
  A mode that fizzles or is declined follows feature 023's outcome rules on its own effect half.
- **FR-029d**: The cost record and every effect half of one modal resolution MUST share one
  `link_id`. This widens `link_id` from joining a pair to joining a cost record with its effect
  halves. Feature 023's contract tests, `CLAUDE.md` and the record-schema contract MUST state the
  widened meaning, and readers that pair halves MUST accept several effect halves per cost record.
- **FR-029e**: A degraded worker, whose brackets cannot split modes, MUST write one effect half
  acting through the root line and carrying every chosen mode's events.
- **FR-029f**: Interventional forks that choose modes (`ForkCollector`) MUST write their effect
  halves per mode by the same rule.

#### Envelope fields (root spec § 6.3)

- **FR-030**: Every record MUST carry `random_seat`: `true` when its `actor_player` is the random
  seat, `false` otherwise.
- **FR-031**: Every `playability` record of subkind `attackers` or `blockers` MUST carry `what_if`:
  `true` for a what-if, `false` for a real decision, classed by FR-027 except on the records a
  random seat writes for itself, which FR-022d stamps `false`. No other record carries it.
- **FR-032**: `random_seat` and `what_if` MUST NOT reach the model, like `mode`, `interventional`,
  `fork` and `synthetic`.
- **FR-033**: Both fields MUST be additive under feature 023's schema-compatibility contract tests:
  no existing field is redefined. Every gen-2 record MUST carry `random_seat`, and every gen-2
  legality record MUST carry `what_if`. The gen-2 corpus is collected from scratch and holds no
  shard written without them, so readers define no default for an absent field.
- **FR-034**: `CLAUDE.md`'s effect-record shard format paragraph and
  `specs/023-ability-effect-model/contracts/record-schema.md` MUST list both fields, and the
  widened `link_id` of FR-029d.

#### Held-out coverage round (root spec § 6.4)

- **FR-035**: `python -m effects collect-coverage` MUST accept `--only-cards PATH`, the list
  `holdout-cards` writes. In this mode it MUST build decks from the listed cards alone, plus basics,
  and write full-strength records.
- **FR-036**: In `--only-cards` mode the coverage unit MUST be the held-out ability text. A text
  MUST count as satisfied once it acts in records from at least `--min-text-games` distinct games
  (default 5).
- **FR-037**: The run MUST end when every held-out text with a castable carrier is satisfied or
  retired, and MUST report the texts left under the floor.
- **FR-038**: `collect-coverage` MUST refuse `--only-cards` beside `--exclude-cards`,
  `--training-corpus` or `--split-from`, before any game.

#### Holdout (root spec § 7)

- **FR-039**: The masked template of a text MUST be its normalised `script_text` with every run of
  digits, `CARDNAME` and every `*Description$` value replaced by a fixed placeholder. A run of digits
  MUST be masked wherever it sits, inside an identifier as much as standing alone: a comparison
  threshold (`GE3` → `GE#`), a counter type (`P1P0` → `P#P#`), a token script name
  (`w_1_1_soldier` → `w_#_#_soldier`), a phase name (`Main1`), a generic mana amount. The one
  exception is the chain labels FR-002a renames, which keep their numbers, so a reference still
  names its segment. Mana symbols' colours and every letter MUST stay unmasked, so colour variants
  of one effect are different templates and `P1P1` and `M1M1` stay apart. Chain labels need no
  mask: FR-002a has already renamed them by position, so texts differing only in label names are
  one text and one template. The masked template keys the holdout only; no encoding reads it.
- **FR-040**: A template MUST be eligible when at most `--holdout-max-carriers` cards carry a text
  with that template. An eligible template MUST be held out when `crc32` of the template modulo
  1000 is below `--holdout-permille`. Every text with a held-out template is a held-out text, and
  every card carrying one is a held-out card.
- **FR-041**: `holdout-cards`, `build-corpus` and `train-effect-model` MUST share the rule through
  one implementation, selected by `--holdout-unit template|text`, default `template`. `text` MUST
  reproduce feature 023's rule exactly.
- **FR-042**: Manifests and checkpoints MUST record the holdout unit. A manifest or checkpoint that
  records none MUST be read as `text`.
- **FR-043**: `holdout-cards` MUST report the held-out templates, texts and cards, and the share of
  converted cards the list depletes.

#### Game-disjoint stratum (root spec § 8.1)

- **FR-044**: `build-corpus --game-disjoint-share F` (default 0.01) MUST replace
  `--game-disjoint-games`, which MUST be removed. A game naming no held-out card MUST enter the
  game-disjoint stratum when `crc32(game_id) mod 10⁶ / 10⁶ < F`.
- **FR-045**: For a game holding a combat record that gate 2's qualifying predicate accepts for a
  keyword in `--game-disjoint-keywords` (default first strike, deathtouch, trample, indestructible,
  wither, infect), the threshold MUST be `--game-disjoint-keyword-share` (default 0.15) instead.
- **FR-046**: Placement MUST depend only on the game, so a rebuild over a grown corpus keeps every
  game it placed before.

#### Rule families and selection (root spec § 8.2–8.4)

- **FR-047**: Every record MUST belong to exactly one rule family: for `resolution` (both halves)
  the acting line's script API type; for `trigger` the trigger mode; for `continuous` the static
  mode; for `rewrite` the replacement type; for `combat` the set of damage-step keywords carried
  by the combat's participants, or none; for `playability` the restriction or rejection reason the
  verdict turns on, by FR-047a.
- **FR-047a**: A `playability` family MUST be derived from payload fields as follows. A `decision`
  record is assigned per candidate, the unit feature 023 already trains on: the `Mode$` of the
  candidate's `responsible_static` when it names one; otherwise the first false verdict bit in the
  order `can_play`, `affordable`, `has_legal_target`, as the family `cannot-play`, `unaffordable` or
  `no-legal-target`; otherwise `none`. An `attackers` or `blockers` record takes the sorted set of
  distinct `Mode$` values of the `responsible_static` keys across its `forbidden` entries, or
  `none` when `forbidden` is empty or names no static. A static's `Mode$` is read through its
  provenance key from the sidecar, the same lookup that gives `continuous` records their family.
- **FR-048**: A record whose acting line is a keyword line MUST belong to that keyword's family,
  whatever its kind.
- **FR-049**: Training records MUST be selected per class, from the class budget the class mixture
  sets as in feature 023, in four steps. A text with `n` distinct records in a cell can supply at
  most `min(--reuse-cap × n, --text-cap)` written records there, repeats included, and a cell's
  capacity is the sum of that over its texts.
  1. The class budget is split equally across the class's families. A family whose capacity is
     below its share is written at capacity: its records are repeated up to `--reuse-cap` times each
     (default 4) toward the share, and no text exceeds `--text-cap` copies. Budget a short family
     leaves unused is split equally across the families that still have capacity, repeatedly,
     until every family is full or exhausted.
  2. In `resolution-effect`, `rewrite`, `combat` and `continuous`, each family's share is split the
     same way across outcome signatures. A record's outcome signature is the set of distinct (zone
     outcome, changed or not) pairs over its affected entities: how many entities share a pair does
     not enter it, so a wipe that kills three creatures and one that kills seven share a signature.
  3. In `playability-legality`, real decisions fill up to half the class budget before what-if
     records fill the rest. Family balancing applies within each half.
  4. Within each cell the steps above define, `--text-cap` bounds the written records per ability
     text, repeats included. A text with more distinct records than the cap contributes a random
     cap's worth under `--seed`, unrepeated. A text whose repeats would pass the cap spreads the
     cap evenly over its records, each repeated the same number of times give or take one.
- **FR-050**: `random_seat` MUST NOT be a selection key.
- **FR-051**: The card-disjoint validation sample MUST fill its resolution slots round-robin over
  the held-out texts, a fixed number of records per text taken by smallest record hash within the
  text, until the class quota is met.
- **FR-052**: Selection MUST be deterministic for a given corpus, flag set and `--seed`.

#### Manifest (root spec § 13)

- **FR-053**: The curated-corpus manifest MUST add: per class and family, the available records,
  share, records written, repeats and shortfall; per family, records written per outcome signature;
  per class, on-policy and off-policy (`random_seat`) record counts; per legality subkind,
  real-decision and what-if counts; the games admitted to the game-disjoint stratum under the
  keyword threshold; the held-out texts with no gate-one resolution record, and those recorded in
  fewer than five games; the holdout unit.

#### Training (root spec § 9)

- **FR-054**: Within a class, record weights MUST be proportional to effective_games^(−0.5), with a
  ceiling of 20 times the weight of the ability text at the 99th percentile of effective games.
- **FR-055**: Each epoch line MUST report the share of trained records by the rarity bucket of their
  text (1 game, 2–4, 5–19, 20 or more) and by rule family.
- **FR-056**: Noise on `e` MUST be applied in the batcher, in training only, to every `e` a batch
  carries, whichever variant produced it. It MUST be Gaussian with covariance `r² · Σ`, where `Σ`
  is a running average of the covariance of the batch's `e` vectors about their mean, held with
  the gradient stopped. The running average MUST start from the first batch's covariance and
  update every step with decay 0.99. `r` MUST rise linearly from 0 to `--e-noise` (default 0.1)
  over the first `--steps-per-epoch` steps.
- **FR-057**: The encoder's own fixed-σ noise (`AbilityEncoderConfig.e_noise`) MUST be removed, so
  `e` receives noise from one place only.
- **FR-058**: A training-only value head MUST read `e` alone and predict values parsed from the
  line's script, summed over its segments: damage dealt, power change, toughness change, counters
  placed, cards drawn, and mana cost by colour and generic, by count regression; whether the cost
  taps and whether it sacrifices, as binary targets. A target MUST be masked where the script states
  no fixed value. Its loss weight MUST be `--value-weight` (default 0.05).
- **FR-058a**: The cost targets MUST be read from the line's `Cost$` only. A card's mana cost covers
  the whole card and MUST NOT be pushed into any one line: no line's encoding text gains the face's
  `ManaCost`, and on a spell line the mana-cost targets MUST be masked even where its `Cost$`
  carries mana (an additional-cost spell such as Bone Splinters, `Cost$ B Sac<1/Creature>`). The tap
  and sacrifice targets of a spell line still read from its `Cost$`.
- **FR-059**: The value head MUST be filtered out at save time, like the MLM and script-API heads.
- **FR-060**: The encoder MUST have no paired-encoding loss and no pairing head.
- **FR-061**: `--encoder-layers` (default 4) and `--encoder-d-model` (default 256) MUST replace the
  hardcoded encoder constants. The checkpoint MUST record both, and every command that loads a
  checkpoint MUST build the encoder from them. A checkpoint that records neither MUST load at 4
  layers and width 256.
- **FR-062**: `train-effect-model` MUST refuse an `--encoder-d-model` not divisible by the encoder's
  head count, before training.
- **FR-063**: Checkpoints MUST record the holdout unit, `--encoder-layers`, `--encoder-d-model`,
  `--e-noise` and `--value-weight`.
- **FR-063a**: The effect model's input MUST add a learned option-kind embedding to every ability
  row that comes from an `option` line. Option rows follow their root row within the card's
  block, and the flag plus adjacency is the link between a mode and its root. No other positional
  scheme is added.

#### Evaluation (root spec § 10)

- **FR-064**: `evaluate-effect-model` MUST report card-disjoint results per text: a per-text mean
  beside the per-record figures, broken down by the rarity bucket of the text's games and by rule
  family, with the family results also averaged over families.
- **FR-065**: It MUST report the memorization gap: per rule family, the game-disjoint result minus
  the card-disjoint result on the same fields.
- **FR-066**: It MUST report every field separately on records with `random_seat = true`.
- **FR-067**: It MUST report the legality fields separately on records with `what_if = false`.
- **FR-068**: It MUST report, for the checkpoint's withheld keyword, per-field results on records
  where that keyword is on the acting line or carried by an entity, beside the same fields over
  records carrying trained keywords.
- **FR-069**: It MUST report gate 1 as skipped when no `--variant-checkpoint identity` is given.
- **FR-070**: It MUST accept `--win-rates PATH` (default `output/sealed/cards-win-rates.txt`) as the
  per-card win-rate file the decodability battery reads.
- **FR-071**: `python -m effects scorer-smoke-test` MUST write the pooled `e` of `--checkpoint`
  (default `models/effects/effect-model/latest.pt`), concatenated with the vectors of
  `--sealed-encoder-checkpoint` (default `models/sealed/encoder/latest.pt`), into the required
  `--scratch-dir`, and run `python -m sealed train-scorer` Phase A against that directory as a
  subprocess. It MUST NOT write under `output/cardsfolder/`, and MUST NOT import from
  `sealed.application`.

#### Embedding probes (root spec § 11.1)

- **FR-072**: Every script in `scripts/effect_embedding_probes/` MUST take `--checkpoint` and
  `--abilities-root` and read that checkpoint's vocabulary, cache and width of `e`. Columns that
  read the `taxonomy` cache MUST be omitted when that cache is absent.
- **FR-073**: `pca_directions.py` MUST report the participation ratio, (Σλ)² / Σλ², over the
  eigenvalues λ of the cache's covariance.

#### Knowledge probes (root spec § 11.2)

- **FR-074**: `scripts/effect_knowledge_probes/` MUST hold the suite, with modules `labels`,
  `ladder`, `sweeps`, `ablation`, `compare` and `run`, reusing the embedding probes' loaders.
- **FR-075**: `run --checkpoint PATH` MUST load any checkpoint, gen-1 included, and resolve its
  vocabulary, cache and width of `e`.
- **FR-076**: `run --freeze-probe-set` MUST enumerate, once per curated corpus, the probe items
  keyed by provenance, their labels, the probe games from both validation strata, and the
  board-sweep records, and write them with a content digest. Every run against that corpus MUST
  read the frozen set and record its digest in the scorecard.
- **FR-077**: Probe records MUST come from both validation strata of the checkpoint's corpus, and
  every board-dependent result MUST be reported per stratum. Line-level probes MUST read every
  line of the checkpoint's cache and report held-out and trained lines separately.
- **FR-078**: The suite MUST measure the corpus's interaction join rate once per probe set: the
  share of fired trigger records whose pending event the join finds in a resolution record of the
  same game.
- **FR-079**: The probe families MUST be: magnitudes and thresholds; mana production; mana usage;
  timing and non-mana costs; interactions; side; evasion and blocking; target legality; duration
  and repeatability; state dependence. Each family's targets and label sources MUST follow the
  probes record's family table and its section on assembled labels.
- **FR-080**: For a board-dependent target, the ladder's rungs MUST be: 0, the trunk's raw input
  features with every `e` zeroed; 1, rung 0 plus the acting line's `e` and, for an entity target,
  that entity's pooled `e`; 1w, rung 1 with each `e` replaced by a fixed random vector per text of
  the same width; 1o, rung 0 plus the parsed script values the target depends on; 2, the trunk's
  output at `[ACT]` or at the target entity's `[CARD]` slot; 3, the model's own prediction. A line
  property that does not depend on the board MUST read `e`, its width control, and the trunk's
  output at `[ACT]`.
- **FR-081**: Rungs 0 to 2 MUST each fit a linear probe and a two-layer MLP, with hyperparameters
  fixed in code for every checkpoint. Both scores MUST be reported at every rung: AUC for yes/no
  targets, R² for amounts.
- **FR-082**: Cross-validation folds MUST never place one ability text on both sides. Items that
  pool a card's lines MUST be grouped by card.
- **FR-083**: The share MUST be (rung 1 − rung 0) / (rung 3 − rung 0), computed separately for
  each probe type with the same type on every rung, with a bootstrap confidence interval. It MUST
  be reported only where rung 3 exceeds rung 0 by a minimum gap set in code. The MLP's share is
  the headline.
- **FR-084**: The four board sweeps MUST each edit one input of a real record and read the full
  model's prediction at each value:
  - toughness: one target creature's toughness, 1 to 8; read that creature's predicted death;
  - affordability: the acting player's untapped production, none to more than the cost; read the
    predicted affordable verdict;
  - board size: opposing creature count, 0 to 8, by copying a creature entity; read predicted
    deaths summed over the board;
  - damage: `NumDmg$` of the acting text, 1 to 8, on a spell and on a triggered ability, each
    encoded by the checkpoint's encoder; read the toughness-4 target's predicted death.
- **FR-085**: The ablation MUST report, per output field, the loss increase when `e` is replaced by
  noise matched to the vectors' mean and covariance, by the mean `e` of the line's API type, or by
  the `e` of the nearest other text, each applied to every slot, to `[ACT]` alone, and to the card
  slots alone.
- **FR-086**: `--per-layer` MUST probe each trunk layer's output through forward hooks, without
  modifying the model. `--method-c` MUST train a shallow trunk, one transformer layer or none, on
  frozen `e`; it is off by default.
- **FR-087**: `compare SCORECARD ...` MUST set scorecards side by side per family and per rung, and
  rank checkpoints on rung 1 minus rung 1w, per probe type.
- **FR-088**: Output MUST go to `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`:
  tables and one JSON scorecard per checkpoint, recording the probe set's digest.
- **FR-089**: Unit tests MUST cover the share and its minimum-gap rule, the label extractors, the
  board-sweep editor, and fold assignment never splitting a text. Fixtures MUST come from real
  records.

#### Boundaries

- **FR-090**: The import-direction test of feature 023 MUST keep passing: `effects` imports only
  its declared surface from `price_predictor` and `sealed`, and nothing from `sealed.application`.
  The knowledge-probe and embedding-probe scripts sit outside `src/` and may import from `effects`.
- **FR-091**: Gates 2 and 3 and the baseline variants MUST stay available and unchanged. Gen-2's
  evaluation does not train the baselines.
- **FR-092**: The record schema beyond `random_seat` and `what_if`, the state snapshot, events,
  per-kind payloads, collectors, caps and budgets beyond FR-021 to FR-038, the provenance join
  beyond FR-005a and FR-029a, synthetic variants, depleted collection, the effect head and its
  output heads and losses, its input beyond FR-063a, the class mixture, and the embedding-cache
  layout MUST stay as feature 023 built them.

### Key Entities

- **Encoding text**: a line's `script_text` under gen-2, the chained script segments joined by
  `[SEG]`, each after the root opened by its SVar label, every chain label renamed by position
  (`SV1`, `SV2`, …). Prose stands in only for lines with no
  script. It keys the rarity table and the holdout.
- **Masked template**: a text's normalised `script_text` with every run of digits (chain labels
  excepted), `CARDNAME` and descriptions masked. The gen-2 holdout unit: texts sharing a template are held out together.
- **Random seat**: a Forge AI seat whose play, target, attack and block choices are replaced by a
  uniform legal draw with probability `P`, a draw that never passes priority while something other
  than a mana ability is playable. A play it draws at random has every casting choice drawn at
  random too, and on a random draw the seat writes its own legality and `decision` records. Present in a share `F` of matches. Its records carry `random_seat = true`.
- **Option line**: a sidecar line for one mode of a modal ability, following its root line. It
  carries the mode's provenance key and chain, encodes to its own `e` row with the option-kind
  flag, and is the acting line of that mode's effect half.
- **Real decision / what-if**: the two classes of legality record. A real decision is the
  declaration of attackers or blockers the player actually makes. A what-if is a query the AI asked
  about a hypothetical. Marked by `what_if`.
- **Rule family**: a record's rules category: API type, trigger mode, static mode, replacement type,
  damage-step keyword set, or restriction reason (a static's mode or a failing verdict bit,
  FR-047a). The unit curation balances across.
- **Outcome signature**: the set of distinct (zone outcome, changed or not) pairs over a record's
  affected entities, counts ignored. The unit curation balances across within a family.
- **Game-disjoint stratum**: validation games chosen by a per-game hash threshold, raised for games
  with rare-keyword combats. Stable across rebuilds.
- **Value head**: a training-only head reading `e` alone to predict script-stated amounts and cost
  flags. Not saved.
- **Sweep arm**: one gen-2 training run, distinguished by `--e-dim`, `--encoder-layers` and
  `--encoder-d-model`, over the one shared curated dataset.
- **Probe set**: the frozen, digested enumeration of probe items, labels, probe games and sweep
  records for one curated corpus.
- **Scorecard**: one checkpoint's knowledge-probe results in JSON, recording the probe set's digest.
  The unit `compare` reads.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every converted line whose trait has a trigger target, a sub-ability or a replacement
  ability carries every one of those segments in its encoding text, every `option` line carries
  its mode's key and chain, and 100% of converted `.txt` files are byte-identical before and after
  the change.
- **SC-001a**: In a patched corpus, every effect half of a modal resolution acts through an
  `option` line, and no modal resolution's effect half acts through a root line.
- **SC-002**: No keyword expansion, over every definition and every keyword line in the corpus,
  contains a `%` or `[UNK]` token.
- **SC-003**: Every pair of texts that differ only in digits outside chain labels (identifiers
  included), `CARDNAME`, description values or chain-label names falls on the same side of the
  holdout.
- **SC-004**: After the held-out coverage round, every held-out text with a castable carrier is
  recorded in at least five distinct games or reported as retired, and every text below that floor
  is named in the report.
- **SC-005**: Random-seat matches add zero rows to `match-outcomes.txt` and `cards-played.txt`.
- **SC-006**: Every real legality decision collected appears in the shards, whatever
  `--legality-rate` is set to.
- **SC-007**: A rebuild of the curated corpus over a grown shard set keeps 100% of the games the
  previous build placed in the game-disjoint stratum.
- **SC-008**: Every rule family with records in a class receives records in the curated dataset,
  and every family below its share appears in the manifest with its shortfall.
- **SC-009**: Every sweep arm trains on the same curated dataset and split, and every checkpoint
  loads in `encode-abilities`, `evaluate-effect-model`, the embedding probes and the knowledge
  probes without the operator restating its encoder size or width of `e`.
- **SC-010**: The knowledge-probe suite produces a scorecard for the gen-1 checkpoint and for every
  gen-2 arm with no code change between them, and all gen-2 scorecards record one digest.
- **SC-011**: A knowledge-probe run fits the 8 GB training GPU and takes at most two GPU hours per
  checkpoint.
- **SC-012**: `scorer-smoke-test` leaves `output/cardsfolder/` unchanged.

## Assumptions

- Stories 1 to 3 are complete before the first gen-2 collection, because the encoding text, the
  holdout, the two envelope fields and per-mode modal records cannot be applied to records
  afterwards. The gen-2 corpus is
  collected from scratch. No gen-1 shard is migrated into it or mixed with it.
- Stories 5 and 7 are first exercised on gen-1's artifacts (run plan stage 0): the knowledge probes
  on the gen-1 checkpoint against a kept-aside copy of gen-1's sidecars, and the noise pilot on the
  gen-1 corpus with one-segment chains.
- The random seat's probability, the noise ratio, the withheld keyword, the sweep arms' sizes and
  the holdout fraction are run-time decisions recorded in the gen-2 record, not defaults this
  feature fixes. `--random-seat-share 0.125` is the run plan's setting, and the flag's default
  stays 0.
- The minimum gap for the share, the probes' hyperparameters, and the number of records per text in
  the card-disjoint sample are constants set in code and recorded in the plan, not flags.
- An SVar is emitted once per chain, and a missing SVar is reported and skipped (FR-003). The root
  spec does not address either case; both rules keep `convert` terminating and total.
- Arm comparison is done by hand from the scorecards, the evaluation reports and the
  embedding-probe output. No command picks a winner, and nothing in this feature gates on a result.
- Consuming `e` in the scorer, picker, draft agent or game agent is out of scope. The scorer smoke
  test is informational only.
- A card's mana cost is a card characteristic, not an ability's. The trunk reads what the snapshot
  entity carries of it, which is the mana value and the colours, not the pip composition, and relates it
  to the card's abilities through their sequence and positions. A permanent's implicit cast spell
  keeps mapping to no line (feature 023's `dropped_keys`).
- Run results land in the gen-2 and probes records' Outcome sections, never in this spec or the root
  spec.
