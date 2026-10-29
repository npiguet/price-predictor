# Ability effect model — generation 2

Normative spec for the second generation of the ability effect model: the encoding text,
script tokenization and keyword expansion, the collection changes, the holdout unit,
curation, training, evaluation reporting, and the knowledge-probe suite that compares the
arms of a hyperparameter sweep. Rationale, evidence and the run order live in
[`../experiments/2026-09-18-effect-model-gen2-improvements-design.md`](../experiments/2026-09-18-effect-model-gen2-improvements-design.md)
(the gen-2 record) and
[`../experiments/2026-09-19-effect-knowledge-probes-design.md`](../experiments/2026-09-19-effect-knowledge-probes-design.md)
(the probes record).

This spec amends
[`2026-09-05-ability-effect-model.md`](2026-09-05-ability-effect-model.md), the base spec.
Everything it does not mention stays as the base spec specifies. Where the two disagree,
this spec governs.

Generations

| gen | what |
|---|---|
| gen-1 | the checkpoint of run `09.17f`, trained under the base spec |
| gen-2 | this feature — the arms of a sweep, trained on the corpus this spec collects; arms differ in the width of `e` or the size of the encoder |

# 1. What it does

Gen-2 collects a new effect-record corpus, builds a curated dataset from it, and trains
several models on that one dataset. Each model reads the whole script chain of an ability,
sees off-policy outcomes alongside Forge's own, and trains on a corpus balanced across
rule families and outcomes. A knowledge-probe suite scores every arm on the same frozen
probe set, and the arms are compared by hand.

# 2. Scope

In scope

- The encoding text and its tokenization (§ 4), and keyword expansion (§ 5).
- The random seat, legality-record handling, two envelope fields, the held-out coverage
  round, and per-mode modal records (§ 6).
- The masked-template holdout (§ 7).
- Curation: the game-disjoint stratum, rule-family and outcome balancing, and the manifest
  (§ 8).
- Training: rarity ceiling, epoch reporting, noise on `e`, the value head, and encoder
  size (§ 9).
- Evaluation reporting and the zero-shot measurement (§ 10).
- Probe tooling: the embedding-probe scripts and the knowledge-probe suite (§ 11).

Reused unchanged

- The record schema beyond the two fields and the redefined `actor_player` of § 6.3, the
  `option` key component of § 4.1, the state snapshot, events and per-kind payloads.
- The collectors, caps and budgets beyond § 6, the provenance join beyond option lines (§
  4.1, § 6.5), synthetic variants and depleted collection.
- The effect head's input beyond the option-kind embedding (§ 9), its output heads, its
  losses beyond wiring the verdict, created-objects, MLM and script-API heads (§ 9), the
  class mixture, and the embedding-cache layout.
- Gates 2 and 3, and the baseline variants, which stay available.

Out of scope

- Consuming `e` in the scorer, picker, draft agent or game agent.
- A paired-prose loss: the encoder has none (§ 9).
- Training the baseline variants as part of gen-2's evaluation.

Base-spec sections this spec amends

| Base spec section | Amended by |
|---|---|
| Record envelope | § 6.3, § 6.5 |
| Collectors | § 6.1, § 6.2, § 6.5 |
| Provenance sidecar | § 4.1 |
| Coverage collector | § 6.4 |
| Ability identity | § 4.1 |
| Ability encoder | § 4, § 5, § 9 |
| Curated corpus | § 7, § 8 |
| Training | § 9 |
| Effect model input | § 9 |
| Embedding cache | § 5 |
| Evaluation | § 10 |
| Keyword definitions | § 5 |

# 3. How it works

```
   holdout-cards --holdout-unit template          masked-template holdout (§ 7)
            │
            ▼
   generate-pools / match-outcomes --exclude-cards  depleted, with a random seat (§ 6.1)
   match-outcomes (full strength)                   card-disjoint games, random seat too
   collect-coverage --only-cards                    held-out texts to their floor (§ 6.4)
   collect-coverage, collect-variants               uncovered cards, variants
            │
            ▼
   build-corpus                                     stable stratum, family and outcome
            │                                       balancing, manifest (§ 8)
            ▼
   one curated dataset ── every arm trains on it
            │
            ▼
   train-effect-model  ×  arms (--e-dim, --encoder-layers, --encoder-d-model)   (§ 9)
            │
            ▼
   encode-abilities → evaluate-effect-model → embedding probes → knowledge probes (§ 10, § 11)
            │
            ▼
   compare scorecards by hand
```

- Every collection-time decision is fixed before the first game: the holdout, the envelope
  fields, the random seat's share, and the stratum rule.
- Every arm reads the same curated dataset and split, and differs only in the swept flags.
- The order of the work, including a pilot collection and the runs on gen-1, is the gen-2
  record's run plan.

# 4. Encoding text and tokenization

## 4.1 Script chain

- The sidecar's `script_text` for a line is the concatenation of every script line its
  trait owns, in this order:
  1. the root line;
  2. for a trigger, the ability its `Execute$` names;
  3. each sub-ability in chain order, including a `RepeatSubAbility` where one exists;
  4. for a replacement effect, the ability its `ReplaceWith$` names, followed by that
     ability's own chain.
- Each SVar appears at most once in a chain, at its first reach. A referenced SVar the
  script does not define is reported by `convert` and left out of the chain.
- Segments are separated by the special token `[SEG]`, seeded in the script vocabulary
  beside `[PAD]` and `[CLS]`. Every segment after the root opens with the SVar label its
  parent referenced it by, followed by a colon (`SV2:`).
- Chain labels are renamed by position. Within one line's `script_text`, the values of
  `Execute$`, `SubAbility$`, `RepeatSubAbility$` and `ReplaceWith$`, each item of
  `Choices$`, and the label opening each segment become `SV1`, `SV2`, … in order of first
  appearance. A label keeps its number at every occurrence in the line, so a reference and
  the segment it names match. Numbering restarts on every line. Amount SVars (`NumDmg$ X`)
  and their `Count$` definitions keep their names.
- A charm is a set of separate abilities:
  - its root line's `script_text` is its own script line alone: its `Choices$`, the number
    of modes to choose, and whether modes may repeat;
  - each `option` line `convert` emits for a mode carries the mode trait's provenance key
    and a `script_text` holding that mode's chain, opened by its label (`SV1:`), then its
    sub-abilities in the order above;
  - option lines follow their root line in the sidecar, in `Choices$` order, and the root's
    `Choices$` reads `SV1,SV2,…`;
  - a mode's provenance key is the root key extended by an `option` component, the mode's
    0-based position in `Choices$`. A key without the component names the root. The
    component is additive in sidecars and in records' `ability` keys;
  - a mode's `option` line carries the mode's own `script_api_type`, the API type of its
    opening segment.
- An `option` line that is not a charm mode, such as a die-roll outcome, keeps the base
  spec's empty provenance and `script_text`.
- A charm with no description renders no root line. Its root key maps to no line and is
  listed in `dropped_keys`.
- The rendered prose and the converted `.txt` files do not change.
- The prose surface is the encoding text only for a line with no script: a keyword-derived
  line or a synthetic land mana line.
- The encoder truncates at 512 tokens. `build-vocab --surface script` reports the
  distribution of script-line lengths in tokens, and every command that encodes reports
  each line it truncates, by provenance key.

## 4.2 Script tokenization

- The script surface uses the prose grammar, `tokenize`.
  `AbilityTokenizer.tokenize_script` does not exist.
- On the script surface the grammar also ends a word where a lowercase letter is followed
  by an uppercase one: `nonDragon` → `non dragon`, `YouCtrl` → `you ctrl`. The split
  applies to parameter keys and values alike (`ConditionCompare$` → `condition compare $`).
  `$` is always its own token, split from the words on both sides, including a `$`-prefix
  glued to its value: `Count$Valid` → `count $ valid`.
- `build-vocab --surface script` seeds every part a parameter key, or a `$`-prefix inside a
  value (`Count$Valid`), splits into, as it seeds the special tokens, so `--target-size`
  cannot drop a rare key's part.
- These stay whole, each one lowercased token:
  - the values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, and
    each comma-separated item of `Choices$`, in their renamed form (`sv2`);
  - the SVar label that opens a segment, renamed likewise;
  - the value of `CounterType$` (`p1p1`, `m1m1`).
- The value of `TokenScript$` also splits at every `_`, which is dropped:
  `w_1_1_soldier` → `w 1 1 soldier`. Everywhere else `_` stays a word character.
- Letters and digits are separate tokens under the grammar, so a comparison threshold is
  operator then operand: `GE3` → `ge 3`. Only the values above stay whole.
- `tokenize` applies the script-surface rules when the loaded vocabulary is a script
  vocabulary, as `surface_of` determines from its path. `MtgTokenizer` in
  `price_predictor` is unchanged.
- `build-vocab --surface script` applies the same rules to the script lines it stages for
  the shared builder, and reports the unknown-token rate over script parameters with every
  `*Description$` value removed and the number of distinct parts the camel-case split
  produces. It seeds the renamed labels, `sv1` up to the longest chain's label count, as
  it seeds the special tokens, so `--target-size` cannot drop them.

# 5. Keyword expansion

- One constant, read by `SurfaceBatcher`'s scoring path and by `encode-abilities`, sets
  the expansion probability for validation, evaluation and the cache. Its value is 0, so
  known keywords stay tokens. Training keeps `--keyword-expand-p`. Unknown keywords expand
  at every probability.
- A keyword line's display name is its script text before the first colon. It matches the
  definitions table case-insensitively, the whole name is replaced however many tokens it
  would split into, and the text after the colon supplies the instance values. A keyword
  word inside another line expands through the token path.
- `build-corpus` fails when a text the display-name match expands belongs to a sidecar
  line whose `script_api_type` is not `Keyword`.
- `HOST_BODIED_KEYWORDS` is compared by display name, so its keywords never expand.
- Template filling:
  - instance values fill the template's placeholders in order, and `%1$s` repeats its
    value at each use;
  - a value is formatted as Forge formats it before filling: `extract-keyword-definitions`
    records each keyword's value formatting in the definitions file, and expansion applies
    that record;
  - every specifier left unfilled (`%s`, `%d`, and the positional forms) is removed, so no
    expansion contains a `%` token;
  - the typographic apostrophe `’` is replaced by `'` when the definitions load.
- `build-vocab` seeds every word of every template, after specifier removal, as it seeds
  the special tokens, so `--target-size` cannot drop them. The build fails if any
  definition expands to a text containing `[UNK]`.
- The definition is the reminder template on both surfaces. `build-vocab` does not scan
  the generated keyword scripts the definitions file records.

# 6. Collection

## 6.1 Random seat

- `python -m sealed match-outcomes --random-seat-share F` (default 0) makes one seat a
  random seat in a share `F` of matches. The seat is chosen at random per match, and the
  other seat is always the standard Forge AI.
- `--random-seat-probability P` is the chance that the random seat acts at random at a
  decision point.
- The random seat overrides four decision points: the spell or ability to play, its
  targets, the attackers to declare, and the blocks to assign. At each, with probability
  `P`, it draws from the legal options as below; otherwise it takes the Forge AI's choice.
  Land drops, mana payment and mulligans are always the Forge AI's.
- Every priority the seat receives is a play decision point with its own `P` draw. No
  per-step or per-turn cap applies.
- How each random draw is made:

  | Decision | Random draw |
  |---|---|
  | spell or ability | uniform over the playable spells and abilities, mana abilities excluded; it never passes while one is playable, whatever the Forge AI would do, and passes when only mana abilities or nothing is playable |
  | attackers | each creature that can attack attacks independently with probability ½ |
  | blockers | each creature that can block picks uniformly from no block plus the attackers it may legally block |
  | targets | a target count drawn uniformly between the ability's minimum and maximum, then that many distinct legal targets picked uniformly |

  A declaration or target set Forge rejects as illegal is redrawn. After a retry limit set
  in code, the seat takes the Forge AI's choice for that decision.
- A play the random draw picked has every choice made while it is cast or activated drawn
  at random, with no `P` gate. Only its mana payment is the Forge AI's.

  | Choice | Random draw |
  |---|---|
  | targets | as above |
  | modes | a count uniform between the charm's minimum and maximum, then that many modes uniform among those Forge allows, repeating only where the charm permits; a Pawprint charm draws one mode at a time among those whose pawprint cost still fits, until the budget is spent |
  | X | uniform from the smallest legal X to the largest the seat's available mana can pay |
  | additional-cost choices (sacrifice, discard, exile, tap) | that many distinct legal objects, uniform |

  A draw Forge rejects as illegal or unpayable is redrawn. After the retry limit, the play
  is abandoned and the play draw repeats over the remaining candidates. The seat passes
  when none remains. Choices made while the play resolves stay the Forge AI's. A play the
  Forge AI chose keeps its own choices, with only its targets under the `P` draw.
- The `P` draw comes before the Forge AI's computation. When it fails, the Forge AI
  decides and its hooks write records as for a Forge seat. When it succeeds, the AI's
  decision code does not run, and the random seat writes the records that code's hooks
  would have written, through the same emitters, from the lists it draws from:
  - a `decision` record per candidate it evaluated, sampled at `--playability-rate` as the
    hook path samples them;
  - an `attackers` record per defender;
  - a `blockers` record per attacker.

  It stamps its own `attackers` and `blockers` records `what_if = false` and writes no
  what-if records.
- While the seat still has a land drop this turn and a land is playable, it makes no `P`
  draw at a play decision point. The Forge AI decides that priority, so the land drop stays
  the AI's. Random plays begin once the drop is used or no land is playable.
- The random seat is a `PlayerControllerAi` subclass the match worker installs on one
  seat. No engine hook is added or changed, and the `effect-record-hooks` branch is
  unchanged.
- A match with a random seat writes effect records only, never `match-outcomes.txt` or
  `cards-played.txt`.

## 6.2 Legality records

- An `attackers` record is a real decision when its snapshot phase is
  `combat_declare_attackers` and its actor is the active player. A `blockers` record is a
  real decision when its snapshot phase is `combat_declare_blockers` and its anchored
  attacker is attacking. Every other legality record is a what-if.
- The legality de-duplication key is the subkind, the payload and the snapshot.
- The collector writes every real decision and no what-if. The what-if classification above
  applies to shards collected while what-ifs were still sampled.

## 6.3 Envelope fields

| Field | Contents |
|---|---|
| `random_seat` | `true` when the record's `actor_player` is the random seat; `false` otherwise |
| `what_if` | `playability` `attackers` and `blockers` records only: `true` for a what-if, `false` for a real decision; classed by § 6.2, except that a random seat stamps its own records `false` (§ 6.1) |

Both are collection metadata: like `mode`, `interventional`, `fork` and `synthetic`, they
never reach the model. Both are additive under the base spec's schema-compatibility rules.
Every gen-2 record carries `random_seat`, and every gen-2 legality record carries
`what_if`.

- Readers still load gen-1 shards, which stage 0 trains and probes on. An absent
  `random_seat` reads as `false`. An absent `what_if` reads as unknown and never counts as a
  real decision.
- Shard generation. A shard whose records carry `random_seat` is a gen-2 shard; one without
  it is a gen-1 shard, whose `actor_player` keeps the base spec's meaning. `build-corpus`,
  `train-effect-model`, `evaluate-effect-model` and the knowledge probes refuse a records
  set holding both, before reading records.
- `validate-corpus` fails a shard in which some records carry a field and others of the
  kinds that must carry it do not.
- On `playability` records, `actor_player` names the deciding player:

  | Subkind | `actor_player` |
  |---|---|
  | `attackers` | the controller of the candidate attackers |
  | `blockers` | the controller of the candidate blockers |
  | `decision` | the controller of the candidate ability |

  This redefines the field for those subkinds. The base spec wrote the active player for
  `attackers` and `decision` and the attacking player for `blockers`, and gen-1 shards keep
  that meaning. The schema-compatibility tests carry the redefinition as a named
  exception. § 6.2's real-decision test reads the redefined field.

## 6.4 Held-out coverage round

- `python -m effects collect-coverage --only-cards PATH` builds decks from the listed
  cards alone, plus basics, and writes full-strength records. `PATH` is the list
  `holdout-cards` writes.
- In this mode the coverage unit is the held-out ability text. A text is satisfied once it
  resolves in at least `--min-text-games` distinct games, default 5.
- A game counts toward a text only through a resolution record the corpus build keeps, the
  same count the manifest's under-five list reads.
- The run ends when every held-out text with a castable carrier is satisfied or retired,
  and reports the texts left under the floor.
- A text has a castable carrier when the base spec's castability consult judges at least
  one card carrying it castable. A text with none is reported as uncastable, never as
  satisfied.

## 6.5 Modal resolutions

- A patched worker records a modal resolution, a charm or a Pawprint charm, as one cost
  record acting through the root line plus one effect half per chosen mode, acting through
  that mode's `option` line. Forge chains the chosen modes as cloned sub-abilities and
  resolves them in turn. The collector maps each clone to its mode through the spell's
  chosen list and attributes to it its own clause's events, through the per-clause hook.
- Each mode's effect half carries a snapshot taken just before that mode resolves, so a
  later mode's snapshot shows the earlier modes' effects.
- A mode chosen more than once produces one effect half per resolution. A mode that
  fizzles or is declined follows the base spec's outcome rules on its own effect half.
- The cost record and every effect half of one modal resolution share one `link_id`.
  `link_id` therefore joins a cost record with its effect halves, not only a pair, and
  readers that pair halves accept several effect halves per cost record.
- A degraded worker writes one effect half acting through the root line, carrying every
  chosen mode's events.
- Interventional forks that choose modes write their effect halves per mode by the same
  rule.

# 7. Holdout

- The holdout unit is the masked template: the normalised `script_text` with every run of
  digits, `CARDNAME` and every `*Description$` value replaced by a fixed placeholder.
  - A run of digits is masked wherever it sits, inside an identifier too: `GE3` → `GE#`,
    `P1P0` → `P#P#`, `w_1_1_soldier` → `w_#_#_soldier`, `Main1` → `Main#`.
  - The chain labels § 4.1 renames keep their numbers, so a reference still names its
    segment. They need no mask, because the renaming already makes label names irrelevant.
  - Mana symbols' colours and every letter stay unmasked.
  - The masked template keys the holdout only. No encoding reads it.
- A template is eligible when at most `--holdout-max-carriers` cards carry a text with
  that template. An eligible template is held out when `crc32` of the template modulo 1000
  is below `--holdout-permille`. Every text with a held-out template is a held-out text,
  and every card carrying one is a held-out card.
- Token scripts are carriers under either unit, counted toward `--holdout-max-carriers`. A token
  script carrying a held-out text is held out by its script file, so a game holding that token is
  a held-out game. `holdout-cards`, `build-corpus` and `train-effect-model` all read the converted
  token tree for it.
- Under `template` a script text is held out when its template is. `collect-variants` deletes,
  before decking, every variant whose lines print a held-out text or template under the holdout
  `build-corpus` chooses. `build-corpus` does not write a training game's record acting through a
  held-out text, and counts it under the `dropped-held-out-text` stratum.
- `holdout-cards`, `build-corpus` and `train-effect-model` share the rule through
  `--holdout-unit template|text`. `text` is the base spec's rule, kept for corpora
  collected under it. Manifests and checkpoints record the unit, and one that records none
  is read as `text`.
  - `holdout-cards` and `build-corpus` default to `template`.
  - `train-effect-model` defaults to the manifest's unit and refuses a given unit that
    differs. It recomputes the holdout under that unit and refuses to train when its
    held-out texts differ from the manifest's.
- `holdout-cards` reports the held-out templates, texts and cards, and the share of
  converted cards the list depletes.

# 8. Curated corpus

## 8.1 Game-disjoint stratum

- `--game-disjoint-share F` (default 0.01) replaces `--game-disjoint-games`. A game that
  names no held-out card enters the game-disjoint stratum when `crc32(game_id)` mod 10⁶,
  divided by 10⁶, is below `F`.
- For a game holding a combat record that gate 2's qualifying predicate accepts for a
  keyword in `--game-disjoint-keywords`, the threshold is `--game-disjoint-keyword-share`
  instead.
- Placement depends only on the game, so a rebuild over a grown corpus keeps every game it
  placed before.

## 8.2 Rule families

Every record belongs to one rule family:

| Record kind | Family |
|---|---|
| `resolution`, both halves | the acting line's script API type |
| `trigger` | the trigger mode |
| `continuous` | the static mode |
| `rewrite` | the replacement type |
| `combat` | the set of damage-step keywords carried by the combat's participants, or none |
| `playability` | the restriction or rejection reason the verdict turns on, as below |

A record whose acting line is a keyword line belongs to the family of that keyword,
whatever its kind.

Playability families are read from payload fields:

- `decision`, per candidate: the `Mode$` of the candidate's `responsible_static` when it
  names one; otherwise the first false verdict bit in the order `can_play`, `affordable`,
  `has_legal_target`, as `cannot-play`, `unaffordable` or `no-legal-target`; otherwise
  `none`.
- `attackers` and `blockers`: one single mode, the rarest of those the `responsible_static`
  keys across `forbidden` name, by the build's count of surveyed legality records per mode
  (ties by name); `none` when `forbidden` names no static. A `Mode$` listing several modes
  counts as each of them. The manifest records the counts, and every reader placing legality
  records in families uses them.
- A static's `Mode$` is read through its provenance key from the sidecar, the lookup that
  gives `continuous` records their family.

## 8.3 Selection order

Training records are selected per class, from the class budget the class mixture sets as
in the base spec, in four steps. Every quota and copy plan is sized on the records of training
games only: the survey routes each game at the end of its shard (held out, else game-disjoint,
else training) before it counts the game's records toward the plan. A text with `n` distinct records in a cell supplies at
most `min(--reuse-cap × n, --text-cap)` written records there, repeats included, and a
cell's capacity is the sum of that over its texts.

1. Families. The class budget is split equally across the class's families. A family whose
   capacity is below its share is written at capacity: its records are repeated up to
   `--reuse-cap` times each towards the share, and no text passes `--text-cap` copies. The
   budget a short family leaves unused is split equally across the families that still
   have capacity, until every family is full or exhausted.
2. Outcome signatures. In the classes whose records carry per-entity outcomes
   (`resolution-effect`, `rewrite`, `combat`, `continuous`), each family's share is split
   the same way across outcome signatures. A record's outcome signature is the set of
   distinct (zone outcome, changed or not) pairs over its affected entities. How many
   entities share a pair does not enter it.
3. Real decisions. In the `playability-legality` class, real decisions fill up to half the
   class budget before what-if records fill the rest. Family balancing applies within each
   half.
4. Texts. Within each cell the steps above define, `--text-cap` bounds the written
   records per ability text, repeats included. A text with more distinct records than the
   cap contributes a random cap's worth under `--seed`, unrepeated. A text whose repeats
   would pass the cap spreads the cap evenly over its records.

`random_seat` is not a selection key.

## 8.4 Card-disjoint validation sample

The card-disjoint validation sample fills its resolution slots round-robin over the
held-out texts, a fixed number of records per text taken by smallest record hash within
the text, until the class quota is met.

## 8.5 Flags

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-unit` | `template` | `template` \| `text` (§ 7) |
| `--game-disjoint-share` | 0.01 | base hash threshold for the game-disjoint stratum |
| `--game-disjoint-keyword-share` | 0.15 | threshold for a game with a qualifying combat for a listed keyword |
| `--game-disjoint-keywords` | first strike, deathtouch, trample, indestructible, wither, infect | keywords whose qualifying combats raise the threshold |
| `--reuse-cap` | 4 | maximum repeats of one record when a family or signature is short |
| `--text-cap` | 200 | maximum written records per ability text in a cell, repeats included |

## 8.6 First-strike-step repair

Gen-2's collector wrote no creature damage, life change or death in the first-strike damage
step (the experiments record gives the cause). `python scripts/repair_first_strike_steps.py
--corpus DIR [--dry-run]` repairs a built gen-2 corpus in place and leaves the raw shards alone.
For every real `combat` record with `combat_substep = first_strike` (not a probe fork) that
carries no `life_change` or `poison_change`:

- a record whose blocks pair any attacker and blocker where either side has first strike or
  double strike is dropped;
- otherwise, from the record's combat `damage_dealt` events to players: one `life_change` per
  player for the damage it took, a positive `life_change` for each lifelinker's controller, and
  a `poison_change` (cause: the attacker's controller) instead of life loss for an infect
  creature's damage, in the collector's event shape.

Keywords resolve as gate 2 resolves them. The counts are appended to `DIR/repairs.json`; the
manifest is unchanged. A second run changes nothing. The collector tests both damage-step
names, so a corpus built from records collected afterwards needs no repair.

# 9. Training

- Rarity weight. Within a class, records weight ∝ effective_games^(−0.5), with a ceiling
  of 20× the weight of the ability text at the 99th percentile of effective games.
- Epoch reporting. Each epoch line reports the share of trained records by the rarity
  bucket of their text (1 game, 2–4, 5–19, 20 or more) and by rule family.
- Noise on `e`. Applied in the batcher, in training only, to every `e` a batch carries,
  whichever variant produced it:
  - the noise is Gaussian with covariance `r² · Σ`, where `Σ` is a running average of the
    covariance of the batch's `e` vectors about their mean, held with the gradient
    stopped, and `r` is the noise ratio;
  - the running average starts from the first batch's covariance and updates every step
    with decay 0.99;
  - `r` rises linearly from 0 to `--e-noise` over the first `--steps-per-epoch` steps.
- Value head. A training-only head reads `e` alone and predicts values parsed from the
  line's script, summed over its segments: damage dealt, power change, toughness change,
  counters placed, counters removed, cards drawn, and mana cost by colour and generic, by count
  regression; whether the cost taps and whether it sacrifices, as binary targets. The targets
  are read from the line's script text on both encoding surfaces. A target is masked where the
  script states no fixed value. The head is filtered out at save time, like the MLM and
  script-API heads.

  | Target | Read from | Absent key |
  |---|---|---|
  | damage | `NumDmg$` | 0 |
  | power change | `NumAtt$`, `AddPower$` | 0 |
  | toughness change | `NumDef$`, `AddToughness$` | 0 |
  | counters placed | `CounterNum$` on `PutCounter`, `PutCounterAll` | 1 |
  | counters removed | `CounterNum$` on `RemoveCounter` | 1 |
  | counters removed | `CounterNum$` on `RemoveCounterAll` | masked |
  | cards drawn | `NumCards$` on `Draw` | 1 |

  A `MoveCounter`, `AddOrRemoveCounter` or `ReplaceCounter` segment masks both counter targets.
  A non-literal value masks its target.
- At-least targets. For each amount target, the value head also predicts whether the
  amount's magnitude is at least k, for k = 1 to 40, each as a binary target under the
  amount's mask, beside the amount's count or signed loss. The forty are averaged, so each
  amount's block weighs as one binary target. A magnitude above 40 sets all forty.
- Input layout. The model's inputs map the names the collector writes onto its fixed
  vocabularies, and carry thermometer codes of the board's amounts:

  | Input | Collector's name | Model's slot |
  |---|---|---|
  | phase | `combat_begin`, `combat_declare_attackers`, `combat_declare_blockers`, `combat_first_strike_damage`, `combat_end` | `begin_combat`, `declare_attackers`, `declare_blockers`, `first_strike_damage`, `end_combat` |
  | counter type | `+1/+1`, `-1/-1` | `P1P1`, `M1M1` |
  | keyword | `protection_from_<colour>` | `protection` |
  | keyword | `landwalk` | `landwalk`, a slot appended to the keyword list |

  - A thermometer code of a value n is 40 positions, the magnitude cap, position k set when
    n ≥ k.
  - Each card slot carries thermometer codes of its total power, its total toughness, and its
    total toughness minus its marked damage; each player slot carries one of its life. An
    entity with no power and toughness carries zeros.
  - The checkpoint records `input_layout = gen-2` in its training settings. A checkpoint that
    records none loads with the earlier layout: no name mapping, no thermometer codes, no
    landwalk slot.
- Cost targets. The value head reads cost targets from the line's `Cost$` only. A card's
  mana cost covers the whole card, so no line's encoding text gains the face's
  `ManaCost`, and a spell line's mana-cost targets are masked even where its `Cost$`
  carries mana (`Cost$ B Sac<1/Creature>`). A spell line's tap and sacrifice targets still
  read from its `Cost$`. A charm's root line states no amounts, so its amount targets are
  empty.
- No pairing term. The encoder has no paired-encoding loss and no pairing head.
- Magnitude cap. Every count and signed-delta magnitude is scored at no more than 40, its
  sign kept: in the per-entity field losses, the created-objects counts, the
  constant-predictor floor and the evaluation's Poisson deviance. A per-entity count is also
  scored from zero, never below. The value head's script-stated amounts are not capped.
- Verdict and created-objects heads. Training applies both heads' losses as the base spec
  defines them, beside the per-entity loss: the verdict bits on `decision` records, the cost
  paid on cost halves, the trigger-fired bit on `trigger` records, and the created-objects
  slots on effect halves and `rewrite` records. A `decision` record's `[ACT]` reads the `e`
  of its candidate's acting line.
- MLM and script-API heads. Training applies the base spec's MLM loss (`--mlm-weight`,
  `--mlm-mask-prob`) and script-API loss (`--api-weight`, default 0.003). A masked position's
  numeric value and number flag are zeroed with its token. The script-API head predicts the
  line's `script_api_type` and the parameter keys of every segment of its `script_text`, on both
  encoding surfaces. The
  MLM head is sized from `--encoder-d-model`. Both are training-only and filtered at save
  time.
- Encoder size. `--encoder-layers` and `--encoder-d-model` replace the hardcoded encoder
  constants. The checkpoint records both, and every command that loads a checkpoint builds
  the encoder from them. A checkpoint that records neither loads at 4 layers and width 256.
  A checkpoint that records the removed `e_noise` field loads with it ignored.
- Option-kind embedding. The effect model adds a learned option-kind embedding to every
  ability row from an `option` line. Option rows follow their root row within the card's
  block, and the flag plus adjacency links a mode to its root. A snapshot names abilities by
  root key, so the input adds one row per `option` line after the root row it follows. No
  other positional scheme is added.

- Lethality copies. With `--lethality-copies K`, each training step appends K copies of every
  record that qualifies, labelled by rule 704.5f-g rather than by a collected outcome:

  | step | rule |
  |---|---|
  | qualifies | a resolution effect half whose chain has one `DealDamage`/`DamageAll` segment with a literal or paid-X amount, or one `Pump`/`PumpAll` lowering toughness by one; no `Remember*`, `Condition*`, `ReplaceDying*`, `Imprint*`, `DamageSource`, `DividedAsYouChoose`, `DivideEvenly`, `ExcessSVar`, `ExcessDamage` or `KW` parameter in any segment |
  | hit | each battlefield creature the amount hit changes in the amount and its zone only; each player it hit loses exactly the amount; no other creature is hit |
  | excluded | a source with deathtouch, wither, infect or lifelink; a hit creature with indestructible, a shield counter or an attachment; a real outcome the rule contradicts |
  | copy | a new amount and one hit creature's new toughness, each drawn uniformly from 1 to 8, the creature's marked damage cleared; a literal amount is edited in the copy's text, read under a key of its own, and a paid X in the copy's references. On the prose surface the edit is to the converted prose's amount, with its role spans shifted by any change in length |
  | label | every hit creature dies when its damage reaches its toughness or its toughness falls to 0 or below, and stays otherwise; its damage or toughness change, and each hit player's life change, is the new amount |

  The copies are drawn from the step's generator before its other draws, so K = 0 trains
  exactly as before, and the setting is recorded only when K > 0. The epoch line counts the
  copies under the family `(lethality copy)`.

- Resume. At the end of every epoch, after validation, training rewrites `resume.pt` in
  `--model-output` atomically. It holds:
  - the epoch, the optimizer step, and whether the early stopper ended the run;
  - the encoder's and the effect model's full state, training-only heads included;
  - the optimizer state, the early stopper's best and epochs since, and the noise state on `e`;
  - the CPU and CUDA random-generator states;
  - the settings a resumed epoch must share: seed, corpus digest, variant, sizes, batching,
    curriculum, withheld keyword, augmentation rates, loss weights, tokenizer rules and input
    layout.

  Whenever an epoch sets a new card-disjoint best, `resume.pt` is also copied to
  `resume-best.pt`; copied into another run directory as its `resume.pt`, it continues from
  the best epoch. `--resume` continues from it at the next epoch. A resumed run trains what the uninterrupted
  run would have. A resume file whose settings differ is refused, naming each difference; the
  epoch count and the patience may change. A file at an epoch at or past `--epochs`, or one
  this run's `--patience` would stop at, trains nothing; under a larger patience, or 0, a run
  that stopped early continues.

| Flag | Default | Meaning |
|---|---|---|
| `--resume` | off | continue from `resume.pt` in `--model-output` |
| `--lethality-copies` | 0 | counterfactual copies per qualifying damage or -X/-X resolution in a training step |
| `--patience` | 5 | epochs without a new card-disjoint best before stopping; 0 never stops early |
| `--lr-scale` | 1.0 | factor on every step's learning rate; not a resume setting |
| `--save-every-epoch` | off | also keep each epoch's checkpoint under `<model-output>/epochs/` |
| `--e-noise` | 0.1 | noise ratio `r`, relative to the running covariance of `e` |
| `--value-weight` | 0.05 | value-head loss weight |
| `--api-weight` | 0.003 | script-API loss weight |
| `--encoder-layers` | 4 | encoder transformer layers |
| `--encoder-d-model` | 256 | encoder width |

# 10. Evaluation

`evaluate-effect-model` reports, beyond the base spec:

- Card-disjoint results per text: a per-text mean beside the per-record figures, broken
  down by the rarity bucket of the text's games and by rule family, with the family
  results also averaged over families.
- The memorization gap: per rule family, the difference between the game-disjoint and the
  card-disjoint results on the same fields.
- The off-policy slice: every field reported separately on records with `random_seat =
  true`.
- Real legality decisions: the legality fields reported separately on records with
  `what_if = false`.
- The zero-shot keyword check: for the checkpoint's withheld keyword, per-field results on
  records where that keyword is on the acting line or carried by an entity, beside the
  same fields over records carrying trained keywords.
- Gate 1 reports as skipped when no `--variant-checkpoint identity` is given. A checkpoint
  that records no curated corpus is refused (exit 2). Its unique-text stratum is the curated card-disjoint
  stratum's resolution records whose acting text no record of the curated `training/` stratum
  acts with.
- Gate 2 reads the real combat records of the checkpoint's game-disjoint games, from the curated
  corpus's `validation/game-disjoint`; forks and interventional records never qualify. Its 200 counts distinct records, and its agreement averages over
  (record, carrier) observations. Each field is read as the gate's probability times the field's
  expected value, or times the class probability for a categorical.
- Matched real-vs-fork agreement, the probe-diff re-check and the role-polarity probe are not
  implemented. Each reports `NOT_IMPLEMENTED`, printed `[todo]`, on every run.
- `--variant-scripts` defaults to the tree the corpus manifest records.
- Gate 2's first strike and double strike rows read first-strike-step combat records only
  (`combat_substep = first_strike`). Each scores what the carrier does in that step to
  whoever it fights: the blocker's damage taken, and its death where the hit is lethal, or
  the defending player's life when the carrier is unblocked. With the keyword the carrier
  deals that damage, and without it, none.
- A failed gate 1 or gate 3 is printed as `GATES FAILED: <gates>` and the command still
  exits 0. Only an error, such as a hash, split or corpus mismatch, exits non-zero.

`--win-rates PATH` (default `output/sealed/cards-win-rates.txt`) names the per-card
win-rate file the decodability battery reads.

# 11. Probe tooling

## 11.1 Embedding probes

- Every script in `scripts/effect_embedding_probes/` takes `--checkpoint` and
  `--abilities-root` and reads that checkpoint's vocabulary, cache and width of `e`.
  Columns that read the `taxonomy` cache are omitted when that cache is absent.
- `pca_directions.py` reports the participation ratio, (Σλ)² / Σλ², over the eigenvalues λ
  of the cache's covariance.
- `effect_profiles.py` profiles each curated record once, merging the training stratum's copies
  of a record by `record_id` across shards.

## 11.2 Knowledge probes

`scripts/effect_knowledge_probes/` is an importable package holding the suite, with modules
`common`, `labels`, `ladder`, `sweeps`, `ablation`, `board`, `board_edits`, `record_walk`,
`probe_set`, `scorecard`, `compare` and `run`. The entry scripts run by path.

- `run --checkpoint PATH` loads any checkpoint and resolves its vocabulary, cache and
  width of `e`. The curated corpus, the sidecar trees and the variant tree resolve from the
  checkpoint and its corpus manifest. A checkpoint naming no curated corpus, a corpus rebuilt
  since the checkpoint trained on it, and a probe set frozen against another corpus are refused.
- `run --freeze-probe-set` enumerates, once per curated corpus, the probe items keyed by
  provenance, their labels, the probe games from both validation strata, and the
  board-sweep records, and writes them with a content digest. Every run against that
  corpus reads the frozen set and records its digest in the scorecard.
- The families are magnitudes and thresholds, mana production, mana usage, timing and
  non-mana costs, interactions, side, evasion and blocking, target legality, duration and
  repeatability, and state dependence. The probes record gives each family's targets and
  label sources.
- Every amount label is capped at a magnitude of 40, its sign kept, as the model's targets
  are.
- The observed-mana label sums each `mana_produced` event's `mana_by_color` counts. An empty map
  counts for nothing, and a record whose every such event is empty has no label.
- A line counts as held out only when every card carrying it is a held-out card.
- For a board-dependent target, the ladder's rungs are:

  | Rung | Input |
  |---|---|
  | 0 | the trunk's raw input features, every `e` zeroed |
  | 1 | rung 0 plus the acting line's `e`, and for an entity target that entity's pooled `e` |
  | 1w | rung 1 with each `e` replaced by a fixed random vector per text, of the same width |
  | 1o | rung 0 plus the parsed script values the target depends on |
  | 2 | the trunk's output at `[ACT]`, or at the target entity's `[CARD]` slot |
  | 3 | the model's own prediction, gated by the affected probability; `damage_taken` reads the amount alone |

  A line property that does not depend on the board reads `e`, its width control, and the
  trunk's output at `[ACT]`.

- Rungs 0 to 2 each fit a linear probe and a two-layer MLP, with hyperparameters fixed in
  code for every checkpoint. Both scores are reported at every rung: AUC for yes/no
  targets, R² for amounts.
- Cross-validation folds never place one ability text on both sides. Items that pool a
  card's lines are grouped by card.
- The share is (rung 1 − rung 0) / (rung 3 − rung 0), computed separately for each probe
  type with the same type on every rung, with a bootstrap confidence interval. It is
  reported only where rung 3 exceeds rung 0 by a minimum gap set in code. The MLP's share
  is the headline.
- Each board sweep edits one input of a real record and reads the full model's prediction
  at each value:

  | Sweep | Edited input | Read |
  |---|---|---|
  | toughness | one target creature's toughness, 1 to 8 | that creature's predicted death |
  | affordability | the acting player's untapped production, none to more than the cost, with the player's lands untapped or tapped to match | the predicted affordable verdict |
  | board size | opposing creature count, 0 to 8, by copying a creature entity | predicted deaths summed over the board |
  | damage | `NumDmg$` of the acting text, 1 to 8, on a spell and on a triggered ability, each encoded by the checkpoint's encoder | the toughness-4 target's predicted death |

- The ablation reports, per output field, the loss increase when `e` is replaced by noise
  matched to the vectors' mean and covariance, by the mean `e` of the line's API type, or
  by the `e` of the nearest other text. Each replacement is applied to every slot, to
  `[ACT]` alone, and to the card slots alone.
- The board-abilities ablation runs the same measurement on six subsets of the validation
  strata, the first 300 records of each per stratum in shard order (`--no-board` skips it):

  | subset | records |
  |---|---|
  | legality-static | attack or block legality, a creature kept out by a static that is not a keyword line |
  | legality-keyword | the same, kept out by keyword lines only |
  | legality-none | the same, nobody kept out |
  | resolution-replacement | an effect half, a replacement line on a permanent other than the source |
  | resolution-static | an effect half, a non-keyword static on another permanent, no replacement |
  | resolution-none | an effect half, no other permanent with a static, trigger or replacement line |

  A permanent's lines are read from its sidecar, because the snapshot lists its spell and
  activated abilities only. The report is `board-abilities.md`: per field, each subset's loss
  and its increase with noise in the card slots.
- `--per-layer` probes each trunk layer's output. `--method-c` trains a shallow trunk, one
  transformer layer or none, on frozen `e`.
- `compare SCORECARD ...` sets scorecards side by side per family and per rung, and ranks
  checkpoints on rung 1 minus rung 1w, per probe type and per score: AUC and R² gains are ranked
  apart.
- Unit tests cover the share and its minimum-gap rule, the label extractors, the
  board-sweep editor, and fold assignment never splitting a text. Fixtures come from real
  records.

# 12. Operator surface

```
python -m sealed match-outcomes
    [--random-seat-share F]              default 0
    [--random-seat-probability P]        required when --random-seat-share > 0,
                                         as is --effect-records

python -m effects holdout-cards
    [--holdout-unit template|text]       default template

python -m effects collect-coverage
    [--only-cards PATH]                  held-out coverage round
    [--min-text-games N]                 default 5; with --only-cards

python -m effects build-corpus
    [--holdout-unit template|text]       default template
    [--game-disjoint-share F]            default 0.01; replaces --game-disjoint-games
    [--game-disjoint-keyword-share F]    default 0.15
    [--game-disjoint-keywords LIST]      default first strike, deathtouch, trample,
                                         indestructible, wither, infect
    [--reuse-cap N]                      default 4
    [--text-cap N]                       default 200; counts repeats

python -m effects train-effect-model
    [--holdout-unit template|text]       default the manifest's unit
    [--cards-folder DIR ...]             default output/cardsfolder/, output/tokenscripts/
    [--e-noise R]                        default 0.1; noise ratio
    [--value-weight W]                   default 0.05
    [--api-weight W]                     default 0.003
    [--encoder-layers N]                 default 4
    [--encoder-d-model N]                default 256

python -m effects encode-abilities
    [--cards-folder DIR ...]             default output/cardsfolder/, output/tokenscripts/

python -m effects evaluate-effect-model
    [--win-rates PATH]                   default output/sealed/cards-win-rates.txt
    [--cards-folder DIR ...]             default output/cardsfolder/, output/tokenscripts/
    [--variant-scripts DIR]              default the tree the corpus manifest records

python scripts/effect_knowledge_probes/run.py
    --checkpoint PATH
    [--freeze-probe-set] [--per-layer] [--method-c]

python scripts/effect_knowledge_probes/compare.py SCORECARD ...
```

Startup validation, before any game, build or training step:

- `match-outcomes` refuses a `--random-seat-share` above 0 without
  `--random-seat-probability` or without `--effect-records`, and either value outside 0
  to 1;
- `collect-coverage` refuses `--only-cards` beside `--exclude-cards`, `--training-corpus`
  or `--split-from`;
- `train-effect-model` refuses an `--encoder-d-model` not divisible by the encoder's head
  count, a `--holdout-unit` that differs from the manifest's, and a recomputed holdout that
  differs from the manifest's;
- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the knowledge probes
  refuse a records set mixing gen-1 and gen-2 shards (§ 6.3).

# 13. Records and artifacts

- Effect-record shards carry the two envelope fields of § 6.3, and a modal resolution's
  `link_id` joins its cost record with every effect half (§ 6.5).
- Sidecars carry each `option` line's provenance key and chain (§ 4.1).
- The curated-corpus manifest adds:
  - per class and family: available records, share, records written, repeats and
    shortfall;
  - per family: records written per outcome signature;
  - per class: on-policy and off-policy (`random_seat`) record counts;
  - per legality subkind: real-decision and what-if counts;
  - the games admitted to the game-disjoint stratum under the keyword threshold;
  - the held-out texts with no gate-one resolution record, and those recorded in fewer
    than five games;
  - the holdout unit;
  - per class, the training target (`class_targets`), and per family, the planned count;
  - the `dropped-held-out-text` stratum's record count.
- Checkpoints add the holdout unit, `--encoder-layers`, `--encoder-d-model`, `--e-noise`
  and `--value-weight`.
- The definitions file adds each keyword's value formatting (§ 5).
- Knowledge-probe output goes to
  `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`: tables and one JSON
  scorecard per checkpoint, recording the probe set's digest.

Run results land in the gen-2 and probes records' Outcome sections, never in this spec.
