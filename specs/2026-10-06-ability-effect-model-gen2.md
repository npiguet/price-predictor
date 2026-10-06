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
- Evaluation reporting, the zero-shot measurement, and the scorer smoke-test command (§
  10).
- Probe tooling: the embedding-probe scripts and the knowledge-probe suite (§ 11).

Reused unchanged

- The record schema beyond the two fields of § 6.3, the state snapshot, events and
  per-kind payloads.
- The collectors, caps and budgets beyond § 6, the provenance join beyond option lines (§
  4.1, § 6.5), synthetic variants and depleted collection.
- The effect head's input beyond the option-kind embedding (§ 9), its output heads and
  losses, the class mixture, and the embedding-cache layout.
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
    `Choices$` reads `SV1,SV2,…`.
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
  by an uppercase one: `nonDragon` → `non dragon`, `YouCtrl` → `you ctrl`.
- These stay whole, each one lowercased token:
  - the values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, and
    each comma-separated item of `Choices$`, in their renamed form (`sv2`);
  - the SVar label that opens a segment, renamed likewise;
  - the value of `CounterType$` (`p1p1`, `m1m1`).
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
- How each random draw is made:

  | Decision | Random draw |
  |---|---|
  | spell or ability | uniform over the playable spells and abilities, mana abilities excluded; it never passes while one is playable, whatever the Forge AI would do, and passes when only mana abilities or nothing is playable |
  | attackers | each creature that can attack attacks independently with probability ½ |
  | blockers | each creature that can block picks uniformly from no block plus the attackers it may legally block |
  | targets | a target count drawn uniformly between the ability's minimum and maximum, then that many distinct legal targets picked uniformly |

  A declaration or target set Forge rejects as illegal is redrawn. After a retry limit set
  in code, the seat takes the Forge AI's choice for that decision.
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
- `--legality-rate` samples what-if records only. Every real decision is written.

## 6.3 Envelope fields

| Field | Contents |
|---|---|
| `random_seat` | `true` when the record's `actor_player` is the random seat; `false` otherwise |
| `what_if` | `playability` `attackers` and `blockers` records only: `true` for a what-if, `false` for a real decision |

Both are collection metadata: like `mode`, `interventional`, `fork` and `synthetic`, they
never reach the model. Both are additive under the base spec's schema-compatibility rules.
Every gen-2 record carries `random_seat`, and every gen-2 legality record carries
`what_if`. The gen-2 corpus holds no shard without them, so readers define no default.

## 6.4 Held-out coverage round

- `python -m effects collect-coverage --only-cards PATH` builds decks from the listed
  cards alone, plus basics, and writes full-strength records. `PATH` is the list
  `holdout-cards` writes.
- In this mode the coverage unit is the held-out ability text. A text is satisfied once it
  acts in records from at least `--min-text-games` distinct games, default 5.
- The run ends when every held-out text with a castable carrier is satisfied or retired,
  and reports the texts left under the floor.

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

- The holdout unit is the masked template: the normalised `script_text` with every number,
  `CARDNAME` and every `*Description$` value replaced by a fixed placeholder. Mana symbols'
  colours stay unmasked. Chain labels need no mask, because § 4.1 has already renamed them
  by position.
- A template is eligible when at most `--holdout-max-carriers` cards carry a text with
  that template. An eligible template is held out when `crc32` of the template modulo 1000
  is below `--holdout-permille`. Every text with a held-out template is a held-out text,
  and every card carrying one is a held-out card.
- `holdout-cards`, `build-corpus` and `train-effect-model` share the rule through
  `--holdout-unit template|text`, default `template`. `text` is the base spec's rule, kept
  for corpora collected under it. Manifests and checkpoints record the unit, and one that
  records none is read as `text`.
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
- `attackers` and `blockers`: the sorted set of distinct `Mode$` values of the
  `responsible_static` keys across `forbidden`, or `none` when `forbidden` names no static.
- A static's `Mode$` is read through its provenance key from the sidecar, the lookup that
  gives `continuous` records their family.

## 8.3 Selection order

Training records are selected per class, from the class budget the class mixture sets as
in the base spec, in four steps. A text with `n` distinct records in a cell supplies at
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
  counters placed, cards drawn, and mana cost by colour and generic, by count regression;
  whether the cost taps and whether it sacrifices, as binary targets. A target is masked
  where the script states no fixed value. The head is filtered out at save time, like the
  MLM and script-API heads.
- Cost targets. The value head reads cost targets from the line's `Cost$` only. A card's
  mana cost covers the whole card, so no line's encoding text gains the face's
  `ManaCost`, and a spell line's mana-cost targets are masked even where its `Cost$`
  carries mana (`Cost$ B Sac<1/Creature>`). A spell line's tap and sacrifice targets still
  read from its `Cost$`. A charm's root line states no amounts, so its amount targets are
  empty.
- No pairing term. The encoder has no paired-encoding loss and no pairing head.
- Encoder size. `--encoder-layers` and `--encoder-d-model` replace the hardcoded encoder
  constants. The checkpoint records both, and every command that loads a checkpoint builds
  the encoder from them. A checkpoint that records neither loads at 4 layers and width 256.
- Option-kind embedding. The effect model adds a learned option-kind embedding to every
  ability row from an `option` line. Option rows follow their root row within the card's
  block, and the flag plus adjacency links a mode to its root. No other positional scheme
  is added.

| Flag | Default | Meaning |
|---|---|---|
| `--e-noise` | 0.1 | noise ratio `r`, relative to the running covariance of `e` |
| `--value-weight` | 0.05 | value-head loss weight |
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
- Gate 1 reports as skipped when no `--variant-checkpoint identity` is given.

`--win-rates PATH` (default `output/sealed/cards-win-rates.txt`) names the per-card
win-rate file the decodability battery reads.

`python -m effects scorer-smoke-test` runs the base spec's pooled-`e` scorer smoke test.
It writes the pooled `e` of `--checkpoint`, concatenated with the vectors of
`--sealed-encoder-checkpoint`, into `--scratch-dir`, and runs `python -m sealed
train-scorer` Phase A against that directory as a subprocess. It never writes under
`output/cardsfolder/`.

# 11. Probe tooling

## 11.1 Embedding probes

- Every script in `scripts/effect_embedding_probes/` takes `--checkpoint` and
  `--abilities-root` and reads that checkpoint's vocabulary, cache and width of `e`.
  Columns that read the `taxonomy` cache are omitted when that cache is absent.
- `pca_directions.py` reports the participation ratio, (Σλ)² / Σλ², over the eigenvalues λ
  of the cache's covariance.

## 11.2 Knowledge probes

`scripts/effect_knowledge_probes/` holds the suite, with modules `labels`, `ladder`,
`sweeps`, `ablation`, `compare` and `run`.

- `run --checkpoint PATH` loads any checkpoint and resolves its vocabulary, cache and
  width of `e`.
- `run --freeze-probe-set` enumerates, once per curated corpus, the probe items keyed by
  provenance, their labels, the probe games from both validation strata, and the
  board-sweep records, and writes them with a content digest. Every run against that
  corpus reads the frozen set and records its digest in the scorecard.
- The families are magnitudes and thresholds, mana production, mana usage, timing and
  non-mana costs, interactions, side, evasion and blocking, target legality, duration and
  repeatability, and state dependence. The probes record gives each family's targets and
  label sources.
- For a board-dependent target, the ladder's rungs are:

  | Rung | Input |
  |---|---|
  | 0 | the trunk's raw input features, every `e` zeroed |
  | 1 | rung 0 plus the acting line's `e`, and for an entity target that entity's pooled `e` |
  | 1w | rung 1 with each `e` replaced by a fixed random vector per text, of the same width |
  | 1o | rung 0 plus the parsed script values the target depends on |
  | 2 | the trunk's output at `[ACT]`, or at the target entity's `[CARD]` slot |
  | 3 | the model's own prediction |

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
  | affordability | the acting player's untapped production, none to more than the cost | the predicted affordable verdict |
  | board size | opposing creature count, 0 to 8, by copying a creature entity | predicted deaths summed over the board |
  | damage | `NumDmg$` of the acting text, 1 to 8, on a spell and on a triggered ability, each encoded by the checkpoint's encoder | the toughness-4 target's predicted death |

- The ablation reports, per output field, the loss increase when `e` is replaced by noise
  matched to the vectors' mean and covariance, by the mean `e` of the line's API type, or
  by the `e` of the nearest other text. Each replacement is applied to every slot, to
  `[ACT]` alone, and to the card slots alone.
- `--per-layer` probes each trunk layer's output. `--method-c` trains a shallow trunk, one
  transformer layer or none, on frozen `e`.
- `compare SCORECARD ...` sets scorecards side by side per family and per rung, and ranks
  checkpoints on rung 1 minus rung 1w, per probe type.
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
    [--e-noise R]                        default 0.1; noise ratio
    [--value-weight W]                   default 0.05
    [--encoder-layers N]                 default 4
    [--encoder-d-model N]                default 256

python -m effects evaluate-effect-model
    [--win-rates PATH]                   default output/sealed/cards-win-rates.txt

python -m effects scorer-smoke-test
    [--checkpoint PATH]                  default models/effects/effect-model/latest.pt
    [--sealed-encoder-checkpoint PATH]   default models/sealed/encoder/latest.pt
    --scratch-dir DIR

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
  count.

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
  - the holdout unit.
- Checkpoints add the holdout unit, `--encoder-layers`, `--encoder-d-model`, `--e-noise`
  and `--value-weight`.
- The definitions file adds each keyword's value formatting (§ 5).
- Knowledge-probe output goes to
  `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`: tables and one JSON
  scorecard per checkpoint, recording the probe set's digest.

Run results land in the gen-2 and probes records' Outcome sections, never in this spec.
