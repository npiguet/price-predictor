# Ability effect model — generation 2

Generation 2 of the ability effect model: a newly collected corpus, the whole script chain as encoding text, a repaired script tokenizer and keyword expansion, curation balanced across rule families and outcomes, covariance-shaped training noise, an auxiliary value head, and a knowledge-probe suite that compares the arms of a hyperparameter sweep. Rationale, evidence and the run order: [`../experiments/2026-09-18-effect-model-gen2-improvements-design.md`](../experiments/2026-09-18-effect-model-gen2-improvements-design.md) (the gen-2 record) and [`../experiments/2026-09-19-effect-knowledge-probes-design.md`](../experiments/2026-09-19-effect-knowledge-probes-design.md) (the probes record).

This spec amends [`2026-09-05-ability-effect-model.md`](2026-09-05-ability-effect-model.md) (the base spec). Everything this spec does not mention stays as the base spec specifies it. Where the two disagree, this spec governs.

**Generations.** Gen-1 is the checkpoint of run `09.17f`, trained under the base spec. Gen-2 is the set of checkpoints, the arms of a sweep, trained on the corpus this spec collects. Each arm differs from the others in the width of `e` or the size of the encoder.

**Scope:** the encoding text, script-surface tokenization, keyword expansion, collection (random seat, legality records, the held-out coverage round), the holdout unit, curation, training, evaluation reporting, and probe tooling.

**Out of scope:** consuming `e` in the scorer, picker, draft or game agent; the paired-prose loss, which the encoder does not have (§ Training); re-running the four baseline variants, which stay available and are not part of gen-2's evaluation.

# Amended sections of the base spec

| Base spec section | Amendment |
|---|---|
| Record envelope | two collection-metadata fields: `random_seat`, `what_if` (§ Collection) |
| Collectors | random seat; legality record classification, de-duplication and sampling (§ Collection) |
| Coverage collector | `--only-cards` mode for the held-out coverage round (§ Collection) |
| Ability identity | `script_text` is the whole script chain (§ Encoding text) |
| Ability encoder | tokenization, keyword expansion, noise, and training-only heads (§ Encoding text, § Script tokenization, § Keyword expansion, § Training) |
| Curated corpus | holdout unit, game-disjoint stratum, rule-family and outcome balancing, card-disjoint sampling, manifest contents (§ Holdout, § Curated corpus) |
| Training | rarity ceiling, epoch reporting, noise, value head, encoder size flags; no pairing term (§ Training) |
| Embedding cache | keyword expansion probability shared with scoring (§ Keyword expansion) |
| Evaluation | reporting per text, family and slice; zero-shot measurement; win-rates path; scorer smoke-test command (§ Evaluation) |
| Keyword definitions | value formatting recorded per keyword; generated scripts not scanned (§ Keyword expansion) |

# Encoding text

- The sidecar's `script_text` for a line is the concatenation of every script line its trait owns, in this order:
  1. the root line;
  2. for a trigger, the ability its `Execute$` names;
  3. each sub-ability in chain order, including a `RepeatSubAbility` where one exists;
  4. for a charm, each mode its `Choices$` names, in list order;
  5. for a replacement effect, the ability its `ReplaceWith$` names, followed by that ability's own chain.
- Segments are separated by the special token `[SEG]`, seeded in the script vocabulary beside `[PAD]` and `[CLS]`. Every segment after the root opens with the SVar label its parent referenced it by (`DBChange:`).
- The rendered prose and the converted `.txt` files do not change.
- The prose surface is the encoding text only for a line with no script: a keyword-derived line or a synthetic land mana line.
- The encoder truncates at 512 tokens. `build-vocab --surface script` reports the distribution of script-line lengths in tokens, and every command that encodes reports each line it truncates, by provenance key.

# Script tokenization

- The script surface uses the prose grammar (`tokenize`). `AbilityTokenizer.tokenize_script` does not exist.
- On the script surface the grammar also ends a word where a lowercase letter is followed by an uppercase one: `nonDragon` → `non dragon`, `YouCtrl` → `you ctrl`.
- Exempt from that split, each kept as one lowercased token:
  - the values of `Execute$`, `SubAbility$`, `ReplaceWith$`, and each comma-separated item of `Choices$`;
  - the SVar label that opens a segment;
  - the value of `CounterType$` (`p1p1`, `m1m1`).
- `tokenize` applies the script-surface rules when the loaded vocabulary is a script vocabulary, as `surface_of` determines from its path. `MtgTokenizer` in `price_predictor` is unchanged.
- `build-vocab --surface script` applies the same rules to the script lines it stages for the shared builder.
- `build-vocab --surface script` reports the unknown-token rate over script parameters with every `*Description$` value removed, and the number of distinct parts the camel-case split produces.

# Keyword expansion

- **One scoring probability.** A single constant, read by `SurfaceBatcher`'s scoring path and by `encode-abilities`, sets the expansion probability for validation, evaluation and the cache. Its value is 0: known keywords stay tokens. Training keeps `--keyword-expand-p`. Unknown keywords expand at every probability.
- **Keyword lines match by display name.** A keyword line's display name is its script text before the first colon. It matches the definitions table case-insensitively, the whole name is replaced however many tokens it would split into, and the text after the colon supplies the instance values. A keyword word inside another line expands through the token path.
- **Matched texts are checked.** `build-corpus` fails when a text the display-name match expands belongs to a sidecar line whose `script_api_type` is not `Keyword`.
- **Host-bodied keywords.** `HOST_BODIED_KEYWORDS` is compared by display name, so its keywords never expand whatever their tokens.
- **Template filling.**
  - Instance values fill the template's placeholders in order; `%1$s` repeats its value at each use.
  - A value is formatted as Forge formats it before filling. `extract-keyword-definitions` records each keyword's value formatting in the definitions file, and expansion applies that record.
  - Every specifier left unfilled (`%s`, `%d`, and the positional forms) is removed. No expansion contains a `%` token.
  - The typographic apostrophe `’` is replaced by `'` when the definitions load.
- **Definition words are reserved.** `build-vocab` seeds every word of every template, after specifier removal, as it seeds the special tokens, so `--target-size` cannot drop them. The build fails if any definition expands to a text containing `[UNK]`.
- **The definition is the reminder template** on both surfaces. `build-vocab` does not scan the generated keyword scripts the definitions file records.

# Collection

## Random seat

- `python -m sealed match-outcomes --random-seat-share F` (default 0) makes one seat a random seat in a share `F` of matches. The seat is chosen at random per match. The other seat is always the standard Forge AI.
- `--random-seat-probability P` is the chance that the random seat acts at random at a decision point. It is required whenever `--random-seat-share` is above 0 and has no default.
- The random seat overrides four decision points: the spell or ability to play, its targets, the attackers to declare, and the blocks to assign. At each, with probability `P`, it draws uniformly from the legal options; otherwise it takes the Forge AI's choice. Land drops, mana payment and mulligans are always the Forge AI's.
- The random seat is a `PlayerControllerAi` subclass the match worker installs on one seat. No engine hook is involved.
- A match with a random seat writes effect records only, never `match-outcomes.txt` or `cards-played.txt`.

## Legality records

- An `attackers` record is a real decision when its snapshot phase is `combat_declare_attackers` and its actor is the active player. A `blockers` record is a real decision when its snapshot phase is `combat_declare_blockers` and its anchored attacker is attacking. Every other legality record is a what-if.
- The legality de-duplication key is the subkind, the payload and the snapshot.
- `--legality-rate` samples what-if records only. Every real decision is written.

## Envelope fields

| Field | Contents |
|---|---|
| `random_seat` | `true` when the record's `actor_player` is the random seat; `false` otherwise |
| `what_if` | `playability` `attackers` and `blockers` records only: `true` for a what-if, `false` for a real decision |

Both are collection metadata: like `mode`, `interventional`, `fork` and `synthetic`, they never reach the model. Both are additive under the base spec's schema-compatibility rules.

## Held-out coverage round

- `python -m effects collect-coverage --only-cards PATH` builds decks from the listed cards alone, plus basics, and writes full-strength records. `PATH` is the list `holdout-cards` writes. `--only-cards` is refused beside `--exclude-cards`, `--training-corpus` or `--split-from`.
- In this mode the coverage unit is the held-out ability text. A text is satisfied once it acts in records from at least `--min-text-games` distinct games (default 5).
- The run ends when every held-out text with a castable carrier is satisfied or retired. It reports the texts left under the floor.

# Holdout

- The holdout unit is the masked template: the normalised `script_text` with every number, `CARDNAME` and every `*Description$` value replaced by a fixed placeholder.
- A template is eligible when at most `--holdout-max-carriers` cards carry a text with that template. An eligible template is held out when `crc32` of the template modulo 1000 is below `--holdout-permille`. Every text with a held-out template is a held-out text, and every card carrying one is a held-out card.
- `holdout-cards`, `build-corpus` and `train-effect-model` share the rule through `--holdout-unit template|text` (default `template`). `text` is the base spec's rule, kept for building and reading corpora collected under it. Manifests and checkpoints record the unit.
- `holdout-cards` reports the held-out templates, texts and cards, and the share of converted cards the list depletes.

# Curated corpus

## Game-disjoint stratum

- `--game-disjoint-share F` (default 0.01) replaces `--game-disjoint-games`. A game that names no held-out card enters the game-disjoint stratum when `crc32(game_id)` mod 10⁶, divided by 10⁶, is below `F`.
- The threshold is `--game-disjoint-keyword-share` (default 0.15) instead for a game holding at least one combat record that gate 2's qualifying predicate accepts for a keyword in `--game-disjoint-keywords` (default: first strike, deathtouch, trample, indestructible, wither, infect).
- Placement depends only on the game, so a rebuild over a grown corpus keeps every game it placed before.
- The manifest lists the games that entered under the keyword threshold.

## Rule families

Every record belongs to one rule family:

| Record kind | Family |
|---|---|
| `resolution` (both halves) | the acting line's script API type |
| `trigger` | the trigger mode |
| `continuous` | the static mode |
| `rewrite` | the replacement type |
| `combat` | the set of damage-step keywords carried by the combat's participants, or none |
| `playability` | the restriction or rejection reason the verdict turns on, or none |

A record whose acting line is a keyword line belongs to the family of that keyword, whatever its kind.

## Selection order

Training records are selected per class, in four steps. The class's record budget is the one the class mixture sets, as in the base spec.

1. **Families.** The class budget is split equally across the class's families. A family with fewer records than its share is written whole, and its records are repeated up to `--reuse-cap` times each (default 4) towards the share. The budget a short family leaves unused is split equally across the families that still have records, until every family is full or exhausted.
2. **Outcome signatures.** Within a family of a class whose records carry per-entity outcomes (`resolution-effect`, `rewrite`, `combat`, `continuous`), the family's share is split the same way across outcome signatures. A record's outcome signature is the multiset of (zone outcome, changed or not) over its affected entities.
3. **Real decisions.** In the `playability-legality` class, real decisions fill up to half the class budget before what-if records fill the rest. Family balancing applies within each half.
4. **Texts.** Within each cell the steps above define, `--text-cap` bounds the records per ability text, and records beyond it are dropped at random under `--seed`.

`random_seat` is not a selection key.

## Card-disjoint validation sample

The card-disjoint validation sample fills its resolution slots round-robin over the held-out texts, a fixed number of records per text taken by smallest record hash within the text, until the class quota is met.

## Manifest additions

- Per class and family: available records, share, records written, repeats, shortfall.
- Per family: records written per outcome signature.
- Per class: on-policy and off-policy (`random_seat`) record counts.
- Per legality subkind: real-decision and what-if counts.
- The games admitted to the game-disjoint stratum under the keyword threshold.
- The held-out texts with no gate-one resolution record, and those recorded in fewer than five games.
- The holdout unit.

New flags:

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-unit` | `template` | `template` \| `text` |
| `--game-disjoint-share` | 0.01 | base hash threshold for the game-disjoint stratum |
| `--game-disjoint-keyword-share` | 0.15 | threshold for a game with a qualifying combat for a listed keyword |
| `--game-disjoint-keywords` | first strike, deathtouch, trample, indestructible, wither, infect | keywords whose qualifying combats raise the threshold |
| `--reuse-cap` | 4 | maximum repeats of one record when a family or signature is short |

# Training

- **Rarity weight.** Within a class, records weight ∝ effective_games^(−0.5), with a ceiling of 20× the weight of the ability text at the 99th percentile of effective games.
- **Epoch reporting.** Each epoch line reports the share of trained records by the rarity bucket of their text (1 game, 2–4, 5–19, 20 or more) and by rule family.
- **Noise on `e`.** Applied in the batcher, in training only, to every `e` a batch carries, whichever variant produced it:
  - The noise is Gaussian with covariance `r² · Σ`, where `Σ` is a running average of the covariance of the batch's `e` vectors about their mean, held with the gradient stopped, and `r` is the noise ratio.
  - The running average starts from the first batch's covariance and updates every step with decay 0.99.
  - `r` rises linearly from 0 to `--e-noise` over the first `--steps-per-epoch` steps.
  - `--e-noise` (default 0.1) is the noise ratio.
- **Value head.** A training-only head reads `e` alone and predicts values parsed from the line's script, summed over its segments:
  - damage dealt, power change, toughness change, counters placed, cards drawn — count regression;
  - mana cost by colour and generic — count regression;
  - whether the cost taps, whether it sacrifices — binary.

  A target is masked where the script states no fixed value. `--value-weight` (default 0.05) weights its loss. It is filtered out at save time, like the MLM and script-API heads.
- **No pairing term.** The encoder has no paired-encoding loss and no pairing head.
- **Encoder size.** `--encoder-layers` (default 4) and `--encoder-d-model` (default 256) replace the hardcoded encoder constants. The checkpoint records both, and every command that loads a checkpoint builds the encoder from them.

New and changed flags:

| Flag | Default | Meaning |
|---|---|---|
| `--e-noise` | 0.1 | noise ratio `r`, relative to the running covariance of `e` |
| `--value-weight` | 0.05 | value-head loss weight |
| `--encoder-layers` | 4 | encoder transformer layers |
| `--encoder-d-model` | 256 | encoder width |

# Evaluation

`evaluate-effect-model` reports, beyond the base spec:

- **Card-disjoint results per text.** A per-text mean beside the per-record figures, broken down by the rarity bucket of the text's games and by rule family, with the family results also averaged over families.
- **Memorization gap.** Per rule family, the difference between the game-disjoint and the card-disjoint results on the same fields.
- **Off-policy slice.** Every field reported separately on records with `random_seat = true`.
- **Real legality decisions.** The legality fields reported separately on records with `what_if = false`.
- **Zero-shot keyword.** For the checkpoint's withheld keyword, per-field results on records where that keyword is on the acting line or carried by an entity, beside the same fields over records carrying trained keywords.
- **Gate 1** reports as skipped when no `--variant-checkpoint identity` is given.

`--win-rates PATH` (default `output/sealed/cards-win-rates.txt`) names the per-card win-rate file the decodability battery reads.

`python -m effects scorer-smoke-test` runs the base spec's pooled-`e` scorer smoke test: it writes the pooled `e` of `--checkpoint`, concatenated with the vectors of `--sealed-encoder-checkpoint`, into `--scratch-dir`, and runs `python -m sealed train-scorer` Phase A against that directory as a subprocess. It never writes under `output/cardsfolder/`.

# Probe tooling

## Embedding probes

- Every script in `scripts/effect_embedding_probes/` takes `--checkpoint` and `--abilities-root` and reads that checkpoint's vocabulary, cache and width of `e`. Columns that read the `taxonomy` cache are omitted when that cache is absent.
- `pca_directions.py` reports the participation ratio, (Σλ)² / Σλ², over the eigenvalues λ of the cache's covariance.

## Knowledge probes

`scripts/effect_knowledge_probes/` holds the suite, with modules `labels`, `ladder`, `sweeps`, `ablation`, `compare` and `run`.

- **Adapter.** `run --checkpoint PATH` loads any checkpoint and resolves its vocabulary, cache and width of `e`.
- **Probe set.** `run --freeze-probe-set` enumerates, once per curated corpus, the probe items keyed by provenance, their labels, the probe games from both validation strata, and the board-sweep records, and writes them with a content digest. Every run against that corpus reads the frozen set and records its digest in the scorecard.
- **Families.** Magnitudes and thresholds, mana production, mana usage, timing and non-mana costs, interactions, side, evasion and blocking, target legality, duration and repeatability, state dependence. The probes record gives each family's targets and label sources.
- **Ladder.** For a board-dependent target:

  | Rung | Input |
  |---|---|
  | 0 | the trunk's raw input features, every `e` zeroed |
  | 1 | rung 0 plus the acting line's `e`, and for an entity target that entity's pooled `e` |
  | 1w | rung 1 with each `e` replaced by a fixed random vector per text, of the same width |
  | 1o | rung 0 plus the parsed script values the target depends on |
  | 2 | the trunk's output at `[ACT]`, or at the target entity's `[CARD]` slot |
  | 3 | the model's own prediction |

  A line property that does not depend on the board reads `e`, its width control, and the trunk's output at `[ACT]`.
- **Probes.** Rungs 0 to 2 each fit a linear probe and a two-layer MLP, with hyperparameters fixed in code for every checkpoint. Both scores are reported at every rung.
- **Scores.** AUC for yes/no targets, R² for amounts.
- **Folds.** Cross-validation folds never place one ability text on both sides. Items that pool a card's lines are grouped by card.
- **Share.** (rung 1 − rung 0) / (rung 3 − rung 0), computed separately for each probe type with the same type on every rung, with a bootstrap confidence interval. It is reported only where rung 3 exceeds rung 0 by a minimum gap set in code. The MLP's share is the headline.
- **Board sweeps.** Each edits one input of a real record and reads the full model's prediction at each value:

  | Sweep | Edited input | Read |
  |---|---|---|
  | toughness | one target creature's toughness, 1 to 8 | that creature's predicted death |
  | affordability | the acting player's untapped production, none to more than the cost | the predicted affordable verdict |
  | board size | opposing creature count, 0 to 8, by copying a creature entity | predicted deaths summed over the board |
  | damage | `NumDmg$` of the acting text, 1 to 8, on a spell and on a triggered ability, each encoded by the checkpoint's encoder | the toughness-4 target's predicted death |

- **Ablation.** Per output field, the loss increase when `e` is replaced by noise matched to the vectors' mean and covariance, by the mean `e` of the line's API type, or by the `e` of the nearest other text; each applied to every slot, to `[ACT]` alone, and to the card slots alone.
- **Optional views.** `--per-layer` probes each trunk layer's output. `--method-c` trains a shallow trunk, one transformer layer or none, on frozen `e`.
- **Output.** `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`: tables and one JSON scorecard per checkpoint.
- **Comparison.** `compare SCORECARD ...` sets several scorecards side by side per family and per rung, and ranks checkpoints on rung 1 minus rung 1w, per probe type.
- **Tests.** Unit tests cover the share and its minimum-gap rule, the label extractors, the board-sweep editor, and fold assignment never splitting a text. Fixtures come from real records.

# CLI summary (changes only)

```
python -m sealed match-outcomes
    [--random-seat-share F]          default 0
    [--random-seat-probability P]    required when --random-seat-share > 0

python -m effects collect-coverage
    [--only-cards PATH]              held-out coverage round; refuses the exclusion flags
    [--min-text-games N]             default 5; with --only-cards

python -m effects holdout-cards
    [--holdout-unit template|text]   default template

python -m effects build-corpus
    [--holdout-unit template|text]   default template
    [--game-disjoint-share F]        default 0.01; replaces --game-disjoint-games
    [--game-disjoint-keyword-share F]    default 0.15
    [--game-disjoint-keywords LIST]  default first strike, deathtouch, trample,
                                     indestructible, wither, infect
    [--reuse-cap N]                  default 4

python -m effects train-effect-model
    [--e-noise R]                    default 0.1; noise ratio
    [--value-weight W]               default 0.05
    [--encoder-layers N]             default 4
    [--encoder-d-model N]            default 256

python -m effects evaluate-effect-model
    [--win-rates PATH]               default output/sealed/cards-win-rates.txt

python -m effects scorer-smoke-test
    [--checkpoint PATH]              default models/effects/effect-model/latest.pt
    [--sealed-encoder-checkpoint PATH]   default models/sealed/encoder/latest.pt
    [--scratch-dir DIR]              required

python scripts/effect_knowledge_probes/run.py
    --checkpoint PATH
    [--freeze-probe-set]
    [--per-layer] [--method-c]

python scripts/effect_knowledge_probes/compare.py SCORECARD ...
```

Run results land in the gen-2 and probes records' Outcome sections, never in this spec.
