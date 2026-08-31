# Draft-agent behaviour probes

The probe battery behind
[`experiments/2026-08-29-draft-agent-behaviour.md`](../../experiments/2026-08-29-draft-agent-behaviour.md)
— inference-only analyses of the gen-4 production draft agent and its
predecessors. Nothing here trains anything; every script replays picks that are
already recorded and reads the policy's logits, optionally after editing the
state. Outputs land in `output/draft-probes/` (gitignored).

Run from the repo root: `python scripts/draft_probes/<script>.py`.

External inputs: the yardstick corpora and checkpoints under
`models/draft/agent/{gen1,gen3,gen4}/`, `output/cardsfolder-512/`, the staged
scorer/encoder probe outputs (`output/scorer-probes/forge_hints.csv`,
`t2_card_values.csv`, `text_pca_512.npz`), and the win-rate table at
`Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1\cards-win-rates.txt`.

## Three rules the battery is built around

1. **Every measurement carries a gen-1 column.** Gen-1 is a distillation of
   Forge's drafting AI, so a behaviour gen-1 already has is Forge's and not
   something the reinforcement learning taught.
2. **Only within-state logit contrasts are behavioural.** The policy head is
   invariant to a per-state constant, so every logit is centred inside its own
   state before anything is compared.
3. **Edits preserve token count, and carry a placebo.** Deleting a token block
   changes how many tokens the trunk averages over, which moves the logits on
   its own; blocks are blanked by substitution instead, and read against a
   random substitution of the same size.

## Script → document section

| script | what it feeds | output |
|---|---|---|
| `probe_lib.py` | shared harness: corpus replay through any checkpoint, the state interventions, batched forward passes, CONTEXT-token extraction | — |
| `d1_channels.py` | "It reads its own pool, and nothing else on the table" — count-preserving ablation of every input channel with a size-matched placebo, four checkpoints, split by pack and pick | `d1_channels.json` |
| `d1_report.py` | the same run's per-pack block sizes, and each block read against the placebo of its own size (run after `d1_channels.py`) | `d1_placebo_table.csv` |
| `d2_pickorder.py` | "The pick order and where it came from" — the exact pack-1-pick-1 ranking, the unconditional colour prior, the three-way attribution against human pick order / scorer value / encoder axes, and how much of the policy is context-free | `d2_pickorder.json`, `d2_p1p1_values.csv`, `d2_p1p1_logits.csv` |
| `d3_exchange.py` | "Skill or trajectory" — state exchange between agents, and the per-pick policy-movement profile | `d3_*.json` / `.csv` |
| `d4_commitment.py` | "Colour commitment hardens across the draft" — donor-pool transplants at seven clocks, with an other-colour placebo | `d4_commitment.json` |
| `d5_corpus.py` | "The geometry it cannot see", build-around traps, and the training-log drift analysis — corpus and log arithmetic only, no GPU | `d5_corpus.json`, `d5_a*.csv` |
| `d6_buildfilter.py` | "Where the reinforcement learning moved the policy" — the gen-4 − gen-1 logit residual by within-pack card quality, by lane, and by position in the draft, against a sibling-checkpoint noise floor | `d6_buildfilter.json` |
| `d7_contextprobe.py` | "What the CONTEXT token knows" — ridge probes on the trunk summary token for the seat's eventual colours and its final pod-relative reward | `d7_contextprobe.json` |
| `d8_duplicates.py` | "A card already in the pool is worth more, not less" — a two-arm pool edit copying one of two colour- and logit-matched pack cards, with a dose ladder, a third-card placebo and an embedding-similarity split | `d8_duplicates.json` |
| `d9_signatures.py` | "Which habits track strength, and which track training length" — the same metrics over six checkpoints of known yardstick margin and cumulative learner picks | `d9_signatures.json` |

## Replay fidelity

`probe_lib.iter_corpus_states` builds states with
`draft.domain.draft_state.build_state`, the full-record oracle the live
`OnlineDraftStateTracker` is pinned to. Replaying an argmax corpus through the
checkpoint that generated it reproduces **100 %** of the recorded picks.

The trainers' own walk, `draft.application.draft_pick_states.iter_seat_pick_states`,
reproduces 97.7 % of them. The two disagree only when a card name recurs in a
later pack: `build_state` recomputes a `TAKEN` card's recency from the current
clock, while the training walk freezes it at the moment the card left the pack.
The probes use `build_state` so that the policy being measured is the one that
was deployed.
