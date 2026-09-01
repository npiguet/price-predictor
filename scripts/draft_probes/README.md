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
`t2_card_values.csv`, `text_pca_512.npz`), and two files under
`Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1\` —
`cards-win-rates.txt` and, for `d11` only, the 1 GB sealed self-play corpus
`match-outcomes-bo1-embedding.txt`. Both are read-only.

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
| `d2_pickorder.py` | "Half its picks are settled before it looks at the board" — the exact pack-1-pick-1 ranking, the unconditional colour preference, the attribution against the scorer's card values and the encoder's text axes, and how much of the policy is context-free | `d2_pickorder.json`, `d2_p1p1_values.csv`, `d2_p1p1_logits.csv` |
| `d3_exchange.py` | state exchange between agents (feeds the Method section's shared-state note) and the per-pick policy-movement profile behind "The learning went where the reward could see it" | `d3_*.json` / `.csv` |
| `d4_commitment.py` | "Colour commitment hardens across the draft" — donor-pool transplants at seven pick numbers, with an other-colour placebo | `d4_commitment.json` |
| `d5_corpus.py` | "Gen-4 starves the drafter it feeds", build-arounds, and the training-log drift analysis — corpus and log arithmetic only, no GPU | `d5_corpus.json`, `d5_a*.csv` |
| `d6_buildfilter.py` | "The learning went where the reward could see it" — the gen-4 − gen-1 logit residual by within-pack card quality, by colour, and by position in the draft, against a sibling-checkpoint noise floor | `d6_buildfilter.json` |
| `d7_contextprobe.py` | "Gen-1's trunk reads the final score better than gen-4's" — ridge probes on the trunk summary token for the seat's eventual colours and its final pod-relative reward | `d7_contextprobe.json` |
| `d8_duplicates.py` | "A card already in the pool is worth more, not less" — a two-arm pool edit copying one of two colour- and logit-matched pack cards, with a dose ladder, a third-card placebo and an embedding-similarity split | `d8_duplicates.json` |
| `d9_signatures.py` | "The training step was set by the clip, not by the signal" — the same metrics over six checkpoints of known yardstick margin and cumulative learner picks | `d9_signatures.json` |
| `d10_rewardcolour.py` | "The reward pays for the colours gen-4 takes" — the reward priced per unit of deck colour share, controlled for the deck's card quality by two independent yardsticks and repeated inside each drafting agent's own decks; corpus arithmetic only, no GPU | `d10_rewardcolour.json` |
| `d11_winratecolour.py` | "The colour preference comes from Forge's games" — the same colour pricing on the sealed self-play corpus the encoder and scorer were fitted to, at the card level and at the deck level, against build-method fixed effects and Forge's shipped `draft_rank` as the one non-circular card-quality control; reads `Y:` and needs no GPU | `d11_winratecolour.json` |
| `make_figures.py` | the seven figures embedded in the document, rendered from the JSON/CSV outputs above (needs `matplotlib`) | `experiments/images/2026-08-29-draft-*.png` / `.svg` |

## Replay fidelity

`probe_lib.iter_corpus_states` builds states with
`draft.domain.draft_state.build_state`, the full-record oracle the live
`OnlineDraftStateTracker` is pinned to. Replaying an argmax corpus through the
checkpoint that generated it reproduces **100 %** of the recorded picks.

The trainers' own walk, `draft.application.draft_pick_states.iter_seat_pick_states`,
reproduces 97.7 % of them. The two disagree only when a card name recurs in a
later pack: `build_state` recomputes a `TAKEN` card's recency from the current
pick number, while the training walk freezes it at the moment the card left the pack.
The probes use `build_state` so that the policy being measured is the one that
was deployed.
