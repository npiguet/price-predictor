# Draft agent — online self-play GRPO fine-tuning (generation 3)

Normative spec: the new `train-draft-agent-online` command, its streaming
self-play loop, the GRPO update, the per-round diagnostics, and the CLI. Rationale
and the gen-2 post-mortem live in
[`../experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md`](../experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md)
and
[`../experiments/2026-06-10-draft-agent-gen2-design.md`](../experiments/2026-06-10-draft-agent-gen2-design.md).

**Generations** (agent lineage, not a phase number):

| gen | what |
|-----|------|
| gen-0 | Forge AI |
| gen-1 | offline imitation + critic ([`2026-05-28`](2026-05-28-draft-agent.md)) |
| gen-2 | offline RL ([`2026-06-10`](2026-06-10-draft-agent-gen2-rl.md)) |
| **gen-3** | this feature — a base checkpoint (gen-1 or gen-2) fine-tuned online |

"gen-3" below means "the next generation" generically.

# 1. What it does

A single command fine-tunes the draft agent with online, critic-free, GRPO
reinforcement learning on its own self-play. Each round:

- generate a small batch of fresh drafts from the current policy;
- take one gradient pass over that batch; discard it;
- regenerate from the updated policy; repeat.

Every update is on-policy by construction. The loss is a single term — no critic,
GAE, KL anchor, or entropy bonus. Progress is read live from a printed anchor
margin (§ 6); promotion uses the gen-2 yardstick (§ 7). The deliverable is a gen-3
checkpoint that beats its base (and Forge) on that yardstick.

# 2. Scope

**In scope**

- The `train-draft-agent-online` command: owns the loop, drives one resident Forge
  worker, applies the update, writes checkpoints (§ 5, § 9).
- The critic-free GRPO update contract (§ 4).
- Per-round stdout diagnostics + the live anchor margin (§ 6).

**Out of scope** (reused unchanged, or deferred)

- Model architecture, typed-token state, recency scheme, `.npz` cache,
  `drafts.jsonl` schema — reused from gen-1 / live-play.
- Forge draft-driving (worker protocol, pick service, state reconstruction), the
  deck builder, and the scorer — reused; the loop changes only worker *lifetime*
  (§ 5).
- The gen-2 offline trainer `train-draft-agent-rl` — untouched; gen-3 is a
  separate command.
- The gen-2 yardstick — reused as-is; no new evaluation code.
- Deferred / excluded: automated promotion (stays manual), any corpus-schema or
  provenance change, in-draft critic use, PPO / vine advantage estimation, encoder
  / picker / scorer training, the Bradley-Terry "beat the pod" objective, and
  shipping the agent into the Forge client.

# 3. How it works

```
   base checkpoint π₀  (gen-1 or gen-2, an operator input; round 0 only)
            │
            ▼
   START a long-lived Forge draft worker ONCE  (~20 s JVM startup, paid here)
     pick services:  learner label → CURRENT policy πₖ  (updated every round)
                     anchor  label → FROZEN gen-1 agent (never updated)
            │
            ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  ROUND k   (worker stays resident — no JVM restart)            │
   │                                                                │
   │   drive the resident worker → ~10 fresh drafts, sample T      │
   │      πₖ pilots the LEARNER seats; frozen + Forge bots the rest │
   │      each seat's pool → build (--build-method) → score         │
   │      (--scorer-checkpoint) → deck_score = reward              │
   │            │                                                   │
   │            ▼  fresh batch = πₖ's own picks (consumed once)     │
   │   one pass:  L = − A · logπ_T(a|s)                             │
   │      A = standardise_over_round(pod-relative LOO deck_score)   │
   │            │                                                   │
   │            ▼  πₖ → πₖ₊₁   (optimizer + LR stay in RAM;         │
   │                 push πₖ₊₁ into the learner pick service)       │
   │   print round diagnostics (§ 6)  +  save latest.pt            │
   └───────────────────────────────────────────────────────────────┘
            │  repeat: next round's learner seats are piloted by πₖ₊₁
            ▼
   watch the live anchor margin (§ 6); on a plateau, run the yardstick (§ 7)
```

- **On-policy by construction** — each round trains on the current policy's own
  fresh batch; no stale corpus, no provenance bookkeeping.
- **One resident worker** — started once, driven every round; the served policy is
  swapped between rounds. One worker suffices (drafts play no games).
- **New code** = this command: the loop, the persistent-worker driver, the
  single-term update, and the diagnostics. Everything else is reused.

# 4. The GRPO update contract

Input each round: the current checkpoint and the fresh batch it just generated in
sample mode at temperature `T`. Per learner seat, per pick `t` (a learner seat is
one whose mix label is the `--learner` label; § 8):

| Quantity | Definition |
|----------|------------|
| state `sₜ` | the typed-token state at `(seat, pack, pick)`, reconstructed exactly as gen-1's loader builds it. |
| action `aₜ` | the card recorded as taken at this pick. |
| reward `R` | the seat's pod-relative leave-one-out `deck_score` (the `--scorer-checkpoint`'s score on the seat's built deck) — `deck_score_seat − mean({other non-failed seats' deck_score})`. Terminal (γ=1): the same scalar for all 45 of the seat's picks. |
| advantage `Aₜ` | the batch-standardised reward — the round's learner-seat rewards centred and scaled to unit variance. No critic, no GAE. Shared across the seat's picks. |
| policy weight | `logπ_T(aₜ given sₜ)` — the trained policy's log-prob of the taken action, at temperature `T`. |

**Loss** (per learner pick, averaged over the round):

```
L = − Aₜ · logπ_T(aₜ | sₜ)
```

- The whole loss. No critic/value term, GAE, KL anchor, or entropy bonus.
- Exploration comes only from sampling at `T`; the pod-relative reward is the
  RLOO/group baseline (the pod is the group), so variance reduction lives in the
  reward, not a critic.
- One pass per round; the batch is discarded, never reused.
- Optimiser: AdamW, per-group max-norm clip, warmup-then-constant LR — with
  optimiser + LR state continuous across rounds (one warmup at run start).

**Failure / integrity**

- Failed build (`deck = []`, `deck_score = null`) → seat excluded from reward, pod
  mean, and gradient.
- Pod with no other non-failed seat → that learner's LOO reward is undefined →
  seat excluded.
- Round with fewer than two surviving learner rewards, or zero variance among them
  → safe no-op, logged as skipped/no-signal, never a divide-by-zero.

# 5. The online loop

The loop is owned by one in-process command. At startup it launches a single Forge
worker and binds two pick services: the learner label to the live in-training
policy (`--learner`, updated every round) and the anchor label to a frozen agent
(`--frozen` / `--anchor`, never updated). Each round:

1. **Generate** — drive the resident worker for `--drafts-per-round` fresh drafts
   (`--pick-mode sample`, temperature `T`). The current policy pilots the learner
   seats; frozen + Forge/random seats pilot the rest (`--mix`). Each seat's pool is
   built (`--build-method`) and scored (`--scorer-checkpoint`) → `deck_score`.
2. **Update** — one-pass GRPO over the learner picks (§ 4): πₖ → πₖ₊₁.
3. **Swap** — push πₖ₊₁ into the learner pick service (frozen services untouched).
4. **Log + save** — print diagnostics (§ 6), write `latest.pt` (+ periodic
   snapshot), repeat.

Rules:

- **On-policy** — learner seats always draft with the current weights, regenerated
  every round.
- **Continuity** — optimiser + LR schedule live in memory across rounds (one
  warmup at run start); checkpoints are for snapshot/recovery only, not to carry
  training state.
- **Stop** — runs until Ctrl-C or `--max-rounds`; no automatic stop, no
  held-out-loss early-stopping (§ 6).
- **Frozen anchor** — the anchor stays fixed for the whole run; never swapped to a
  later generation or "previous round".
- **Worker crash** — a crashed worker JVM is restarted (startup paid once more)
  and the run continues.

# 6. Diagnostics & live progress

The single-term loss is not a progress signal, so the run is steered by printed
diagnostics. Every round emits four axes to stdout — enough to diagnose progress,
no-progress, and collapse from the log alone:

| Axis | Printed | Catches |
|------|---------|---------|
| **Reward** | pod-relative reward mean/std; advantage spread + fraction near-zero (`abs(A) < 0.1`) | reward not discriminating picks → advantages → 0 (nothing learned) |
| **Exploration** | entropy, perplexity `exp(H)`, off-argmax rate | perplexity → 1 / off-argmax → 0 / entropy → 0 = exploration collapse |
| **Movement** | mean `logπ`, mean policy-loss term, gradient norm (pre-clip), KL-to-previous-round | step too large for the round size → large per-round KL |
| **Progress** | the anchor margin + raw component means (learner / anchor / Forge) + window size | whether the learner is pulling away from the fixed reference |

**Anchor margin** — the live progress signal:

```
anchor_margin = mean(learner deck_score) − mean(frozen-anchor deck_score)   over the last --anchor-window drafts
```

- Anchor frozen ⇒ an absolute progress curve (climbs, then plateaus);
  `learner − Forge` comes free.
- Computed from `deck_score`s already in hand — no extra command; also
  recomputable post-hoc via `analyze-generated-decks`.
- Guideline exploration band: perplexity ≈ 2–3 / off-argmax ≈ 25–40 %; watch it
  across rounds (entropy decays even at fixed `T`).

Also printed:

- a consolidated round-summary line (round index; draft + learner-pick counts;
  generation + training wall-clock; dropped-seat count; headline figures);
- a startup echo of the resolved config + validation results;
- a final summary at run end / interrupt (rounds, total drafts, latest checkpoint,
  best anchor margin and when).

A plateau in the anchor margin triggers a yardstick check (§ 7) — not an auto-stop
and not the promotion decision.

# 7. Yardstick + promotion

Reused from gen-2 (§ 6 of [`2026-06-10`](2026-06-10-draft-agent-gen2-rl.md)) — no
new code:

- One greedy (deterministic) run, one fixed agent mix (candidate + base + Forge +
  random-bot minority), randomly co-seated over many pods.
- Metric: per-agent mean `deck_score` (frozen scorer's raw scale), from
  `analyze-generated-decks --agent <each>`.
- Promote iff the candidate beats its base beyond the run-to-run noise band — a
  manual judgment; no auto-promote.
- Composition (colours, curve, types) inspectable with the same command
  (descriptive only).

# 8. CLI

`python -m draft train-draft-agent-online`. Rollout generation reuses live-play's
Forge draft worker — a single worker kept resident across rounds (§ 5), not a
per-round `generate-draft-data` subprocess.

**Agent wiring** (three categories, one flag prefix each — § 8.1):

| Flag | Default | Meaning |
|------|---------|---------|
| `--learner NAME=PATH` | _(required; exactly one)_ | The agent under training. `NAME` = its mix label; `PATH` = warm-start checkpoint. Weights update every round, held in memory. |
| `--frozen NAME=PATH` | _(repeatable)_ | A frozen (untrained) reference bound to a checkpoint — e.g. the gen-1 anchor. |
| `--anchor NAME` | _(sole `--frozen` label)_ | Which frozen label is the anchor-margin baseline (§ 6). Required only with >1 `--frozen`. |
| `--mix "label:weight,…"` | `gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` | Per-seat pod composition over learner + frozen + Forge labels. |

**Deck building & reward** (same inputs as `generate-draft-data`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer → each deck's `deck_score` (the reward). Required. |
| `--build-method` | `greedy` | Pool → deck before scoring: `greedy` (SA) or `picker`. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Used only when `--build-method picker`. |
| `--cards-path` | `output/cardsfolder/` | The `.npz` embedding cache (fixes model/scorer input width). |

**Rollout & optimisation:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--rollout-temperature` / `-T` | _(required; no default)_ | Sample temperature; all policy distributions (logπ, entropy) use it. Positive, or fail fast. |
| `--lr` | _(tunable)_ | Step size. |
| `--drafts-per-round` | `10` | Fresh drafts generated + trained on (one pass) per round. |
| `--anchor-window` | `~100` | Sliding window (drafts) for the anchor margin. |
| `--snapshot-every` | _(N rounds)_ | Cadence of timestamped snapshots (besides per-round `latest.pt`). |
| `--max-rounds` | _(none; until Ctrl-C)_ | Optional round budget. |
| `--set` | _(none; random set per draft)_ | Restrict rollouts to one set. |
| `--output-path` | `output/draft/drafts.jsonl` | Corpus every generated draft is appended to (shared file, always opened in append mode). |
| `--seed` | `42` | Torch/numpy init, per-round batch shuffling, pick-sampling RNG. Forge-side rollout randomness is not seeded. |
| `--max-consecutive-faults` | `5` | Inherited pick-fault abort: this many consecutive abandoned drafts ends the run nonzero. |
| `--warmup-steps` | `200` | Linear LR ramp over the first N optimizer steps of the **run**, then constant. An online run has no total step count, so the ramp is expressed in steps, not a fraction. |
| `--batch-size`, `--max-grad-norm` | _(gen-1 defaults; fixed)_ | Batch stays within the 8 GB VRAM budget. |

No `--gae-lambda` / `--kl-coef` / `--entropy-coef` / `--value-weight` — those
pieces are dropped.

## 8.1 Agent categories — `--learner`, `--frozen`, `--anchor`, `--mix`

A mix label names a kind of seat. Every label is exactly one of three categories:

| Category | Flag | Notes |
|----------|------|-------|
| **Forge-piloted** | _(none)_ | Built-ins `forge-full` (pure Forge AI), `forge-r30` / `forge-r100` (30 % / 100 % of picks made uniformly random). Just name them in `--mix`. |
| **Frozen** | `--frozen NAME=PATH` (repeatable) | Untrained reference bound to a checkpoint; the gen-1 anchor is one. `--anchor` picks which is the margin baseline (defaults to the sole frozen agent). |
| **Learner** | `--learner NAME=PATH` (exactly one) | The policy being trained; weights held in memory, updated each round. One learner only. |

`--mix "label:weight,…"` — each of the pod's seats is drawn independently from this
categorical (weights are relative, not exact counts).
`gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` ≈ 50 % / 30 % / 10 % / 10 % per seat, so
an 8-seat pod averages ~4 learner, ~2–3 anchor, ~1 of each random bot.

At least one learner seat per pod is guaranteed: because seats are drawn
independently a pod could come up learner-free (no on-policy picks, useless), so a
learner-free draw is resampled (or one seat forced to the learner) before the draft
is played. A high learner weight makes this rare.

**Startup validation** (fail fast):

- every `--mix` label is a Forge built-in, a `--frozen` label, or the `--learner`
  label;
- the `--learner` label appears in `--mix`;
- `--anchor` names a `--frozen` label in the mix (omit with a single `--frozen`);
- the `--learner` `PATH` and `--scorer-checkpoint` exist (and `--picker-checkpoint`
  when `--build-method picker`); `--learner` architecture matches the `.npz` width;
  `--rollout-temperature` is supplied and positive.

### Example — fine-tune gen-3 from gen-1, anchored to gen-1

```
python -m draft train-draft-agent-online \
  --learner  gen-3=models/draft/agent/gen1/latest.pt \   # agent under training (warm-started from gen-1)
  --frozen   gen-1=models/draft/agent/gen1/latest.pt \   # frozen reference — the anchor
  --anchor   gen-1 \                                      # optional: sole --frozen, so it defaults here
  --mix      gen-3:5,gen-1:3,forge-r30:1,forge-r100:1 \   # ~50/30/10/10 per seat
  --scorer-checkpoint models/sealed/scorer/latest.pt \   # frozen scorer → deck_score = reward
  --build-method greedy \
  -T 2.0  --lr 1e-4  --drafts-per-round 10  --set BLB
```

The gen-1 checkpoint appears twice — as the learner's warm-start and as the frozen
anchor. They start identical; the learner evolves each round while the anchor stays
put, so the anchor margin reads as improvement over a fixed point.

### Example — base off gen-2, still anchored to gen-1

```
python -m draft train-draft-agent-online \
  --learner  gen-3=models/draft/agent/gen2/latest.pt \   # start the learner from gen-2's weights
  --frozen   gen-1=models/draft/agent/gen1/latest.pt \   # anchor stays the frozen gen-1
  --mix      gen-3:5,gen-1:3,forge-r30:1,forge-r100:1 \
  --scorer-checkpoint models/sealed/scorer/latest.pt \
  --build-method greedy \
  -T 2.0  --lr 1e-4  --drafts-per-round 10
```

Only the `--learner` warm-start changes; `--anchor` defaults to the sole `--frozen`
label. Any consistent label set works as long as the validation rules hold.

# 9. Records & artifacts

- **Corpora** — the unchanged `drafts.jsonl` (live-play § 7); no schema or
  provenance change. The loop appends each round's drafts (for post-hoc
  margin/composition); training uses each batch once, then discards it.
- **Checkpoints** — `models/draft/agent/{timestamp}.pt` (snapshots) + `latest.pt`
  (every round), in the gen-1 format (trunk + policy + critic + recency/context
  tables, `config`, round counters) plus gen-3 `rl_metadata`: generation index,
  base-checkpoint identity, `algorithm=online-grpo`, and `lr` /
  `rollout_temperature` / `drafts_per_round`. The critic head is carried unchanged
  (untrained, unused); no critic / GAE / KL / entropy params; no encoder weights
  (Phase A).
