# Draft agent — live Forge integration

This spec is **normative** — it specifies *what* to build and *how it behaves
from the outside*: the feature, the command-line surface and the user-visible
output, and the worker↔supervisor communication protocol. It deliberately stops
short of an implementation manual — the internal component design, module
layout, and step-by-step build plan come later via speckit. Design rationale and
rejected alternatives belong in `../experiments/`.

It is the live-integration capability deferred by
[`2026-05-28-draft-agent.md`](2026-05-28-draft-agent.md), which trained the
two-headed agent offline and explicitly left "integrating the trained policy as
a live Forge draft seat" to a follow-on.

**Generation convention** (the agent lineage, not a phase number): **gen-0** is
the Forge AI; **gen-1** is the first trained agent (the
[`2026-05-28`](2026-05-28-draft-agent.md) checkpoint); **gen-2** would be the
next agent trained on self-play data. This spec is not itself a generation — it
is the machinery that pilots a trained agent (gen-1 today, or any later
generation) as a live seat and emits the self-play corpus that **trains** the
next generation.

# 1. What this feature does

Let a trained draft agent (`models/draft/agent/*.pt`) **pilot a seat in a live
Forge draft**: Forge runs a real pod, and at each of the agent's picks it hands
the current pack to our Python process, which runs the agent's policy and returns
the chosen card. Any subset of the eight seats can be model-piloted; the rest
stay on the gen-0 Forge AI (or its random-override variants `forge-r30` /
`forge-r100`). Every finished draft is labeled and appended to the same
`drafts.jsonl` corpus produced by gen-1.

The result is two things at once:

- a drop-in **self-play corpus** — the model's own drafts feed the next
  generation's training;
- a **strength measurement** — within each pod, the model seats' built-deck
  scores can be compared against the Forge seats' over shared boosters.

This spec covers only live play and self-play data generation. RL fine-tuning of
the agent on these rollouts — and training the gen-2 agent on this corpus — are
separate, later specs.

# 2. Scope

**In scope:**

- A request/response side-channel layered onto the existing draft worker so
  Forge can ask Python for a model seat's pick (§ 4).
- Live model-piloting of selected seats: those seats' picks come from the trained
  policy; all other seats remain pure Forge AI, and the finished-draft transcript
  is shaped identically regardless of who piloted each seat.
- An extension of the `generate-draft-data` command that registers model-backed
  agent labels and routes their picks through the policy (§ 5).

**Out of scope (later specs / unchanged):**

- RL fine-tuning (actor-critic / PPO) from the self-play rollouts.
- An automated multi-generation self-play *loop* — this spec produces one corpus
  per invocation; chaining generations is operator-driven.
- Live use of the critic head for in-draft decisions — only the policy selects;
  the critic is at most logged.
- Changes to the offline training command (`train-draft-agent`), the model
  architecture, the `.npz` embedding cache, or the `drafts.jsonl` schema.
- Parallel/multi-worker draft throughput — one worker per invocation, as today;
  the supervisor still restarts it on JVM crash.

# 3. How it works (overview)

```
   ┌──────────────────────────┐                        ┌──────────────────────────┐
   │  DraftWorkerMain (JVM)   │  ── PICK-REQUEST ──►   │  generate-draft-data     │
   │  drives the Forge pod    │     (pack in hand)     │  supervisor (Python)     │
   │                          │  ◄── PICK-RESPONSE ──  │  - trained policy        │
   │                          │     (chosen card)      │  - picker + scorer label │
   │                          │   ── EVENT-JSON ──►    │                          │
   │                          │    (finished draft)    │                          │
   └──────────────────────────┘                        └──────────────────────────┘
                                                                    │ append
                                                                    ▼
                                                         output/draft/drafts.jsonl
```

All three transport lines (`<<DRAFT-PICK-REQUEST>>`, `<<DRAFT-PICK-RESPONSE>>`,
`<<DRAFT-EVENT-JSON>>`) are detailed in § 4.

The worker drives a real Forge pod pick by pick. For a seat whose agent is a
Forge built-in (`forge-full`, `forge-r30`, `forge-r100`) it picks in-JVM exactly
as gen-1 does. For a **model** ("external") seat it hands the pack currently in
hand to the supervisor and waits for the chosen card. The Python supervisor runs
the trained policy and answers. At draft completion the worker emits the
unchanged completed-draft transcript, which the supervisor labels (build + score
each seat's deck with the frozen picker and scorer) and appends to
`drafts.jsonl` — exactly the gen-1 path, with the only difference being where
each pick came from.

The deck-labeling, scoring, run-id stamping, resume counting, progress logging,
and JSONL append are all reused from gen-1 unchanged, so the blast radius is the
pick side-channel and the policy glue. The single new correctness requirement is
that the state the supervisor reconstructs online — incrementally, from the packs
it has shown a model seat — is **identical** to the state the offline trainer
built for that same `(seat, pack, pick)` from a finished draft record (same
typed-token multiset, same per-card recency). This is what lets the pick-request
stay minimal (§ 4.1). It is the **gating test**: replaying a model seat's
pick-requests through the online reconstruction must reproduce that offline state
at every `(seat, pack, pick)`; an end-to-end smoke draft additionally confirms a
model seat completes a pod and its record round-trips through the loader.

# 4. The pick protocol (stdin/stdout, UTF-8)

This is the worker↔supervisor contract and the primary review surface.

Three line types share the worker↔supervisor pipe. All are single lines (no
embedded newlines), UTF-8, terminated by `\n`, flushed immediately by the
sender. Each is a sentinel prefix followed by compact JSON.

| Line | Direction | Channel | Meaning |
|------|-----------|---------|---------|
| `<<DRAFT-PICK-REQUEST>>{…}` | worker → supervisor | worker **stdout** | "Pick for this external seat; here is the pack in hand." |
| `<<DRAFT-PICK-RESPONSE>>{…}` | supervisor → worker | worker **stdin** | "Take this card." |
| `<<DRAFT-EVENT-JSON>>{…}` | worker → supervisor | worker **stdout** | A completed draft transcript (unchanged from gen-1). |

**Strict synchrony (the load-bearing invariant).** The worker processes seats
sequentially within a pick, so **at most one pick-request is ever outstanding**.
The worker writes a request, blocks on stdin until it reads the matching
response, then proceeds; it emits a transcript only when no request is
outstanding. The supervisor's single-threaded read loop therefore sees a clean
stream — every request is immediately answered before the next line is read — so
no worker-side threading and no out-of-order matching are needed. Both sides
flush after every line; because messages are one line each and strictly
alternate, pipe buffers never fill (no deadlock).

Forge's incidental stdout is redirected to stderr in the worker, so the
real-stdout FD carries only sentinel lines (as in gen-1). stderr is piped to a
log file and otherwise ignored.

## 4.1 Pick-request payload

The request carries **only the pack in hand plus routing/recency scalars** — it
is deliberately minimal. Everything else (the seat's pool, what it passed, what
it now knows opponents took, and per-card recency) is reconstructed on the Python
side, which has seen every earlier pack handed to this seat and remembers its own
earlier responses.

```json
{
  "draft_id": "<uuid>",
  "seat": 3,
  "agent": "draft-agent",
  "pod_size": 8,
  "pack_number": 1,
  "pick_number": 5,
  "set_code": "BLB",
  "pack": ["Card A", "Card B", "Card C", "..."]
}
```

- `draft_id` is allocated by the worker **at draft start** and is the same id
  later carried in the transcript.
- `seat` is the 0-based pod seat; `agent` is its label from the mix.
- `pod_size`, `pack_number`, `pick_number` are the live geometry. `pack` is the
  card names remaining in the held pack, in the booster's pick (offset) order.
  Card order within the pack is insignificant to the model.
- `set_code` is informational (logging); card identity drives the model.

## 4.2 Pick-response payload

```json
{
  "draft_id": "<uuid>",
  "seat": 3,
  "pack_number": 1,
  "pick_number": 5,
  "pick": "Card B"
}
```

The worker validates that `draft_id` / `seat` / `pack_number` / `pick_number`
match the outstanding request and that `pick` is a card in the held pack. A
mismatch is a protocol desync and **aborts the draft** (§ 4.3) — the worker
never tries to repair it with a substitute pick.

## 4.3 Failure handling — drop the draft, never substitute

A model seat's recorded picks must always be the policy's **genuine** choices: a
substituted pick would silently teach the next generation a move the model never
made. So the corpus is kept clean by construction — **any fault that prevents a
model seat from making its real pick abandons the entire in-flight draft**. The
draft is not recorded to `drafts.jsonl`, the supervisor logs the error, and the
run continues toward `--n-drafts` (the abandoned draft does not count). No
Forge/first-card/random substitute is ever written. Faults that trigger this:

- A Python-side policy or state-tracking error, or a request whose legal actions
  are *entirely* un-embeddable (not expected with a complete `.npz` cache — note
  that dropping *individual* un-embeddable cards from the action set is normal,
  matching offline training, not a fault).
- A malformed or mismatched `<<DRAFT-PICK-RESPONSE>>` (a protocol desync).
- stdin EOF / a vanished peer.

The worker still **never hangs the pod**: instead of blocking forever or
substituting a pick, it abandons the current draft (emitting no transcript for
it) and is ready for the next. Discarding an in-flight draft this way is exactly
the gen-1 JVM-crash path — nothing partial or substituted leaks into the corpus.
A genuine worker JVM crash is handled identically (in-flight draft discarded,
worker restarted).

Because a *deterministic* fault would abandon every draft and stall progress, the
supervisor surfaces these errors prominently; persistent failure makes the run
visibly produce no records (so the operator investigates) rather than silently
filling the corpus with degraded data.

# 5. CLI

`generate-draft-data` is **extended** (no new subcommand): with no model agents
it is byte-for-byte the gen-1 command; model agents are opt-in.

| Flag | Default | Meaning |
|------|---------|---------|
| `--agent-mix` | `forge-full:6,forge-r30:1,forge-r100:1` | Unchanged grammar. Labels may now also be **model labels** (any label given a checkpoint below). Each of `pod_size` seats samples a label independently, so the model/Forge split varies per draft. |
| `--agent-checkpoint` | _(none; repeatable)_ | `LABEL=PATH` binding a mix label to an agent checkpoint (e.g. `draft-agent=models/draft/agent/latest.pt`). Repeat to pit checkpoints against each other (e.g. `a=…/best_A.pt b=…/best_B.pt`). A bare `PATH` is shorthand for label `draft-agent`. A mix label that is neither a Forge built-in nor bound here ⇒ fail fast. |
| `--pick-mode` | `argmax` | `argmax` (strongest line — for evaluation and high-quality self-play) or `sample` (temperature-scaled softmax, for rollout diversity). |
| `--temperature` | `1.0` | Softmax temperature for `--pick-mode sample`; ignored for `argmax`. |
| `--seed` | _(none → nondeterministic)_ | Seeds the Python-side sampling RNG for reproducible rollouts. Forge-side randomness (boosters, Forge-AI seats) is the JVM's own RNG and is not seeded here. |

All gen-1 flags (`--n-drafts`, `--set`, `--scorer-checkpoint`, `--build-method`,
`--picker-checkpoint`, `--cards-path`, `--output-path`, `--resume`) keep their
meanings.

Example — a pod of one model seat (sampled ~⅛ of seats) against Forge:

```
python -m draft generate-draft-data --n-drafts 500 --set BLB \
  --agent-mix forge-full:7,draft-agent:1 \
  --agent-checkpoint draft-agent=models/draft/agent/latest.pt
```

Example — two checkpoints drafting against each other and Forge:

```
python -m draft generate-draft-data --n-drafts 500 \
  --agent-mix forge-full:4,a:2,b:2 \
  --agent-checkpoint a=models/draft/agent/best_A.pt \
  --agent-checkpoint b=models/draft/agent/best_B.pt \
  --pick-mode sample --temperature 1.2 --seed 7
```

# 6. User-facing output

**Console.** The run logs as gen-1 does — startup configuration, then per-draft
progress toward `--n-drafts` with an ETA. The model-agent additions are visible
at startup (which labels are model-backed, their checkpoint paths, the pick mode
and seed) so the operator can confirm the pod was configured as intended.
Illustrative:

```
generate-draft-data: target 500 drafts, set BLB, resume off
  agent mix: forge-full:7, draft-agent:1
  model agents: draft-agent -> models/draft/agent/latest.pt  (pick-mode=argmax)
  scorer: models/sealed/scorer/latest.pt   builder: picker
draft 1/500  set BLB  model seats [3]  done in 14s   ETA ~1h57m
draft 2/500  set BLB  model seats [0,5] done in 13s  ETA ~1h49m
ERROR pick fault on draft <uuid> seat 5 (pack 2 pick 3): <reason> — draft abandoned, not recorded
draft 3/500  set BLB  model seats [2]  done in 15s   ETA ~1h47m
...
```

(Exact wording is not normative; the requirement is that the operator can see the
target/progress/ETA, which seats each draft model-piloted, and — per § 4.3 — a
prominent error whenever a pick fault abandons a draft. An abandoned draft is not
counted toward `--n-drafts`.)

**Files.** The only persistent output is the appended `drafts.jsonl` (§ 7) plus
the worker's stderr log. No model artifacts are written — the policy loads
existing `models/draft/agent/*.pt` checkpoints and writes nothing under
`models/`.

# 7. Records (`drafts.jsonl`)

Output is the **unchanged** gen-1 `drafts.jsonl`: one self-contained JSON record
per completed draft, `seats[i].agent` carrying the mix label (now possibly a
model label), `seats[i].deck` / `deck_score` from the frozen picker + scorer over
that seat's reconstructed pool, and `boosters` pinning all geometry. A run mixing
model and Forge labels is therefore:

- a **self-play corpus** — concatenate it with prior `drafts.jsonl` and retrain;
  the model's own picks (under its label) become imitation targets only if that
  label is whitelisted via `train-draft-agent --imitation-agents`, while the
  critic trains on every seat regardless;
- a **strength measurement** — within each draft, compare the model seats'
  `deck_score` against the Forge seats' in the same pod (shared boosters cancel
  set/pool luck).

No schema change, so existing readers, the loader, and analysis tooling consume
these records as-is.
