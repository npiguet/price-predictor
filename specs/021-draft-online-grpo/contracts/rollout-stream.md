# Contract: resident rollout stream (`iter_records`) + required-agent property

**Feature**: 021-draft-online-grpo | Extends
[`specs/019-draft-live-play/contracts/pick-protocol.md`](../../019-draft-live-play/contracts/pick-protocol.md)
and [`specs/018-draft-agent/contracts/worker-protocol.md`](../../018-draft-agent/contracts/worker-protocol.md),
both **unchanged**. Authority: [spec.md](../spec.md) FR-003, FR-005, FR-012,
FR-029; research D1, D2, D3.

Two additions let the online loop keep one Forge draft worker alive across
rounds and guarantee every pod carries a learner seat.

## 1. `GenerateDraftDataSupervisor.iter_records(launch, labeler)`

```python
def iter_records(
    self, launch: Callable[[], Popen], labeler: DeckLabeler,
) -> Iterator[DraftRecord]
```

An **endless** generator over completed, labeled draft records.

| Property | Contract |
|---|---|
| Worker lifetime | Launches a worker on first `next()`. Keeps it alive across yields. On worker exit (crash or stdin EOF) it logs, relaunches, and continues — no records are lost beyond the in-flight draft. |
| Suspension | While the consumer is not pulling, the generator is suspended and the worker stays resident. This **is** the resident-worker mechanism (FR-005); there is no separate "pause" call. |
| Back-pressure | Safe by construction: the pick protocol is strictly synchronous and every pod has ≥1 learner seat (§2), so the worker blocks on stdin at the first learner pick of the next draft as soon as reading stops. It can neither run ahead nor fill the pipe. |
| Round boundary | Because the worker blocks *before* any learner pick of the next draft, no draft ever straddles a weight swap: every learner pick of a draft is answered by one policy version. |
| Pick routing | Identical to today: `<<DRAFT-PICK-REQUEST>>` answered from the registry, `<<DRAFT-ABANDONED>>` logged + counted, malformed requests force a worker restart, `MaxConsecutiveFaultsError` propagates out of the generator. |
| Shutdown | `close()` (or garbage collection) runs the `finally` that terminates the worker process tree. The installed SIGINT/SIGTERM handler also stops the loop. |
| Yield unit | A fully assembled `DraftRecord` — pools reconstructed, all seats built and scored by `labeler`, `run_id`/`timestamp` set. Nothing is written to disk by the generator; the consumer owns persistence. |

`GenerateDraftDataSupervisor.run()` is refactored to consume this generator
(append + count + progress + `--n-drafts` target + resume). Its observable
behaviour — including the existing tests in
`tests/integration/test_generate_draft_data.py`,
`tests/integration/test_draft_live_play.py`,
`tests/integration/test_draft_supervisor_restart.py`, and
`tests/unit/draft/test_supervisor_pick_routing.py` — is unchanged.

## 2. Live-model pick services

`AgentPickService.from_model(model, config, locator, *, device, pick_mode,
temperature, seed)` builds a service around an **already-constructed**
`DraftAgentModel` instead of loading a checkpoint path.

| Property | Contract |
|---|---|
| Ownership | The caller owns the model: `from_model` does **not** call `.eval()`, `.train()`, or `.to(device)`. The online trainer sets `eval()` for generation and `train()` for the update. |
| Freshness | The service holds the model by reference, so a weight update is visible to the very next pick with no copy or explicit push (FR-012). |
| Everything else | Identical to the path-loading constructor: per-`(draft_id, seat)` `OnlineDraftStateTracker`, `PACK`-masked logits, `argmax`/seeded-`sample` selection, `PickFault` on any condition preventing a genuine pick. |

`AgentRegistry.build(agent_checkpoints, mix_labels, *, preloaded=None, …)` gains
an optional `preloaded: dict[str, AgentPickService]`. Preloaded labels count as
**bound** during FR-003 label validation and go through the same geometry checks
(`config.packs == PACKS`, `config.P ≥ pack_size` when known). The online trainer
passes `{learner_label: live_service}` and `--frozen` as `agent_checkpoints`.

The two model-piloted categories are served in **different** pick modes
(research D5): the learner in `pick_mode="sample"` at its `--agent-temp` value,
since sampling is its only exploration mechanism; every frozen agent in
`pick_mode="argmax"`, its best play. A sampled frozen agent would pass downstream
the cards it wrongly declined, feeding the learner a field that plays worse than
it can — a training-environment distortion, not merely a margin offset.

The online trainer therefore builds **all** services itself and passes them via
`preloaded`, leaving `agent_checkpoints` empty; `AgentRegistry.build`'s single
`pick_mode` argument is not used to configure them.

## 3. Java worker: `-Ddraft.required.agent`

New optional JVM system property on `DraftWorkerMain`, forwarded by
`DraftWorkerConnector.start(..., required_agent=<label>)`.

```
-Ddraft.required.agent=<label>
```

| Case | Behaviour |
|---|---|
| Absent or blank | Unchanged: each of the 8 seats draws independently from `-Ddraft.agent.mix`. |
| Present | After the per-seat draw, if **no** seat carries `<label>`, one uniformly-chosen seat index is overwritten with `<label>`. The rest of the draw is untouched. |

The rewrite happens before any pick, so the transcript's `seats[].agent` array —
and therefore the recorded corpus — always reflects the realised pod. This is the
mechanism behind FR-003's "every generated draft MUST contain at least one
learner seat"; a learner-free pod is never played.

**Rebuild required**: `cd forge-connector && mvn package -DskipTests` before the
first gen-3 run.
