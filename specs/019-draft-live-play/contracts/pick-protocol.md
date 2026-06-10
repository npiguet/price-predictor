# Protocol contract: live pick side-channel (`DraftWorkerMain` ↔ supervisor)

Extends the gen-1 worker protocol
([`specs/018-draft-agent/contracts/worker-protocol.md`](../../018-draft-agent/contracts/worker-protocol.md)),
which is unchanged. This adds a request/response side-channel so the worker can
ask the Python supervisor for a **model-piloted seat's** pick. Authority: design
note §4. All lines are single, newline-free, UTF-8, `\n`-terminated, flushed
immediately.

## Channels

| Line | Direction | FD | Meaning |
|------|-----------|----|---------|
| `<<DRAFT-PICK-REQUEST>>{…}` | worker → supervisor | worker **stdout** | "Pick for this external seat; here is the pack in hand." |
| `<<DRAFT-PICK-RESPONSE>>{…}` | supervisor → worker | worker **stdin** | "Take this card" — or `abort` the draft. |
| `<<DRAFT-ABANDONED>>{…}` | worker → supervisor | worker **stdout** | Worker-detected fault; draft dropped, no transcript. |
| `<<DRAFT-EVENT-JSON>>{…}` | worker → supervisor | worker **stdout** | Completed-draft transcript (gen-1, unchanged). |

Forge's incidental stdout is redirected to stderr in the worker (gen-1
discipline), so the real-stdout FD carries only sentinel lines. Worker stderr is
piped to a per-run log file (FR-015).

## Strict synchrony (load-bearing invariant)

Seats are processed sequentially within a pick, so **at most one pick-request is
ever outstanding**. The worker writes a request, blocks on stdin until it reads
the matching response, then proceeds; it emits a transcript only when no request
is outstanding. The supervisor's single-threaded read loop therefore sees a
clean stream — every request is answered before the next line is read. Both
sides flush after every line; messages alternate one-for-one, so pipe buffers
never fill (no deadlock). No worker-side threading.

## `<<DRAFT-PICK-REQUEST>>` payload (design §4.1)

```json
{
  "draft_id": "<uuid>",
  "seat": 3,
  "agent": "draft-agent",
  "pod_size": 8,
  "pack_number": 1,
  "pick_number": 5,
  "set_code": "BLB",
  "pack": ["Card A", "Card B", "Card C"]
}
```

- `draft_id` is allocated by the worker **at draft start** and is the same id
  later carried in the transcript.
- `pack` is the names remaining in the held pack, in pick (offset) order; card
  order is insignificant to the model.
- `set_code` is informational (logging); card identity drives the model.

## `<<DRAFT-PICK-RESPONSE>>` payload (design §4.2 + abort)

Normal pick:

```json
{"draft_id":"<uuid>","seat":3,"pack_number":1,"pick_number":5,"pick":"Card B"}
```

Supervisor-initiated abandonment (Python-side fault — policy error or all-legal-
actions un-embeddable):

```json
{"draft_id":"<uuid>","seat":3,"pack_number":1,"pick_number":5,"abort":true}
```

Worker validation: `draft_id` / `seat` / `pack_number` / `pick_number` MUST match
the outstanding request, and `pick` (when present) MUST be a card in the held
pack. A mismatch/garbled response is a protocol desync → the worker abandons the
draft (it never repairs with a substitute). On `abort:true` the worker drops the
in-flight draft and continues to the next.

## `<<DRAFT-ABANDONED>>` payload

```json
{"draft_id":"<uuid>","reason":"response mismatch at seat 3 pack 1 pick 5"}
```

Emitted by the worker when it self-detects a fault and can still continue to the
next draft. The supervisor logs it prominently and counts it toward the
consecutive-fault threshold. (stdin EOF abandonment is instead observed as worker
exit and handled by the restart path.)

## Failure handling — drop the draft, never substitute (design §4.3, SC-002)

A model seat's recorded picks are always the policy's genuine choices. **Any**
fault that prevents a genuine pick abandons the entire in-flight draft: no
transcript/record, error logged, run continues toward `--n-drafts` (abandoned
draft not counted). Faults:

- Python-side policy/tracker error, or a request whose legal actions are
  *entirely* un-embeddable (dropping *individual* un-embeddable cards is normal,
  not a fault) → supervisor sends `abort:true`.
- Malformed/mismatched `<<DRAFT-PICK-RESPONSE>>` → worker abandons, emits
  `<<DRAFT-ABANDONED>>`.
- stdin EOF / vanished peer → worker exits; supervisor restarts it (gen-1 crash
  path).

The worker never hangs the pod. Discarding an in-flight draft is exactly the
gen-1 JVM-crash path — nothing partial or substituted leaks into the corpus. A
genuine worker JVM crash is handled identically (in-flight draft discarded,
worker restarted) but does **not** count toward the consecutive-fault auto-abort.

## Consecutive-fault auto-abort (FR-016, SC-008)

The supervisor keeps a consecutive pick-fault counter: incremented on each
abandoned draft (abort-initiated or `<<DRAFT-ABANDONED>>`), **reset to zero on any
completed draft**. When it reaches `--max-consecutive-faults` (default 5) the run
aborts with a nonzero exit and a prominent error, rather than looping
indefinitely on a deterministic fault. Recovered worker crashes do not increment
it.

## Worker JVM system properties (added)

- `-Ddraft.external.agents=<label,label,…>` — the set of mix labels that are
  model-piloted (the worker routes only these seats through the request path).
  Absent/empty ⇒ no requests are ever emitted (byte-for-byte gen-1 behavior).

`-Ddraft.agent.mix` and `-Ddraft.set` keep their gen-1 meanings.
