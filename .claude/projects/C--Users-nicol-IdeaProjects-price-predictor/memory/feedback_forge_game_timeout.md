---
name: Forge game timeout crashes JVM
description: Excessively long Forge AI games crash the worker JVM rather than hanging — no timeout mechanism needed, crash recovery handles it
type: feedback
---

Don't add game-level timeout mechanisms for Forge AI workers. When a game takes too long, the worker JVM crashes on its own, so the supervisor's crash-restart mechanism already covers this case.

**Why:** User's prior experience running Forge AI games shows this is the consistent behavior — long games lead to JVM crashes, not hangs.

**How to apply:** When designing Forge AI worker pipelines, rely on process-level crash recovery rather than adding per-game timeouts.
