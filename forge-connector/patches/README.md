# Forge engine patches

The patches in this directory are applied to the **sibling `../forge` checkout**, not to anything in
this repository. They add the hooks the effect-record collectors need for exact attribution; without
them the collectors still run, fall back to bracket attribution, and stamp every record
`"mode": "degraded"`.

**They must be re-applied after every Forge upgrade.** Forge is rebuilt independently of this repo and
an upgrade silently reverts them, which is why attribution mode is detected at worker startup rather
than being a build flag: a lapsed patch mislabels a corpus otherwise.

## Applying

```bash
cd ../forge
git apply ../price-predictor/forge-connector/patches/*.patch
mvn install -DskipTests
```

## Verifying

Run a short collection and check the `mode` field of any record:

```bash
python -m sealed match-outcomes --effect-records output/effects/records/
```

Records carry `"mode": "patched"` when the hooks are present and `"mode": "degraded"` when they are
not. The worker prints the detected mode at startup.

## The patch set

Each file is documented at the top of the patch itself; the set is listed here so a re-apply after an
upgrade can be checked for completeness.

| Patch | Hook |
|---|---|
| `01-trigger-cause.patch` | trigger-handler cause channel, plus `AbilityKey.Cause` at the `Destroyed` firing site |
| `02-replacement-hook.patch` | replacement-execution-point hook (parameter map deep-copied before the call) |
| `03-subability-pointer.patch` | threaded currently-resolving-sub-ability pointer |
| `04-logging-points.patch` | trigger-fire condition evaluation, and the AI's candidate computation, combat-setup legality, and legality/cost-adjustment checks |
