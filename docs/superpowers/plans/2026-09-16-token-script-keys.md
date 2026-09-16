# Token Script Keys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the collector resolve every token's abilities to its `tokenscripts/` file, including tokens rebuilt by `GameCopier` in forked games, and audit every other place the collector touches tokens so this is the last token-path fix.

**Architecture:** `ProvenanceKey.scriptFileOf` already asks `tokenScriptStem(host)` first. That method only recognises a host whose paper card is a `PaperToken`, which a forked game's tokens no longer have. The copied card keeps its image key, and a token image key is `t:<script stem>[|edition|…]`, so the stem is recoverable from it and checkable against Forge's token database. One resolver, one fallback order, and a test built the way `TokenInfo.toCard` builds the copy.

**Tech Stack:** Java 17, Maven, JUnit 5 with the connector's `ForgeExtension`; Forge's `forge-core` (`ImageKeys`, `StaticData`, `TokenDb`) and `forge-game` (`Card`, `TokenInfo`).

**Spec:** `docs/superpowers/specs/2026-09-16-corpus-and-trainer-rework.md` (FR-150) over `specs/023-ability-effect-model/spec.md` § provenance.

## Global Constraints

- The connector compiles against the sibling Forge checkout at `../forge` (built with `mvn install -DskipTests`); run its tests with `cd forge-connector && mvn test -Dtest=<Class>`.
- A provenance key must name a file the converted tree holds, or fail loudly; no new fabricated paths (`SourceTree` class doc).
- Token scripts stay in `tokenscripts/`, never derived under `cardsfolder/` (FR-096).
- Before editing any file under `specs/` or `experiments/`, load the `feature-workflow` skill: those files carry binding wording and structure rules.

---

## File map

| File | Responsibility |
|---|---|
| Modify `forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceKey.java` | `tokenScriptStem` reads the image key when there is no `PaperToken`; verified against `TokenDb` |
| Modify `forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceKeyTest.java` | tests for the copied-token shape |
| Modify `forge-connector/src/test/java/com/pricepredictor/connector/effects/TestCards.java` | `copiedToken(scriptStem)` helper |
| Create `docs/superpowers/notes/2026-09-16-token-access-audit.md` | the audit table (Task 3) |
| Modify `src/effects/application/training_loop.py` (Python) | the unresolved-scripts log line names the tree it expected |

---

### Task 1: A copied token keys into the token tree

**Files:**
- Modify: `forge-connector/src/test/java/com/pricepredictor/connector/effects/TestCards.java`
- Modify: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceKeyTest.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceKey.java:603-624`

**Interfaces:**
- Test helper `TestCards.copiedToken(String scriptStem) -> Card`: a card shaped the way `TokenInfo.toCard` shapes a copy: `new Card(id, game)`, `setName(original.getName())`, `setImageKey(original.getImageKey())`, `setGamePieceType(GamePieceType.TOKEN)`, types, P/T; paper card **not** set.
- `ProvenanceKey.tokenScriptStem(Card)` stays private; `scriptFileOf(Card)` stays package-private and is what the tests call.

- [ ] **Step 1: Write the failing tests**

In `TestCards.java`, after `token(String scriptStem)`:

```java
    /**
     * A token the way {@code GameCopier} rebuilds one in a forked game:
     * {@code TokenInfo.toCard} makes a bare {@code Card}, copies the name,
     * image key, colour, types and P/T, and never sets a paper card. Every
     * ability-bearing token in a forked combat record arrives like this.
     */
    static Card copiedToken(String scriptStem) {
        Card original = token(scriptStem);
        Card copy = new Card(nextCardId++, GAME);
        copy.setName(original.getName());
        copy.setImageKey(original.getImageKey());
        copy.setGamePieceType(GamePieceType.TOKEN);
        for (forge.card.CardType.CoreType type : original.getType().getCoreTypes()) {
            copy.addType(type.toString());
        }
        for (String subtype : original.getType().getSubtypes()) {
            copy.addType(subtype);
        }
        copy.setBasePower(original.getBasePower());
        copy.setBaseToughness(original.getBaseToughness());
        return copy;
    }
```

Add `import forge.card.GamePieceType;` to `TestCards.java`. `CardType.getCoreTypes()` and `getSubtypes()` are the two accessors the sibling Forge exposes (`forge-core/src/main/java/forge/card/CardType.java:295-303`); the types are not what the assertions read.

In `ProvenanceKeyTest.java`, in the `// ── tokens ──` section:

```java
    /**
     * A forked game's tokens are rebuilt by {@code TokenInfo}, which drops the
     * {@code PaperToken}. The first corpus keyed those to
     * {@code cardsfolder/z/zombie_token.txt}, a file no tree holds, so every
     * Food, Treasure and Zombie on a forked board reached the model as no text.
     * The image key survives the copy and names the script stem (FR-150).
     */
    @Test
    void aCopiedTokenWithNoPaperCardStillKeysIntoTheTokenTree() {
        Card copy = TestCards.copiedToken("c_a_food_sac");

        assertNull(copy.getPaperCard(), "the fixture must reproduce the copier's shape");
        assertEquals("tokenscripts/c_a_food_sac.txt", ProvenanceKey.scriptFileOf(copy));
    }

    /** And a real card's copy is not mistaken for a token: no image-key path for it. */
    @Test
    void aCopiedPrintedCardStillKeysIntoTheCardTree() {
        Card bolt = card("Lightning Bolt");
        Card copy = new Card(TestCards.nextCardId(), TestCards.game());
        copy.setName(bolt.getName());
        copy.setImageKey(bolt.getImageKey());

        assertEquals("cardsfolder/l/lightning_bolt.txt", ProvenanceKey.scriptFileOf(copy));
    }

    /** A token image key the database does not know is not turned into a path. */
    @Test
    void anUnknownTokenImageKeyFallsThroughToNull() {
        Card copy = TestCards.copiedToken("c_a_food_sac");
        copy.setImageKey(forge.ImageKeys.getTokenKey("no_such_token_xyz"));

        // Falls through to the name-derived cardsfolder path, exactly as before
        // this change; what must not happen is a fabricated tokenscripts path.
        assertEquals("cardsfolder/f/food_token.txt", ProvenanceKey.scriptFileOf(copy));
    }
```

The name-derived expectation in the last test depends on the Food token's printed name (`Food Token` in Forge's `c_a_food_sac.txt`); check the `Name:` line and adjust the stem if it differs.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd forge-connector && mvn -q test -Dtest=ProvenanceKeyTest`
Expected: `aCopiedTokenWithNoPaperCardStillKeysIntoTheTokenTree` FAILS with `expected: <tokenscripts/c_a_food_sac.txt> but was: <cardsfolder/f/food_token.txt>`; the other two pass already.

- [ ] **Step 3: Read the stem from the image key**

Replace `tokenScriptStem` in `ProvenanceKey.java`:

```java
    /**
     * A token's script-file stem, or null when the card did not come from a
     * token script — including a token copy of a printed card.
     *
     * <p>Two sources, tried in order. A {@link PaperToken} is the token as
     * Forge read it from {@code tokenscripts}, and its image filename starts
     * with the script stem. A token that was rebuilt by {@code TokenInfo} —
     * every token in a {@code GameCopier} fork — has no paper card at all, but
     * {@code TokenInfo.toCard} copies the image key, and a token image key is
     * {@code t:<stem>[|edition|…]}. The stem read either way is checked
     * against the token database, so a key that names no script (a token copy
     * of a printed card carries the printed card's image key, which is not a
     * token key) yields null rather than a fabricated path.
     */
    private static String tokenScriptStem(Card host) {
        try {
            IPaperCard paper = host.getPaperCard();
            String image = null;
            if (paper instanceof PaperToken token) {
                image = token.getImageFilename(1);
            } else if (host.isToken()) {
                image = ImageKeys.getTokenImageName(host.getImageKey());
            }
            if (image == null || image.isEmpty()) {
                return null;
            }
            int bar = image.indexOf('|');
            String stem = (bar < 0 ? image : image.substring(0, bar)).replace(' ', '_');
            if (stem.isEmpty() || !StaticData.instance().getAllTokens().containsRule(stem)) {
                return null;
            }
            return stem;
        } catch (RuntimeException e) {
            return null;
        }
    }
```

Add `import forge.ImageKeys;` and `import forge.StaticData;` if the file lacks them. `Card.getImageKey()` may return null for a bare card; `ImageKeys.getTokenImageName(null)` would throw, and the `catch` returns null, which is the intended answer for that shape.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd forge-connector && mvn -q test -Dtest=ProvenanceKeyTest`
Expected: all pass, including the two existing token tests (`aTokensAbilityKeysIntoTheTokenTree`, `aStackCopyOfATokensAbilityKeysIntoTheTokenTree`), which now go through the `containsRule` check as well.

- [ ] **Step 5: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceKey.java forge-connector/src/test/java/com/pricepredictor/connector/effects/
git commit -m "fix(connector): key a copied token's abilities into the token tree via its image key (FR-150)"
```

---

### Task 2: Prove it end to end on a forked record

**Files:**
- Modify: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ForkCollectorTest.java`

The unit test in Task 1 reproduces the copier's shape by hand. This task checks the real path once: a fork of a game with a Food token on the battlefield, and the token's printed line in the fork's snapshot keyed into `tokenscripts/`.

- [ ] **Step 1: Find the existing fork fixture**

Run: `grep -n "GameCopier\|void .*fork\|@Test" forge-connector/src/test/java/com/pricepredictor/connector/effects/ForkCollectorTest.java | head -30`

Pick the test that builds a two-player game and forks it (the class doc at `ForkCollectorTest.java:1-40` names it). Its setup is what the new test reuses.

- [ ] **Step 2: Write the failing test**

Using that fixture's helpers (names will differ; the shape is what matters):

```java
    /**
     * The whole path, not the shape: a Food token on the battlefield of a
     * forked game keys its printed line into {@code tokenscripts}, so a forked
     * combat record's context entities carry text (FR-150).
     */
    @Test
    void aTokenOnAForkedBoardKeysIntoTheTokenTree() {
        Game game = twoPlayerGame();                       // the fixture's helper
        Player owner = game.getPlayers().get(0);
        Card food = CardFactory.getCard(
                StaticData.instance().getAllTokens().getToken("c_a_food_sac"), owner, game);
        game.getAction().moveToPlay(food, null, null);

        Game fork = new GameCopier(game).makeCopy();
        Card copied = fork.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(food.getName()))
                .findFirst().orElseThrow();

        assertNull(copied.getPaperCard(), "GameCopier rebuilds tokens without a paper card");
        String snapshot = new SnapshotBuilder(fork).entityJsonFor(copied);   // or the builder's real entry point
        assertTrue(snapshot.contains("\"script_file\":\"tokenscripts/c_a_food_sac.txt\""),
                snapshot);
        assertFalse(snapshot.contains("cardsfolder/f/food_token.txt"), snapshot);
    }
```

Use whichever `SnapshotBuilder` entry point the fixture already calls to render a state; the assertion is on the `printed[].script_file` of the copied token. If `GameCopier.makeCopy()` has a different name in the sibling Forge, read `forge-ai/src/main/java/forge/ai/simulation/GameCopier.java` for the public method that returns the copy.

- [ ] **Step 3: Run it against the pre-fix code to confirm it catches the bug**

Temporarily `git stash` the Task 1 change, run `mvn -q test -Dtest=ForkCollectorTest#aTokenOnAForkedBoardKeysIntoTheTokenTree`, and confirm it fails on the `tokenscripts` assertion. Then `git stash pop`.

- [ ] **Step 4: Run it against the fix**

Run: `cd forge-connector && mvn -q test -Dtest=ForkCollectorTest`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add forge-connector/src/test/java/com/pricepredictor/connector/effects/ForkCollectorTest.java
git commit -m "test(connector): a token on a forked board keys into the token tree"
```

---

### Task 3: Audit every token access in the collector

**Files:**
- Create: `docs/superpowers/notes/2026-09-16-token-access-audit.md`
- Possibly modify: files the audit finds, each as its own commit

- [ ] **Step 1: Enumerate the sites**

Run from `price-predictor/`:

```bash
grep -rnE 'isToken\(|PaperToken|tokenscripts|TOKENSCRIPTS|getAllTokens|TokenInfo|token_script_id|_token' \
    forge-connector/src/main/java --include=*.java | grep -vE '^\s*//|^\S+:\s*\*'
grep -rnE 'tokenscripts|token_script|_token\.txt' src/effects --include=*.py | grep -v tests
```

- [ ] **Step 2: Classify each site in the note**

Write the note as a table with one row per hit: file and line, what it does with the token, whether it goes through `ProvenanceKey.scriptFileOf` (or the Python `SidecarCache` with a `tokenscripts` root), and a verdict: `shared resolver`, `consistent by construction`, or `needs change`. Known rows to expect:

| site | verdict expected |
|---|---|
| `ProvenanceKey.scriptFileOf` / `tokenScriptStem` | the shared resolver (Task 1) |
| `SnapshotBuilder.entityToJson` `token_script_id` = `card.getName()` | **needs decision**: the created-objects head (FR-078) vocabulary is keyed by token-script id, and a printed name ("Zombie Token") collapses `b_2_2_zombie`, `b_2_2_zombie_decayed` and `b_x_x_zombie`. Changing it to `tokenScriptStem(card)` changes the corpus schema and the vocab; record the recommendation and leave the change to a separate plan |
| `SourceTree.TOKENSCRIPTS` layout, `CardFilenames.scriptFileForStem` flat-tree branch | consistent by construction |
| `MatchGenerator` / `PoolGenerator` `isBasicLand` filters | not token-related; note and move on |
| Python `SidecarCache` roots `{"cardsfolder", "tokenscripts"}` and every `cards_folders` default | consistent; confirm every CLI default lists `output/tokenscripts/` (FR-096) |

- [ ] **Step 3: Fix what is cheap, flag what is not**

For every `needs change` row that is a one-line consistency fix (a default missing `output/tokenscripts/`, a site deriving a token path from a name instead of calling `scriptFileOf`), make the change with a test and commit it separately. For the `token_script_id` row, write the recommendation in the note and stop.

- [ ] **Step 4: Commit the note**

```bash
git add docs/superpowers/notes/2026-09-16-token-access-audit.md
git commit -m "docs(connector): audit of every token-script access in the collector"
```

---

### Task 4: Make the unresolved-script log say which tree it looked in

**Files:**
- Modify: `src/effects/application/training_loop.py:680-691`
- Test: none (a log line)

The line that caught this bug reads `288 ability scripts the converted corpus does not hold … Most asked: cardsfolder/z/zombie_token.txt (1108)`. It should say what would have resolved, so the next mis-key is diagnosable from the log alone.

- [ ] **Step 1: Change the message**

```python
                logger.info(
                    "%d ability scripts the converted corpus does not hold, "
                    "%d lookups so far; those abilities reach the model as no "
                    "text. Most asked: %s. A `_token` stem under cardsfolder/ "
                    "is a token keyed to the wrong tree; a variant-scripts/ key "
                    "needs --variant-scripts.",
                    len(sidecars.unresolved),
                    sum(sidecars.unresolved.values()),
                    ", ".join(f"{name} ({hits})" for name, hits in worst),
                )
```

- [ ] **Step 2: Run the effects suite**

Run: `python -m pytest tests/unit/effects -q`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add src/effects/application/training_loop.py
git commit -m "chore(effects): the unresolved-script log names the likely mis-key"
```

---

### Task 5: Verify on the corpus

**Files:** none (operational). After rebuilding the connector JAR (`cd forge-connector && mvn package -DskipTests`) and re-collecting at least one forked-combat shard, or after the next full collection.

- [ ] **Step 1: Count token keys under the wrong tree**

```bash
python - <<'EOF'
import gzip, glob, re
from collections import Counter
hits = Counter()
for p in glob.glob('output/effects/records/**/*.jsonl.gz', recursive=True)[-20:]:
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            for m in re.findall(r'"script_file":\s*"(cardsfolder/[a-z]/[a-z0-9_]+_token\.txt)"', line):
                hits[m] += 1
print(hits.most_common(10) or "no token keys under cardsfolder/")
EOF
```

Expected on shards collected with the fixed JAR: `no token keys under cardsfolder/`. On the old shards the top entries are the Zombie, Food, Treasure and Warrior tokens; leave those shards alone, `build-corpus` reads them as they are and the trainer's unresolved-script line will keep reporting them until they are re-collected.

- [ ] **Step 2: Record**

Add the before and after counts to the audit note from Task 3.

---

## Self-review

- **Spec coverage.** FR-150 (Tasks 1, 2); the audit the design record asks for (Task 3); log diagnosability (Task 4); verification (Task 5).
- **Type consistency.** `TestCards.copiedToken(String)` is used by `ProvenanceKeyTest`; `ProvenanceKey.scriptFileOf(Card)` is the existing package-private static; `tokenScriptStem` keeps its signature.
- **Risk.** `containsRule` is on `TokenDb`, not on the `ITokenDatabase` interface; `StaticData.getAllTokens()` returns `TokenDb`, so the call compiles. If the sibling Forge's `getAllTokens()` return type is the interface, fall back to `getAllTokens().getToken(stem) != null`.
