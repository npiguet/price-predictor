package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.DraftPack;
import forge.gamemodes.limited.LimitedPlayerAI;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;

/**
 * Draft worker entry point.
 *
 * <p>Drives Forge's draft AI for all pod seats and streams one completed-draft
 * transcript per draft on stdout. The Python supervisor
 * ({@code draft.application.generate_draft_data}) reads these lines, builds and
 * scores a deck per seat, and appends one self-contained JSON record to
 * {@code drafts.jsonl}.
 *
 * <p>Transport (worker-protocol.md): one flushed line per completed draft,
 * <pre>{@code <<DRAFT-EVENT-JSON>>{"draft_id":..,"boosters":[..],"seats":[{"agent":..}]}}</pre>
 * Diagnostics and Forge's incidental logging go to stderr.
 *
 * <p>System properties:
 * <ul>
 *   <li>{@code -Ddraft.agent.mix=<label:weight,...>} — required; each seat draws
 *       its agent independently per draft (FR-006).</li>
 *   <li>{@code -Ddraft.set=<CODE>} — optional; restrict every draft to one set.
 *       When absent, a random sealed-legal set is chosen per draft (FR-009).</li>
 * </ul>
 *
 * <p>The worker loops forever; the supervisor terminates it once it has enough
 * drafts (or on crash, restarts a fresh one).
 */
public class DraftWorkerMain {

    static final String SENTINEL = "<<DRAFT-EVENT-JSON>>";
    static final int POD_SIZE = 8;
    static final int PACKS = 3;

    public static void main(String[] args) {
        String agentMixSpec = System.getProperty("draft.agent.mix");
        if (agentMixSpec == null || agentMixSpec.isBlank()) {
            System.err.println("Error: -Ddraft.agent.mix system property is required");
            System.exit(2);
        }
        String setCode = System.getProperty("draft.set");  // null => random per draft

        AgentMix mix = AgentMix.parse(agentMixSpec);

        System.err.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.err.println("Forge initialized. Starting draft generation.");

        // UTF-8 stdout: Forge card names carry accented characters the Python
        // side reads as UTF-8.
        PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
        Random random = MyRandom.getRandom();

        List<String> eligibleSets = null;
        if (setCode == null) {
            eligibleSets = MatchGenerator.computeEligibleSets();
            if (eligibleSets.isEmpty()) {
                System.err.println("Error: no eligible sealed-legal sets found");
                System.exit(1);
            }
        }

        // Shared context object for the AI seats (used only by conspiracy-card
        // hooks, which standard sets do not trigger).
        BoosterDraft context = BoosterDraft.createDraft(LimitedPoolType.Full);

        long count = 0;
        while (true) {
            try {
                String draftSet = (setCode != null)
                        ? setCode
                        : eligibleSets.get(random.nextInt(eligibleSets.size()));
                String line = generateDraft(context, draftSet, mix, random);
                if (line != null) {
                    out.println(line);
                    out.flush();
                    count++;
                    if (count % 10 == 0) {
                        System.err.println("Worker: " + count + " drafts generated");
                    }
                }
            } catch (Exception e) {
                System.err.println("Error generating draft: " + e.getMessage());
                e.printStackTrace(System.err);
                // Continue on non-fatal errors; fatal errors propagate and the
                // supervisor restarts the worker.
            }
        }
    }

    /**
     * Run one full pod draft and return its sentinel-prefixed transcript line,
     * or {@code null} if the draft could not be produced uniformly (skipped).
     */
    static String generateDraft(
            BoosterDraft context, String setCode, AgentMix mix, Random random) {
        SealedTemplate template = boosterTemplate(setCode);
        if (template == null) {
            return null;
        }

        // Generate all boosters: booster index k = (pack-1) * POD_SIZE + openingSeat.
        DraftPack[] boosters = new DraftPack[PACKS * POD_SIZE];
        int packSize = -1;
        for (int k = 0; k < boosters.length; k++) {
            List<PaperCard> cards = new ArrayList<>(new UnOpenedProduct(template).get());
            if (packSize == -1) {
                packSize = cards.size();
            } else if (cards.size() != packSize) {
                // Non-uniform pack size breaks the reconstruction geometry; skip.
                return null;
            }
            boosters[k] = new DraftPack(cards, k);
        }
        if (packSize <= 0) {
            return null;
        }

        // One AI seat per pod position; agents sampled per draft (FR-006).
        List<LimitedPlayerAI> players = new ArrayList<>(POD_SIZE);
        String[] agents = new String[POD_SIZE];
        for (int s = 0; s < POD_SIZE; s++) {
            players.add(new LimitedPlayerAI(s, context));
            agents[s] = mix.sample(random);
        }

        List<List<String>> picksByBooster = new ArrayList<>(boosters.length);
        for (int k = 0; k < boosters.length; k++) {
            picksByBooster.add(new ArrayList<>(packSize));
        }

        for (int pack = 1; pack <= PACKS; pack++) {
            int dir = (pack % 2 == 1) ? 1 : -1;  // pass left on odd packs, right on even
            DraftPack[] held = new DraftPack[POD_SIZE];
            for (int o = 0; o < POD_SIZE; o++) {
                held[o] = boosters[(pack - 1) * POD_SIZE + o];
            }
            for (int pick = 1; pick <= packSize; pick++) {
                DraftPack[] nextHeld = new DraftPack[POD_SIZE];
                for (int s = 0; s < POD_SIZE; s++) {
                    DraftPack current = held[s];
                    LimitedPlayerAI player = players.get(s);
                    player.receiveOpenedPack(current);  // make it the player's choice pack
                    PaperCard chosen = decidePick(player, current, agents[s], random);
                    picksByBooster.get(current.getId()).add(chosen.getName());
                    player.draftCard(chosen);
                    nextHeld[s] = player.passPack();    // pull the pack back out
                }
                for (int s = 0; s < POD_SIZE; s++) {
                    held[(s + dir + POD_SIZE) % POD_SIZE] = nextHeld[s];
                }
            }
        }

        return SENTINEL + toJson(UUID.randomUUID().toString(), setCode, boosters,
                picksByBooster, agents);
    }

    private static SealedTemplate boosterTemplate(String setCode) {
        SealedTemplate template = StaticData.instance().getBoosters().get(setCode);
        if (template != null) {
            return template;
        }
        CardEdition edition = StaticData.instance().getEditions().get(setCode);
        return edition == null ? null : edition.getBoosterTemplate("Draft");
    }

    /**
     * Decide a seat's pick: the Forge AI's choice ({@code forge-full}), or a
     * uniform-random legal card for the random-override agents
     * ({@code forge-r30} replaces 30% of picks, {@code forge-r100} all of them).
     */
    static PaperCard decidePick(
            LimitedPlayerAI player, DraftPack pack, String agent, Random random) {
        double randomFraction = randomOverrideFraction(agent);
        if (randomFraction > 0 && random.nextDouble() < randomFraction) {
            return pack.get(random.nextInt(pack.size()));
        }
        PaperCard aiPick = player.chooseCard();
        if (aiPick == null) {
            // Defensive: AI declined (shouldn't happen with a non-empty pack).
            return pack.get(random.nextInt(pack.size()));
        }
        return aiPick;
    }

    static double randomOverrideFraction(String agent) {
        return switch (agent) {
            case "forge-r30" -> 0.30;
            case "forge-r100" -> 1.0;
            default -> 0.0;  // forge-full and any unknown label: pure AI
        };
    }

    private static String toJson(
            String draftId, String setCode, DraftPack[] boosters,
            List<List<String>> picksByBooster, String[] agents) {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("{\"draft_id\":\"").append(draftId).append("\",\"boosters\":[");
        for (int k = 0; k < boosters.length; k++) {
            if (k > 0) {
                sb.append(',');
            }
            sb.append("{\"set_code\":\"").append(jsonEscape(setCode)).append("\",\"picks\":[");
            List<String> picks = picksByBooster.get(k);
            for (int j = 0; j < picks.size(); j++) {
                if (j > 0) {
                    sb.append(',');
                }
                sb.append('"').append(jsonEscape(picks.get(j))).append('"');
            }
            sb.append("]}");
        }
        sb.append("],\"seats\":[");
        for (int s = 0; s < agents.length; s++) {
            if (s > 0) {
                sb.append(',');
            }
            sb.append("{\"agent\":\"").append(jsonEscape(agents[s])).append("\"}");
        }
        sb.append("]}");
        return sb.toString();
    }

    static String jsonEscape(String value) {
        StringBuilder sb = new StringBuilder(value.length() + 8);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    /** Weighted categorical over agent labels, parsed from {@code label:weight,...}. */
    static final class AgentMix {
        private final String[] labels;
        private final int[] cumulative;
        private final int total;

        private AgentMix(String[] labels, int[] cumulative, int total) {
            this.labels = labels;
            this.cumulative = cumulative;
            this.total = total;
        }

        static AgentMix parse(String spec) {
            String[] tokens = spec.split(",");
            List<String> labels = new ArrayList<>();
            List<Integer> weights = new ArrayList<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.isEmpty()) {
                    continue;
                }
                int colon = token.indexOf(':');
                if (colon <= 0 || colon != token.lastIndexOf(':')) {
                    throw new IllegalArgumentException(
                            "agent-mix entry must be 'label:weight', got: " + token);
                }
                String label = token.substring(0, colon).trim();
                int weight = Integer.parseInt(token.substring(colon + 1).trim());
                if (label.isEmpty() || weight < 1) {
                    throw new IllegalArgumentException(
                            "agent-mix entry must have a non-empty label and weight >= 1: " + token);
                }
                labels.add(label);
                weights.add(weight);
            }
            if (labels.isEmpty()) {
                throw new IllegalArgumentException("agent-mix is empty: " + spec);
            }
            int[] cumulative = new int[labels.size()];
            int running = 0;
            for (int i = 0; i < weights.size(); i++) {
                running += weights.get(i);
                cumulative[i] = running;
            }
            return new AgentMix(labels.toArray(new String[0]), cumulative, running);
        }

        String sample(Random random) {
            int roll = random.nextInt(total);
            for (int i = 0; i < cumulative.length; i++) {
                if (roll < cumulative[i]) {
                    return labels[i];
                }
            }
            return labels[labels.length - 1];  // unreachable
        }
    }
}
