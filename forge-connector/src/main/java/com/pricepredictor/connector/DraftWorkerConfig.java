package com.pricepredictor.connector;

import java.util.Set;

/**
 * What the worker's system properties say, read once at start-up.
 *
 * @param agentMixSpec   the raw {@code draft.agent.mix} value
 * @param setCode        the one set every draft uses, or null for a random one per draft
 * @param externalAgents the mix labels routed through the pick side-channel
 * @param requiredAgent  the label every pod must carry, or null for none
 */
record DraftWorkerConfig(
        String agentMixSpec, String setCode, Set<String> externalAgents,
        String requiredAgent) {

    static DraftWorkerConfig fromProperties() {
        return new DraftWorkerConfig(
                WorkerProperties.require(DraftWorkerMain.AGENT_MIX),
                WorkerProperties.optional(DraftWorkerMain.SET).orElse(null),
                Set.copyOf(WorkerProperties.listProperty(DraftWorkerMain.EXTERNAL_AGENTS)),
                WorkerProperties.optional(DraftWorkerMain.REQUIRED_AGENT).orElse(null));
    }
}
