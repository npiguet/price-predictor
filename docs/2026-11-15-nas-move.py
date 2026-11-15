"""Reorganise the NAS archive onto the artifact-layout tree, by rename only.

Plan built from docs/2026-11-15-nas-layout-map.md. Every move is os.rename (server-side on SMB);
nothing is copied. Map "delete" items go to _to-delete/ for the user to review.

  python -I -u nas_move.py plan      # build + validate the plan, write plan.tsv, move nothing
  python -I -u nas_move.py execute   # validate again, then rename, journaling each move
"""
import os, re, sys, json

ROOT = r"Y:\Nicolas\mtg\mtg-models-data"
HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_TSV = os.path.join(HERE, "nas_move_plan.tsv")
JOURNAL = os.path.join(HERE, "nas_move_journal.tsv")
HOLD = "_to-delete"

moves = []      # (src_rel, dst_rel)
covered = set() # src_rel of every file or dir covered (a dir covers its subtree)
problems = []


def P(rel):
    return os.path.join(ROOT, rel.replace("/", "\\"))


def ls(rel):
    return sorted(os.listdir(P(rel)))


def isdir(rel):
    return os.path.isdir(P(rel))


def mv(src, dst):
    src = src.replace("/", "\\"); dst = dst.replace("/", "\\")
    if not os.path.exists(P(src)):
        problems.append(f"missing source: {src}")
        return
    moves.append((src, dst))
    covered.add(src)


def mv_into(src, dst_dir):
    mv(src, dst_dir.rstrip("/") + "/" + os.path.basename(src.replace("\\", "/")))


# ---------------------------------------------------------------- root
mv("mtg-models-data.iml", f"{HOLD}/mtg-models-data.iml")
mv(".claude", f"{HOLD}/.claude")

# ---------------------------------------------------------------- price-predictor
mv("price-predictor/sklearn/20260317-141905.joblib",
   "models/price-predictor/gen1/sklearn/2026-03-17-baseline/20260317-141905.joblib")
mv("price-predictor/transformer/latest.pt",
   "models/price-predictor/gen1/transformer/2026-04-05-production/latest.pt")
mv("price-predictor/transformer/vocab.txt",
   "output/price-predictor/gen1/inputs/2026-03-26-vocabulary/vocab.txt")


# ---------------------------------------------------------------- generic prefix splitter
def split_prefix_files(src_dir, prefixes, route):
    """For each file in src_dir, find the longest matching prefix and call route(prefix, run, file)."""
    for f in ls(src_dir):
        if isdir(f"{src_dir}/{f}"):
            continue
        best = None
        for pre in prefixes:
            if f.startswith(pre) and (best is None or len(pre) > len(best)):
                best = pre
        if best is None:
            continue
        dst = route(best, prefixes[best], f[len(best):])
        if dst:
            mv(f"{src_dir}/{f}", dst)


# ---------------------------------------------------------------- sealed models
S = "sealed/trained-models"
gen1_runs = {f"{n}layers_": f"2026-04-{23 if n <= 4 else 24}-{n}-layers" for n in range(2, 7)}


def r_gen1(pre, run, rest):
    f = pre + rest
    if rest == "training.txt":
        return f"models/sealed/gen1/scorer/{run}/console.log"
    if rest.endswith(".pt.gz"):
        return f"models/sealed/gen1/scorer/{run}/{f}"
    return f"output/sealed/gen1/reports/{run}/{f}"


split_prefix_files(f"{S}/gen1/scorer", gen1_runs, r_gen1)

gen2_runs = {
    "best_l6_full_training": "2026-04-29-l6-full",
    "best_phaseB_full_training": "2026-04-29-phase-b-full",
    "best_l6_1epoch": "2026-04-29-l6-one-epoch",
    "best_phaseB_1epoch_emblr1e-07": "2026-04-29-phase-b-emb-lr1e-7",
    "best_phaseB_1epoch_emblr1e-06": "2026-04-30-phase-b-emb-lr1e-6",
}


def r_gen2(pre, run, rest):
    f = pre + rest
    if rest == "_training.txt":
        return f"models/sealed/gen2/scorer/{run}/console.log"
    if rest == ".pt.gz":
        return f"models/sealed/gen2/scorer/{run}/{f}"
    if rest == "_decks.txt":
        return f"output/sealed/gen2/reports/{run}/{f}"
    problems.append(f"gen2 scorer: unexpected {f}")


split_prefix_files(f"{S}/gen2/scorer", gen2_runs, r_gen2)
for f in ("forge_decks.txt", "gen1_decks.txt", "gen1_pools.txt"):
    mv(f"{S}/gen2/scorer/{f}", f"{HOLD}/sealed-gen2-scorer/{f}")


def r_model(module, gen, model):
    def route(pre, run, rest):
        f = pre + rest
        if rest == "-training.log":
            return f"models/{module}/{gen}/{model}/{run}/console.log"
        return f"models/{module}/{gen}/{model}/{run}/{f}"
    return route


mv(f"{S}/gen3/encoder/vocab.txt", "output/sealed/gen3/inputs/2026-05-11-vocabulary/vocab.txt")
split_prefix_files(f"{S}/gen3/encoder", {
    "full-20260511-223352-attn-6l-2h-2q-0.1mlm-128d": "2026-05-11-full-attn-6l2h2q-128d",
    "full-20260511-232345-attn-6l-2h-2q-0.1mlm-256d": "2026-05-12-full-attn-6l2h2q-256d",
}, r_model("sealed", "gen3", "encoder"))
split_prefix_files(f"{S}/gen3/encoder/experiments", {
    "20260511-120559-6l-4h-4q-0.1mlm": "2026-05-11-dual-6l4h4q",
    "20260511-130933-4l-4h-4q-0.1mlm": "2026-05-11-dual-4l4h4q",
    "20260511-134314-2l-4h-4q-0.1mlm": "2026-05-11-dual-2l4h4q",
    "20260511-140030-1l-4h-4q-0.1mlm": "2026-05-11-dual-1l4h4q",
    "20260511-152849-6l-8h-8q-0.1mlm": "2026-05-11-dual-6l8h8q",
    "20260511-163416-6l-16h-16q-0.1mlm": "2026-05-11-dual-6l16h16q",
    "20260511-165924-6l-4h-4q-0.02mlm": "2026-05-11-dual-6l4h4q-mlm0.02",
    "20260511-174344-attn-6l-4h-4q-0.1mlm": "2026-05-11-attn-6l4h4q",
    "20260511-180739-attn-4l-4h-4q-0.1mlm": "2026-05-11-attn-4l4h4q",
    "20260511-184822-attn-2l-4h-4q-0.1mlm": "2026-05-11-attn-2l4h4q",
    "20260511-193938-attn-1l-4h-4q-0.1mlm": "2026-05-11-attn-1l4h4q",
    "20260511-200453-attn-6l-4h-4q-0.1mlm-128d": "2026-05-11-attn-6l4h4q-128d",
    "20260511-204124-attn-6l-4h-4q-0.1mlm-64d": "2026-05-11-attn-6l4h4q-64d",
    "20260511-210357-attn-6l-2h-2q-0.1mlm-64d": "2026-05-11-attn-6l2h2q-64d",
    "20260511-212205-attn-6l-2h-2q-0.1mlm-128d": "2026-05-11-attn-6l2h2q-128d",
    "20260511-214521-attn-6l-2h-2q-0.1mlm": "2026-05-11-attn-6l2h2q",
}, r_model("sealed", "gen3", "encoder"))

gen3_scorer = {"256-best_l6_h4_s4_ff1088_mlp256_lr1e-05": "2026-05-13-enc256",
               "128-best_l6_h4_s4_ff1088_mlp256_lr1e-05": "2026-05-13-enc128"}


def r_gen3_scorer(pre, run, rest):
    f = pre + rest
    if rest == "-training.log":
        return f"models/sealed/gen3/scorer/{run}/console.log"
    if rest == ".pt":
        return f"models/sealed/gen3/scorer/{run}/{f}"
    if rest.endswith("2.txt"):
        return f"output/sealed/gen3/reports/{run}/eval-2/{f}"
    return f"output/sealed/gen3/reports/{run}/{f}"


split_prefix_files(f"{S}/gen3/scorer", gen3_scorer, r_gen3_scorer)

mv(f"{S}/gen4/encoder/vocab.txt", "output/sealed/gen4/inputs/2026-05-11-vocabulary/vocab.txt")
split_prefix_files(f"{S}/gen4/encoder", {
    "full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d": "2026-05-17-attn-6l8h8q-512d",
}, r_model("sealed", "gen4", "encoder"))
split_prefix_files(f"{S}/gen4/encoder/experiments", {
    "full-20260515-210145-attn-6l-4h-4q-0.1mlm-512d": "2026-05-15-attn-6l4h4q-512d",
    "full-20260515-223639-attn-6l-2h-2q-0.1mlm-512d": "2026-05-15-attn-6l2h2q-512d",
    "full-20260516-091957-attn-6l-8h-2q-0.1mlm-512d": "2026-05-16-attn-6l8h2q-512d",
    "full-20260516-200340-attn-6l-2h-8q-0.1mlm-512d": "2026-05-16-attn-6l2h8q-512d",
    "full-20260516-212351-attn-6l-4h-8q-0.1mlm-512d": "2026-05-16-attn-6l4h8q-512d",
    "full-20260516-232534-attn-6l-8h-4q-0.1mlm-512d": "2026-05-16-attn-6l8h4q-512d",
    "full-20260517-105613-attn-6l-16h-16q-0.1mlm-512d": "2026-05-17-attn-6l16h16q-512d",
    "full-20260517-155247-attn-8l-8h-8q-0.1mlm-512d": "2026-05-17-attn-8l8h8q-512d",
}, r_model("sealed", "gen4", "encoder"))
split_prefix_files(f"{S}/gen4/scorer", {
    "512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog": "2026-05-18-enc512-ff2176-mwlog",
    "256-best_l6_h4_s4_ff1088_mlp256_lr1e-05_mwlin": "2026-05-18-enc256-mwlin",
}, r_model("sealed", "gen4", "scorer"))
split_prefix_files(f"{S}/gen4/scorer/experiments", {
    "256-best_l6_h4_s4_ff1088_mlp256_lr1e-05": "2026-05-17-enc256",
    "512-best_l6_h4_s4_ff1088_mlp256_lr1e-05": "2026-05-17-enc512-ff1088",
    "512-best_l6_h4_s4_ff2176_mlp512_lr1e-05": "2026-05-17-enc512-ff2176",
    "512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlin": "2026-05-18-enc512-ff2176-mwlin",
    "256-best_l6_h4_s4_ff1088_mlp256_lr1e-05_mwlog": "2026-05-18-enc256-mwlog",
}, r_model("sealed", "gen4", "scorer"))
split_prefix_files(f"{S}/gen5/picker", {
    "best_20260524_203230-4l-8h-4top256-1e-5lr": "2026-05-24-4l8h-4top256-lr1e-5",
}, r_model("sealed", "gen5", "picker"))
split_prefix_files(f"{S}/gen5/picker/experiments", {
    "best_20260521_071908-4l-8h-no_normalization": "2026-05-21-4l8h-no-normalization",
    "best_20260521_212220-4l-8h-normalization": "2026-05-21-4l8h-normalization",
    "best_20260521_225348-6l-4h-0.1dropout": "2026-05-21-6l4h-dropout0.1",
    "best_20260522_041629-6l-4h": "2026-05-22-6l4h",
    "best_20260522_171840-6l-8h": "2026-05-22-6l8h",
    "best_20260523_101152-4l-8h-16top64s": "2026-05-23-4l8h-16top64s",
    "best_20260523_132702-4l-8h-16top64s-1e-4lr": "2026-05-23-4l8h-16top64s-lr1e-4",
    "best_20260523_232104-4l-8h-16top128s-1e-4lr": "2026-05-23-4l8h-16top128s-lr1e-4",
}, r_model("sealed", "gen5", "picker"))

# ---------------------------------------------------------------- sealed data
T = "sealed/training-data"
b07 = f"{T}/matches-b07"
for f, dst in {
    "match-outcomes-gen0.txt.gz": "gen1/records/2026-04-23-forge-decks",
    "match-outcomes-gen1-vs-0.txt.gz": "gen1/records/2026-04-26-vs-gen0",
    "match-outcomes-gen2-vs-0-1.txt.gz": "gen2/records/2026-05-04-vs-gen0-gen1",
    "match-outcomes-gen3-vs-0-2.txt": "gen3/records/2026-05-14-vs-gen0-gen2",
    "cards-played-gen3-vs-0-2.txt": "gen3/records/2026-05-14-vs-gen0-gen2",
    "match-outcomes-gen3-vs-forge-best.txt": "gen3/records/2026-05-15-vs-forge-best",
    "cards-played-gen3-vs-forge-best.txt": "gen3/records/2026-05-15-vs-forge-best",
    "match-outcomes-gen4-vs-forge-best-gen3.txt": "gen4/records/2026-05-21-vs-forge-best-gen3",
    "cards-played-gen4-vs-forge-best-gen3.txt": "gen4/records/2026-05-21-vs-forge-best-gen3",
    "match-outcomes-gen5-vs-gen4-forge.txt": "gen5/records/2026-05-26-vs-gen4-forge",
    "cards-played-gen5-vs-gen4-forge.txt": "gen5/records/2026-05-26-vs-gen4-forge",
}.items():
    mv_into(f"{b07}/{f}", f"output/sealed/{dst}")
bo1 = f"{T}/matches-bo1"
for f, dst in {
    "match-outcomes-bo1-embedding.txt": "gen3/records/2026-05-10-bo1-embedding",
    "match-outcomes-bo1-embedding.7z": "gen3/records/2026-05-10-bo1-embedding",
    "cards-played-bo1-embedding.txt": "gen3/records/2026-05-10-bo1-embedding",
    "cards-played-bo1-embedding.cleaned.txt": "gen3/corpus/2026-05-11-bo1-cleaned",
    "cards-played-bo1-embedding.cleaned.7z": "gen3/corpus/2026-05-11-bo1-cleaned",
    "cards-win-rates.txt": "gen3/reports/2026-05-17-card-win-rates",
    "cards-win-rates.7z": "gen3/reports/2026-05-17-card-win-rates",
}.items():
    mv_into(f"{bo1}/{f}", f"output/sealed/{dst}")
for f, dst in {
    "decks/generated-decks-gen0.txt.gz": "gen1/corpus/2026-05-02-decks-forge",
    "decks/generated-decks-gen1.txt.gz": "gen1/corpus/2026-04-30-decks",
    "decks/generated-decks-gen2.txt.gz": "gen2/corpus/2026-05-02-decks",
    "pools/pools-gen2a.txt.gz": "gen2/corpus/2026-05-01-pools-2a",
    "pools/pools-gen2b1.txt.gz": "gen2/corpus/2026-05-01-pools-2b1",
    "pools/pools-gen2ba.txt.gz": "gen2/corpus/2026-05-01-pools-2ba",
    "decks/generated-decks-gen2a.txt.gz": "gen2/corpus/2026-05-01-decks-2a",
    "decks/generated-decks-gen2ba.txt.gz": "gen2/corpus/2026-05-01-decks-2ba",
    "decks/generated-decks-gen2b1.txt.gz": "gen2/corpus/2026-05-02-decks-2b1",
    "pools/pools-gen3-128.txt": "gen3/corpus/2026-05-13-pools-enc128",
    "pools/pools-gen3-256.txt": "gen3/corpus/2026-05-13-pools-enc256",
    "decks/generated-decks-gen3-128.txt": "gen3/corpus/2026-05-14-decks-enc128",
    "decks/generated-decks-gen3-256.txt": "gen3/corpus/2026-05-14-decks-enc256",
    "pools/pools-gen4-256.txt": "gen4/corpus/2026-05-18-pools-enc256",
    "pools/pools-gen4-512.txt": "gen4/corpus/2026-05-18-pools-enc512",
    "decks/generated-decks-gen4-256.txt": "gen4/corpus/2026-05-19-decks-enc256",
    "decks/generated-decks-gen4-512.txt": "gen4/corpus/2026-05-19-decks-enc512",
    "pools/pools-gen5.txt": "gen5/corpus/2026-05-25-pools",
    "decks/generated-decks-gen5.txt": "gen5/corpus/2026-05-26-decks",
}.items():
    mv_into(f"{T}/{f}", f"output/sealed/{dst}")

# ---------------------------------------------------------------- draft
D = "draft/trained-models"
split_prefix_files(f"{D}/gen1", {"l6_lr3e-4_decay_20260604_080249": "2026-06-04-l6-lr-decay"},
                   r_model("draft", "gen1", "agent"))
split_prefix_files(f"{D}/gen1/experiments", {
    "l4_lr3e-4_20260603_121316": "2026-06-03-l4-lr3e-4",
    "l4_lr3e-5_20260603_211958": "2026-06-03-l4-lr3e-5",
    "l4_lr3e-6_20260603_234318": "2026-06-03-l4-lr3e-6",
    "l8_lr3e-4_decay_20260604_232956": "2026-06-04-l8-lr-decay",
}, r_model("draft", "gen1", "agent"))
mv_into(f"{D}/gen1/drafts-gen1-argmax.jsonl", "output/draft/gen1/records/2026-06-10-argmax-live-play")
mv(f"{D}/gen1/drafts-gen1-argmax-deck-analysis.log",
   "output/draft/gen1/reports/2026-06-10-argmax-deck-analysis/console.log")
mv_into(f"{D}/gen1/drafts-gen1-sample-t1.0.jsonl", "output/draft/gen2/records/2026-06-14-gen1-sample-t1")
mv(f"{D}/gen1/drafts-gen1-sample-t1.0-deck-analysis.log",
   "output/draft/gen2/reports/2026-06-14-gen1-sample-t1-analysis/console.log")


def r_draft(gen):
    def route(pre, run, rest):
        f = pre + rest
        if rest == "-training.log":
            return f"models/draft/{gen}/agent/{run}/console.log"
        if rest == ".pt":
            return f"models/draft/{gen}/agent/{run}/{f}"
        if rest == "-drafts.jsonl":
            return f"output/draft/{gen}/records/{run}/{f}"
        if rest.endswith("-greedy-prefix-deck-scores.jsonl"):
            return f"output/draft/{gen}/reports/2026-08-25-greedy-prefix-scores/{f}"
        if rest == "-prefix-deck-scores.jsonl":
            return f"{HOLD}/draft-gen4/{f}"
        return f"output/draft/{gen}/reports/{run}/{f}"
    return route


split_prefix_files(f"{D}/gen2/experiments", {
    "lr3e-5_kl0.1_ent0.1_val1.0_gae0.95_20260614_234040": "2026-06-14-entropy0.1",
    "20260615_074224": "2026-06-15-entropy0.01-value0.1",
    "20260615_190126": "2026-06-15-entropy0.01-value0.1-decay",
}, r_draft("gen2"))
split_prefix_files(f"{D}/gen3/temperature-on-all-agents", {
    "lr1e-4_t2_20260805_183818": "2026-08-05-t2-all-lr1e-4",
    "lr1e-6_t2_20260805_201705": "2026-08-05-t2-all-lr1e-6",
    "lr1e-5_t2_20260805_221050": "2026-08-05-t2-all-lr1e-5",
    "lr1e-5_t3_20260805_230506": "2026-08-05-t3-all-lr1e-5",
}, r_draft("gen3"))
split_prefix_files(f"{D}/gen3", {
    "lr1e-5_t2_20260806_112400": "2026-08-06-t2-learner-lr1e-5",
    "lr1e-5_t3_20260806_144756": "2026-08-06-t3-learner-lr1e-5",
}, r_draft("gen3"))
split_prefix_files(f"{D}/gen4", {
    "lr1e-5_t2all_nodecay": "2026-08-07-t2-all-no-decay",
    "lr1e-5_t2all_decay0.3": "2026-08-08-t2-all-decay0.3",
    "lr1e-5_t3all_decay0.3": "2026-08-08-t3-all-decay0.3",
    "lr1e-5_t3learner_t2field_decay0.3": "2026-08-09-t3-learner-t2-field",
}, r_draft("gen4"))
mv(f"{D}/gen4/rebuild-greedy.log", "output/draft/gen4/reports/2026-08-25-greedy-prefix-scores/console.log")
mv(f"{D}/gen4/picker-built", "output/draft/gen4/reports/2026-08-25-picker-prefix-scores")


# ---------------------------------------------------------------- effects records (dir move + logs/)
def records_run(src, dst):
    mv(src, dst)
    for f in ls(src):
        if f.endswith(".log") or f.endswith(".progress.txt"):
            # moved after the run dir itself: express it against the destination
            moves.append((dst.replace("/", "\\") + "\\" + f, dst.replace("/", "\\") + "\\logs\\" + f))


E = "effects"
records_run(f"{E}/training-data/gen1/records-gen1/full-strength", "output/effects/gen1/records/2026-09-11-full-strength")
records_run(f"{E}/training-data/gen1/records-gen1/depleted", "output/effects/gen1/records/2026-09-13-depleted")
records_run(f"{E}/training-data/gen1/records-gen1/variants", "output/effects/gen1/records/2026-09-15-variants")
records_run(f"{E}/training-data/gen2/records-gen2-tiers123/depleted", "output/effects/gen2/records/2026-10-09-depleted-tiers123")
records_run(f"{E}/training-data/gen2/records-gen2-tiers123/full-strength", "output/effects/gen2/records/2026-10-09-full-strength-tiers123")
records_run(f"{E}/training-data/gen2/records-gen2/depleted", "output/effects/gen2/records/2026-10-11-depleted")
records_run(f"{E}/training-data/gen2/records-gen2/full-strength", "output/effects/gen2/records/2026-10-12-full-strength")
records_run(f"{E}/training-data/gen2/records-gen2/held-out-coverage", "output/effects/gen2/records/2026-10-12-held-out-coverage")
records_run(f"{E}/training-data/gen2/records-gen2/variants", "output/effects/gen2/records/2026-10-12-variants")

# ---------------------------------------------------------------- effects gen1 inputs, corpora, models
mv(f"{E}/training-data/gen1/variant-scripts-gen1", "output/effects/gen1/inputs/2026-09-15-variant-scripts")
mv(f"{E}/trained-models/gen2/vocab.txt", "output/effects/gen1/inputs/2026-09-09-vocabulary/vocab.txt")
mv_into(f"{E}/training-data/gen1/gen1-keyword-definitions.json", "output/effects/gen1/inputs/2026-10-09-keyword-definitions")
mv_into(f"{E}/trained-models/gen1/gen1-vocab-script.txt", "output/effects/gen1/inputs/2026-10-09-vocab-script")
mv_into(f"{E}/trained-models/gen1/vocab-script-gen1-sidecars.txt", "output/effects/gen1/inputs/2026-10-08-vocab-script-sidecars")
mv(f"{E}/training-data/gen1/corpus-20260917-pre-textless", "output/effects/gen1/corpus/2026-09-17-pre-textless")
mv(f"{E}/training-data/gen1/corpus", "output/effects/gen1/corpus/2026-09-17-textless")
EM = f"{E}/trained-models/gen1/effect-model"
mv(f"{EM}/09.17c-build-corpus.log", "output/effects/gen1/corpus/2026-09-17-textless/console.log")
mv(f"{EM}/09.17c-build-corpus.err.log", "output/effects/gen1/corpus/2026-09-17-textless/logs/09.17c-build-corpus.err.log")


def model_run(src, dst):
    mv(src, dst)
    if os.path.exists(P(src) + "\\train.log"):
        moves.append((dst.replace("/", "\\") + "\\train.log", dst.replace("/", "\\") + "\\console.log"))


model_run(f"{E}/trained-models/gen1/2026-09-17-full-textless-corpus", "models/effects/gen1/effect-model/2026-09-17-full-textless-corpus")
for f in ("09.17c.log", "09.17d.log", "09.17e.log"):
    mv(f"{EM}/{f}", f"models/effects/gen1/effect-model/2026-09-17-full-textless-corpus/logs/{f}")
for f in ls(EM):
    if f not in ("09.17c-build-corpus.log", "09.17c-build-corpus.err.log", "09.17c.log", "09.17d.log", "09.17e.log"):
        mv(f"{EM}/{f}", f"{HOLD}/effects-gen1-effect-model/{f}")
model_run(f"{E}/trained-models/gen1/2026-09-17-identity", "models/effects/gen1/effect-model-identity/2026-09-18-baseline")
model_run(f"{E}/trained-models/gen1/2026-09-17-taxonomy", "models/effects/gen1/effect-model-taxonomy/2026-09-18-baseline")
model_run(f"{E}/trained-models/gen1/2026-09-17-state-only", "models/effects/gen1/effect-model-state-only/2026-09-19-baseline")
model_run(f"{E}/trained-models/gen1/2026-09-17-no-state", "models/effects/gen1/effect-model-no-state/2026-09-19-baseline")

# ---------------------------------------------------------------- effects gen2 inputs, corpus
G2D = f"{E}/training-data/gen2"
mv_into(f"{G2D}/keyword-definitions.json", "output/effects/gen2/inputs/2026-10-09-keyword-definitions")
mv_into(f"{G2D}/holdout-cards.txt", "output/effects/gen2/inputs/2026-10-09-holdout-cards")
mv_into(f"{E}/trained-models/gen2/vocab-script.txt", "output/effects/gen2/inputs/2026-10-09-vocab-script")
mv(f"{G2D}/variant-scripts", "output/effects/gen2/inputs/2026-10-12-variant-scripts")
mv(f"{G2D}/corpus-gen2", "output/effects/gen2/corpus/2026-10-13-stage-4")
mv(f"{G2D}/build-corpus-gen2.log", "output/effects/gen2/corpus/2026-10-13-stage-4/console.log")

# ---------------------------------------------------------------- effects gen2 model runs
G2 = f"{E}/trained-models/gen2"
TRAIN = {
    "2026-10-gen2-e64-earlier-inputs": "2026-10-13-e64-earlier-inputs",
    "2026-10-gen2-e64": "2026-10-14-e64",
    "2026-10-gen2-e128": "2026-10-15-e128",
    "2026-10-gen2-e64-lethality": "2026-10-16-e64-lethality",
    "2026-10-gen2-e128-lethality": "2026-10-16-e128-lethality",
    "2026-10-gen2-e64-lethality-ext20": "2026-10-17-e64-lethality-continued",
    "2026-10-gen2-e64-lethality-lr0.1": "2026-10-17-e64-lethality-tenth-lr",
    "2026-10-gen2-e128-lethality-ext20": "2026-10-18-e128-lethality-continued",
    "2026-10-gen2-e128-lethality-lr0.1": "2026-10-19-e128-lethality-tenth-lr",
    "2026-10-gen2-e256-lethality": "2026-10-19-e256-lethality",
    "2026-10-gen2-e256-lethality-ext20": "2026-10-20-e256-lethality-continued",
    "2026-10-gen2-e256-lethality-lr0.1": "2026-10-20-e256-lethality-tenth-lr",
}
PROBE = {  # probe-only dir -> (training run new name, epoch dir)
    "2026-10-gen2-e64-lethality-ext20-last": ("2026-10-17-e64-lethality-continued", "last-epoch"),
    "2026-10-gen2-e64-lethality-lr0.1-last": ("2026-10-17-e64-lethality-tenth-lr", "last-epoch"),
    "2026-10-gen2-e128-lethality-ext20-last": ("2026-10-18-e128-lethality-continued", "last-epoch"),
    "2026-10-gen2-e128-lethality-lr0.1-last": ("2026-10-19-e128-lethality-tenth-lr", "last-epoch"),
    "2026-10-gen2-e256-lethality-ext20-last": ("2026-10-20-e256-lethality-continued", "last-epoch"),
    "2026-10-gen2-e256-lethality-lr0.1-last": ("2026-10-20-e256-lethality-tenth-lr", "last-epoch"),
    "2026-10-gen2-e256-lethality-lr0.1-best": ("2026-10-20-e256-lethality-tenth-lr", "epoch-55"),
}
RULE_CHECKS = ("combat-grid-steps", "combat-grid", "evasion-grid", "colour-pips", "target-restrictions",
               "zone-effects", "ladder-check-amount-alone", "ladder-check", "grid-probe", "grid-training")
RUN_WIDE = ("boundary-by-epoch", "grid-best-epoch", "grid-last-epoch")  # span epochs: reports/<run>/ top level

MODEL_FILES = re.compile(r"^(\d{8}-\d{6}\.pt|latest\.pt|resume\.pt|resume-best\.pt|train\.log|arm\.log|started-from\.txt)$")


def split_gen2_dir(old, run, epoch, probe_only):
    src = f"{G2}/{old}"
    model = f"models/effects/gen2/effect-model/{run}"
    rep = f"output/effects/gen2/reports/{run}/{epoch}"
    cache = f"output/effects/gen2/cache/{run}/{epoch}"
    for f in ls(src):
        s = f"{src}/{f}"
        if f == "abilities":
            problems.append(f"abilities still present: {s}")
        elif f == "epochs":
            mv(s, f"{model}/epochs")
        elif probe_only and f == "latest.pt":
            mv(s, f"{HOLD}/effects-gen2-probe-latest/{old}-latest.pt")
        elif probe_only and f in ("arm.log", "source.txt"):
            mv(s, f"{rep}/logs/{f}")
        elif not probe_only and MODEL_FILES.match(f):
            mv(s, f"{model}/{f}")
        elif f == "encode.log":
            mv(s, f"{cache}/console.log")
        elif f == "evaluate.log":
            mv(s, f"{rep}/evaluate/console.log")
        elif f.startswith("probes-") and f.endswith(".log"):
            mv(s, f"{rep}/embedding-probes/logs/{f}")
        elif f == "knowledge-probes.log":
            mv(s, f"{rep}/knowledge-probes/console.log")
        elif f == "knowledge-probes-board.log":
            mv(s, f"{rep}/knowledge-probes-board/console.log")
        elif f == "reports":
            for r in ls(s):
                rs = f"{s}/{r}"
                if r.startswith("embedding-probes-"):
                    for x in ls(rs):
                        if x == "cache":
                            mv(f"{rs}/cache", f"{cache}/embedding-probes")
                        else:
                            mv(f"{rs}/{x}", f"{rep}/embedding-probes/{x}")
                elif r.startswith("knowledge-probes-"):
                    for x in ls(rs):
                        mv(f"{rs}/{x}", f"{rep}/knowledge-probes/{x}")
                elif r == "compare-vs-earlier-inputs.md":
                    mv(rs, f"output/effects/gen2/reports/{run}/compare-vs-earlier-inputs/{r}")
                elif r == "board-run":
                    for k in ls(rs):
                        for x in ls(f"{rs}/{k}"):
                            mv(f"{rs}/{k}/{x}", f"{rep}/knowledge-probes-board/{x}")
                else:
                    problems.append(f"unknown report dir {rs}")
        else:
            stem = f.rsplit(".", 1)[0]
            if stem in RULE_CHECKS:
                mv(s, f"{rep}/{stem}/{f}")
            elif stem in RUN_WIDE:
                # in a probe-only dir the file is about that one epoch; in a training dir it spans epochs
                base = rep if probe_only else f"output/effects/gen2/reports/{run}"
                mv(s, f"{base}/{stem}/{f}")
            else:
                problems.append(f"unclassified file {s}")


for old, run in TRAIN.items():
    split_gen2_dir(old, run, "kept-epoch", probe_only=False)
for old, (run, epoch) in PROBE.items():
    split_gen2_dir(old, run, epoch, probe_only=True)

CONSOLE = {  # root console prefix -> destination dir
    "e64-retrain": "models/effects/gen2/effect-model/2026-10-14-e64",
    "e128": "models/effects/gen2/effect-model/2026-10-15-e128",
    "e64-lethality": "models/effects/gen2/effect-model/2026-10-16-e64-lethality",
    "e128-lethality": "models/effects/gen2/effect-model/2026-10-16-e128-lethality",
    "e64-lethality-ext20": "models/effects/gen2/effect-model/2026-10-17-e64-lethality-continued",
    "e64-lethality-lr0.1": "models/effects/gen2/effect-model/2026-10-17-e64-lethality-tenth-lr",
    "e128-lethality-ext20": "models/effects/gen2/effect-model/2026-10-18-e128-lethality-continued",
    "e128-lethality-lr0.1": "models/effects/gen2/effect-model/2026-10-19-e128-lethality-tenth-lr",
    "e256-lethality": "models/effects/gen2/effect-model/2026-10-19-e256-lethality",
    "e256-lethality-ext20": "models/effects/gen2/effect-model/2026-10-20-e256-lethality-continued",
    "e256-lethality-lr0.1": "models/effects/gen2/effect-model/2026-10-20-e256-lethality-tenth-lr",
    "e64-probes": "output/effects/gen2/reports/2026-10-14-e64/kept-epoch/logs",
    "e64-lethality-ext20-last": "output/effects/gen2/reports/2026-10-17-e64-lethality-continued/last-epoch",
    "e64-lethality-lr0.1-last": "output/effects/gen2/reports/2026-10-17-e64-lethality-tenth-lr/last-epoch",
    "e128-lethality-ext20-last": "output/effects/gen2/reports/2026-10-18-e128-lethality-continued/last-epoch",
    "e128-lethality-lr0.1-last": "output/effects/gen2/reports/2026-10-19-e128-lethality-tenth-lr/last-epoch",
    "2026-10-gen2-e256-lethality-ext20-last": "output/effects/gen2/reports/2026-10-20-e256-lethality-continued/last-epoch",
    "2026-10-gen2-e256-lethality-lr0.1-last": "output/effects/gen2/reports/2026-10-20-e256-lethality-tenth-lr/last-epoch",
    "b256-best": "output/effects/gen2/reports/2026-10-20-e256-lethality-tenth-lr/epoch-55",
}
for f in ls(G2):
    if isdir(f"{G2}/{f}"):
        continue
    m = re.match(r"^(.*)-console\.(out|err)$", f)
    if m and m.group(1) in CONSOLE:
        d = CONSOLE[m.group(1)]
        if m.group(2) == "out":
            mv(f"{G2}/{f}", f"{d}/console.log")
        else:
            mv(f"{G2}/{f}", f"{d}/logs/console.err")
    elif f == "b256-best-probes.log":
        mv(f"{G2}/{f}", "output/effects/gen2/reports/2026-10-20-e256-lethality-tenth-lr/epoch-55/logs/b256-best-probes.log")
    elif f == "e256-chain.log":
        mv(f"{G2}/{f}", "output/effects/gen2/reports/2026-10-20-e256-lethality-tenth-lr/logs/e256-chain.log")


# ---------------------------------------------------------------- validation
def under_covered(rel):
    parts = rel.split("\\")
    return any("\\".join(parts[:i]) in covered for i in range(1, len(parts) + 1))


OLD_ROOTS = ["draft", "effects", "price-predictor", "sealed", ".claude", "mtg-models-data.iml"]
leftover = []
for r in OLD_ROOTS:
    p = P(r)
    if os.path.isfile(p):
        if not under_covered(r):
            leftover.append(r)
        continue
    for dirpath, dirs, files in os.walk(p):
        rel = os.path.relpath(dirpath, ROOT)
        if under_covered(rel):
            dirs[:] = []
            continue
        for f in files:
            fr = rel + "\\" + f
            if not under_covered(fr):
                leftover.append(fr)

dsts = [d for _, d in moves]
dup = sorted({d for d in dsts if dsts.count(d) > 1})
existing = [d for d in dsts if os.path.exists(P(d))]
# a later move may take its source from an earlier move's destination (records logs/, train.log)
def under_earlier_dst(s, upto):
    earlier = {d for _, d in moves[:upto]}
    parts = s.split("\\")
    return any("\\".join(parts[:i]) in earlier for i in range(1, len(parts) + 1))


first_order_missing = [s for i, (s, _) in enumerate(moves)
                       if not os.path.exists(P(s)) and not under_earlier_dst(s, i)]
bad_roots = [d for d in dsts if not (d.startswith("output\\") or d.startswith("models\\") or d.startswith(HOLD + "\\"))]

print(f"moves: {len(moves)}")
print(f"  to output/: {sum(d.startswith('output') for d in dsts)}  to models/: {sum(d.startswith('models') for d in dsts)}  to {HOLD}/: {sum(d.startswith(HOLD) for d in dsts)}")
for name, lst in [("problems", problems), ("leftover (unmapped) files", leftover), ("duplicate destinations", dup),
                  ("destinations that already exist", existing), ("missing sources", first_order_missing),
                  ("destinations outside output/models/hold", bad_roots)]:
    print(f"{name}: {len(lst)}")
    for x in lst[:30]:
        print("   ", x)

with open(PLAN_TSV, "w", encoding="utf-8") as fh:
    for s, d in moves:
        fh.write(f"{s}\t{d}\n")
print("plan written:", PLAN_TSV)

ok = not (problems or leftover or dup or existing or first_order_missing or bad_roots)
if sys.argv[1:] == ["execute"]:
    if not ok:
        print("REFUSING to execute: validation failed")
        sys.exit(1)
    done = 0
    with open(JOURNAL, "a", encoding="utf-8") as j:
        for s, d in moves:
            os.makedirs(os.path.dirname(P(d)), exist_ok=True)
            os.rename(P(s), P(d))  # server-side rename; raises instead of copying
            j.write(f"{s}\t{d}\n"); j.flush()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(moves)} moved", flush=True)
    print(f"moved {done}/{len(moves)}", flush=True)
    # remove now-empty old directories, deepest first (rmdir fails on non-empty: safe)
    removed = 0
    for r in ["draft", "effects", "price-predictor", "sealed"]:
        for dirpath, dirs, files in os.walk(P(r), topdown=False):
            try:
                os.rmdir(dirpath); removed += 1
            except OSError:
                pass
    print(f"removed {removed} empty old directories", flush=True)
    rest = [os.path.relpath(os.path.join(dp, f), ROOT) for r in ["draft", "effects", "price-predictor", "sealed"]
            if os.path.exists(P(r)) for dp, _, fs in os.walk(P(r)) for f in fs]
    print(f"files left in old tree: {len(rest)}", rest[:20])
elif not ok:
    sys.exit(1)
