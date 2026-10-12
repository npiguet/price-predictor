"""Train and probe one effect-model sweep arm, keeping its whole record in its run directory.

Usage::

    python -u scripts/effect_sweep_arm.py --run-dir models/effects/runs/2026-10-gen2-<arm>/ \\
        --corpus output/effects/corpus-gen2/ -- <train-effect-model flags>

Everything after ``--`` goes to ``train-effect-model`` as given. The arm's run
directory then holds:

- ``latest.pt`` and the timestamped checkpoints;
- ``abilities/``, the ability cache this checkpoint encoded, so one arm's
  probes never read another arm's cache;
- ``reports/``, the embedding-probe and knowledge-probe report directories;
- one log per step (``arm.log``, ``train.log``, ``encode.log``,
  ``evaluate.log``, ``probes-<script>.log``, ``knowledge-probes.log``), each
  opening with the exact command line that produced it and the commit it ran
  at. A rerun appends to a step's log under its own command line.

The knowledge-probe set is frozen once per curated corpus and shared by every
arm, so it stays in ``output/effects/reports/``; the first arm on a corpus
freezes it.
"""

from __future__ import annotations

import argparse
import datetime
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_REPORTS = ROOT / "output" / "effects" / "reports"
EMBEDDING_PROBES = (
    "build_texts", "effect_profiles", "pca_directions", "lexical_probes",
    "linear_probes", "type_merging", "neighbours", "keyword_expansion",
)
STEPS = ("train", "encode", "evaluate", "embedding-probes", "knowledge-probes")


def git_state() -> str:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False,
        )
        return result.stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    return f"{git('rev-parse', 'HEAD')}{' (uncommitted changes)' if dirty else ''}"


def header(command: list[str]) -> str:
    started = datetime.datetime.now().isoformat(timespec="seconds")
    return f"$ {shlex.join(command)}\n# commit {git_state()}, started {started}\n"


def run_logged(command: list[str], log: Path) -> int:
    """Run ``command``, copying its merged output to the console and to ``log``."""
    with log.open("a", encoding="utf-8") as out:
        out.write(header(command))
        out.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            out.write(line)
            out.flush()
        code = process.wait()
        out.write(f"# exit {code}, ended {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    return code


def plan(args: argparse.Namespace, train_args: list[str]) -> list[tuple[str, list[str], str]]:
    """``(step, command, log name)`` for every command the arm runs, in order."""
    python = [sys.executable, "-u"]
    run_dir = args.run_dir
    checkpoint = str(run_dir / "latest.pt")
    abilities = str(run_dir / "abilities")
    reports = str(run_dir / "reports")
    probe = ["--checkpoint", checkpoint, "--abilities-root", abilities]
    # Without the variant tree a variant record resolves to no ability text in
    # training, encoding and evaluation alike, and nothing reports it.
    variants = ["--variant-scripts", str(args.variant_scripts)]
    commands = [
        ("train", python + ["-m", "effects", "train-effect-model",
                            "--corpus", str(args.corpus),
                            "--vocab-path", str(args.vocab_path), *variants,
                            "--model-output", str(run_dir), *train_args], "train.log"),
        ("encode", python + ["-m", "effects", "encode-abilities",
                             "--checkpoint", checkpoint, *variants,
                             "--output-root", abilities],
         "encode.log"),
        ("evaluate", python + ["-m", "effects", "evaluate-effect-model", *probe, *variants],
         "evaluate.log"),
    ]
    for script in EMBEDDING_PROBES:
        commands.append((
            "embedding-probes",
            python + [str(Path("scripts") / "effect_embedding_probes" / f"{script}.py"),
                      *probe, "--reports-dir", reports],
            f"probes-{script}.log",
        ))
    knowledge = python + [str(Path("scripts") / "effect_knowledge_probes" / "run.py"),
                          *probe, "--reports-dir", reports,
                          "--probe-set-dir", str(args.probe_set_dir)]
    if args.freeze_probe_set:
        commands.append(("knowledge-probes", knowledge + ["--freeze-probe-set"],
                         "knowledge-probes.log"))
    commands.append(("knowledge-probes", knowledge, "knowledge-probes.log"))
    return commands


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        split = argv.index("--")
        own, train_args = argv[:split], argv[split + 1:]
    else:
        own, train_args = argv, []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the arm's checkpoint directory, passed as --model-output")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="curated dataset every arm trains on")
    parser.add_argument("--vocab-path", type=Path,
                        default=Path("models/effects/vocab-script.txt"),
                        help="the vocabulary build-corpus was given")
    parser.add_argument("--variant-scripts", type=Path,
                        default=Path("output/effects/variant-scripts"),
                        help="the variant tree build-corpus was given")
    parser.add_argument("--from-step", choices=STEPS, default="train",
                        help="skip the steps before this one, e.g. after a crash")
    parser.add_argument("--probe-set-dir", type=Path, default=SHARED_REPORTS,
                        help="where the shared knowledge-probe set lives")
    parser.add_argument("--freeze-probe-set", action="store_true",
                        help="freeze the knowledge-probe set first (first arm on a corpus)")
    return parser.parse_args(own), train_args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args, train_args = parse_args(argv)
    run_dir: Path = args.run_dir
    if args.from_step == "train" and (run_dir / "latest.pt").exists():
        print(f"{run_dir} already holds a trained arm; pick another --run-dir, "
              "or resume with --from-step", file=sys.stderr)
        return 2
    if args.from_step == "train" and not train_args:
        print("pass the arm's train-effect-model flags after --", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_log = run_dir / "arm.log"
    with arm_log.open("a", encoding="utf-8") as out:
        out.write(header([sys.executable, "-u", *sys.argv]))
    first = STEPS.index(args.from_step)
    for step, command, log in plan(args, train_args):
        if STEPS.index(step) < first:
            continue
        with arm_log.open("a", encoding="utf-8") as out:
            out.write(f"{datetime.datetime.now():%H:%M:%S} {step}: {log}\n")
        code = run_logged(command, run_dir / log)
        if code != 0:
            with arm_log.open("a", encoding="utf-8") as out:
                out.write(f"{datetime.datetime.now():%H:%M:%S} {step} failed "
                          f"(exit {code}); resume with --from-step {step}\n")
            return code
    with arm_log.open("a", encoding="utf-8") as out:
        out.write(f"{datetime.datetime.now():%H:%M:%S} arm complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
