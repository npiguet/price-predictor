"""Comprehensive Rules catalogue: split the rules text, batch it for agents, validate sidecars,
report status.

Subcommands:
  split     parse rules/raw/MagicCompRules-*.txt into rules/rules.jsonl, rules/cr/<unit>.md
            and rules/cr/index.md
  packets   group units into agent work packets (rules/packets.json)
  validate  check every triage and scenario sidecar against the rules index and Forge's cardsfolder
  status    regenerate rules/STATUS.md from the sidecars
  gen3-fields
            write each heads line's gen-3 ids and learnable_after_gen3 and rules/GEN3-FIELDS.md,
            print the rules newly learnable against the list rules/gen3-fields.json recorded at
            the last run, and record the new list there; refuses a turn_action-only rule whose
            verdict is not `recorded`
  directed-rules --counts FILE [--probabilities-output FILE]
            write rules/directed-fork-rules.txt (the rules enabled for directed forks) and
            rules/coverage-wording-cards.txt (the cards of their scenarios) from gen-3's
            per-rule counts, and rules/directed-fork-counts.json (the part of the counts
            the list is decided from); with --probabilities-output, also each listed rule's
            own directed-fork probability as a JSON map, the file `sealed match-outcomes
            --directed-fork-probabilities` reads

A unit is a three-digit section (704), except sections 701 and 702, whose units are their
second-level rules (702.2, one keyword each), because those two sections hold a third of all rules.
"""

import argparse
from pathlib import Path

from _lib.paths import OUTPUT
from rules_catalogue_parts.directed_forks import cmd_directed_rules
from rules_catalogue_parts.gen3_fields import cmd_gen3_fields
from rules_catalogue_parts.locations import RULES
from rules_catalogue_parts.packets import cmd_packets
from rules_catalogue_parts.split import cmd_split
from rules_catalogue_parts.status import cmd_status
from rules_catalogue_parts.triage import cmd_premark
from rules_catalogue_parts.validation import cmd_validate

from effects.domain.variant_rounds import DEFAULT_STEER_FLOOR


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    p = sub.add_parser("packets")
    p.add_argument("--max-chars", type=int, default=16000)
    p = sub.add_parser("validate")
    p.add_argument("units", nargs="*")
    p.add_argument("--no-cards", action="store_true")
    p.add_argument("--warnings", action="store_true")
    p.add_argument("--heads", action="store_true", help="also check rules/heads/")
    sub.add_parser("status")
    sub.add_parser("premark")
    p = sub.add_parser("gen3-fields", help="write each heads line's gen-3 ids and "
                       "learnable_after_gen3, and print the rules newly learnable (FR-016)")
    p.add_argument("--rules-dir", type=Path, default=RULES, help=argparse.SUPPRESS)
    p = sub.add_parser("directed-rules", help="write the rules enabled for directed forks, "
                       "the cards of their scenarios and the counts they were decided from "
                       "(FR-014b, FR-038a)")
    p.add_argument("--counts", type=Path, required=True,
                   help="gen-3's per-rule counts, as `effects rule-counts` writes them")
    p.add_argument("--cards-folder", type=Path, default=OUTPUT / "cardsfolder",
                   help="the converted card tree whose names the wording list uses "
                   "(default: %(default)s)")
    p.add_argument("--probabilities-output", type=Path, default=None,
                   help="also write each listed rule's records per registered side and "
                   "--count-per-side here, the JSON file `sealed match-outcomes "
                   "--directed-fork-probabilities` computes each rule's probability from "
                   "(none by default: nothing is written)")
    p.add_argument("--count-per-side", type=int, default=DEFAULT_STEER_FLOOR,
                   help="records each side of a rule needs; its probability is 0 once both "
                   "have them (default: %(default)s)")
    p.add_argument("--rules-dir", type=Path, default=RULES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    {"split": cmd_split, "packets": cmd_packets, "validate": cmd_validate, "status": cmd_status,
     "premark": cmd_premark, "gen3-fields": cmd_gen3_fields,
     "directed-rules": cmd_directed_rules}[args.cmd](args)


if __name__ == "__main__":
    main()
