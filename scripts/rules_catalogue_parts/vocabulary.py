"""The catalogue's closed vocabularies, spelt as the package spells them."""

from effects.domain import rule_catalogue

# The catalogue's vocabulary is the package's, so the corpus code that reads these
# sidecars and this validator spell every enum alike.
CATEGORIES = rule_catalogue.CATEGORIES
OBSERVABILITY = rule_catalogue.OBSERVABILITY
RECORD_KINDS = rule_catalogue.RECORD_KINDS
SCENARIO_CATEGORIES = rule_catalogue.SCENARIO_CATEGORIES
POLARITIES = rule_catalogue.POLARITIES
# Heads pass (rules/MODEL-FIELDS.md): what the model reads and is trained on per rule.
OUTPUT_STATUS = rule_catalogue.OUTPUT_STATUS
INPUT_STATUS = rule_catalogue.INPUT_STATUS
# Derived from the model's own layout rather than kept by hand.
HEAD_OUTPUTS = rule_catalogue.head_outputs()
HEAD_OUTPUT_PREFIXES = rule_catalogue.HEAD_OUTPUT_PREFIXES
OUTPUT_TAGS = rule_catalogue.OUTPUT_TAGS
INPUT_TAGS = rule_catalogue.INPUT_TAGS
FORK_OPERATIONS = rule_catalogue.FORK_OPERATIONS
SCRIPT_EDIT_OPERATIONS = frozenset(map(str, rule_catalogue.SCRIPT_EDIT_OPERATIONS))


DEPARTURE_STATUS = frozenset(rule_catalogue.DepartureStatus)
DEPARTURE_RESOLUTION = frozenset(rule_catalogue.DepartureResolution)
