"""A loaded checkpoint is run on the input layout it trained on (FR-063c).

The layout changes what the model's input vectors mean and how wide they are,
so a checkpoint has to be fed the one it learned: gen-1, and gen-2 arms trained
before the layout was recorded, record none and read the earlier layout.
"""

from __future__ import annotations

from types import SimpleNamespace

from effects.domain.effect_head_input import (
    EARLIER_LAYOUT,
    INPUT_LAYOUT,
    THERMOMETER_STEPS,
    SlotKind,
)
from effects.infrastructure.model_runner import feature_widths, input_layout_of
from tests.unit.effects.domain.test_effect_head_input import _record


def test_a_checkpoint_recording_no_layout_reads_the_earlier_one():
    assert input_layout_of(SimpleNamespace(training_settings={})) is EARLIER_LAYOUT
    assert input_layout_of(SimpleNamespace(training_settings=None)) is EARLIER_LAYOUT


def test_a_checkpoint_recording_the_layout_reads_it():
    settings = {"input_layout": INPUT_LAYOUT}
    assert input_layout_of(SimpleNamespace(training_settings=settings)) == INPUT_LAYOUT


def test_the_widths_follow_the_layout():
    records = [_record()]
    earlier = feature_widths(records, EARLIER_LAYOUT)
    gen2 = feature_widths(records, INPUT_LAYOUT)
    assert gen2[SlotKind.GLOBAL] == earlier[SlotKind.GLOBAL]
    assert gen2[SlotKind.PLAYER] == earlier[SlotKind.PLAYER] + THERMOMETER_STEPS
    assert gen2[SlotKind.CARD] == earlier[SlotKind.CARD] + 3 * THERMOMETER_STEPS + 1
