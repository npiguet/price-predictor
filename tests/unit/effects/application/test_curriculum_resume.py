"""A run interrupted inside the curriculum's re-warm resumes to the run it
interrupted (finding 15.5, D10).

The re-warmed rate is a function of the step and of flags a resume compares,
and so is the averaged weights' start, so ``resume.pt`` stores nothing new
for them. Three fixture epochs of 6 steps, the switch at step 6 and a 10-step
re-warm: epoch 2 ends at step 12, inside the re-warm, where the average starts
(one epoch after the switch). Runs a real training loop: about five minutes.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.effects.application.test_training_loop_workers import _assert_same_average
from tests.unit.effects.application.training_run_support import RunPlan

pytestmark = pytest.mark.usefixtures("single_thread")

#: Steps 6 to 16 re-warm; epoch 2 ends at step 12.
REWARM = {"epochs": 3, "curriculum_rewarm_steps": 10}


def test_a_run_interrupted_inside_the_re_warm_resumes_to_the_same_weights(
    runner, tmp_path,
):
    whole = runner.run(tmp_path / "whole", RunPlan(workers=0, config=REWARM))
    out = tmp_path / "interrupted"
    first = runner.run(out, RunPlan(workers=0, interrupt_after=2, config=REWARM))
    assert first["average"]["start_step"] == 12
    second = runner.run(out, RunPlan(workers=0, resume=True, config=REWARM))

    assert first["losses"] + second["losses"] == whole["losses"]
    for name, value in whole["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    _assert_same_average(whole["average"], second["average"])
    # Epoch 3 moved the average away from the weights it started at.
    assert any(
        not torch.equal(value, whole["parameters"][f"{owner}.{name}"])
        for owner, tensors in whole["average"]["state"].items()
        for name, value in tensors.items()
    )


def test_a_resume_file_recording_no_re_warm_is_refused_by_name(
    runner, tmp_path,
):
    """A ``resume.pt`` written before the flag existed (D19)."""
    from effects.infrastructure.effect_model_store import (
        ResumeMismatchError,
        load_resume_state,
        save_resume_state,
    )

    out = tmp_path / "older"
    runner.run(out, RunPlan(workers=0, interrupt_after=1, config=REWARM))
    state = load_resume_state(out)
    del state["settings"]["curriculum_rewarm_steps"]
    save_resume_state(out, state)
    with pytest.raises(ResumeMismatchError, match="curriculum_rewarm_steps"):
        runner.run(out, RunPlan(workers=0, resume=True, config=REWARM))
